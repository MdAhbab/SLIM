"""
Tests for the dataset scripts under scripts/data/.

They check the parts whose mistakes would not be visible later: the track
binning against values worked out by hand, the BENGI pair construction on a
toy genome, the liftOver bookkeeping, and the rule that no test cell line may
reach a training file. Nothing here downloads anything.
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import build_pairs  # noqa: E402
import liftover  # noqa: E402
import prepare_tracks  # noqa: E402


def write_gz(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("".join(line + "\n" for line in lines))


# ---------------------------------------------------------------------------
# Track binning
# ---------------------------------------------------------------------------
def test_peak_binning_matches_hand_values(tmp_path):
    peak = tmp_path / "ctcf.bed.gz"
    # signalValue 10 over [250, 1250): half of bin 0, all of bin 1, half of bin 2.
    # A weaker peak inside bin 1 must not lower it.
    write_gz(peak, ["chr21\t250\t1250\t.\t0\t.\t10\t-1\t-1\t-1",
                    "chr21\t600\t700\t.\t0\t.\t2\t-1\t-1\t-1",
                    "chrUn_gl000220\t1\t50\t.\t0\t.\t5\t-1\t-1\t-1"])
    signals = prepare_tracks.bin_peaks(str(peak))
    a = np.arcsinh(10.0)
    assert signals["chr21"][0] == pytest.approx(a * 250 / 500)
    assert signals["chr21"][1] == pytest.approx(a)
    assert signals["chr21"][2] == pytest.approx(a * 250 / 500)
    assert signals["chr21"][3] == 0
    assert len(signals["chr21"]) == int(np.ceil(48129895 / 500)) + 1
    assert set(signals) == set(prepare_tracks.HG19)


def test_bigwig_binning_matches_hand_values(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    path = tmp_path / "signal.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr21", 48129895), ("chrY", 59373566)])
    bw.addEntries(["chr21", "chr21"], [0, 500], ends=[500, 750], values=[2.0, 4.0])
    bw.addEntries(["chrY"], [0], ends=[500], values=[9.0])
    bw.close()
    signals = prepare_tracks.bin_bigwig(str(path))
    assert signals["chr21"][0] == pytest.approx(np.arcsinh(2.0))
    # Bin 1 holds 4.0 over its first half only; the exact mean counts covered bases.
    assert signals["chr21"][1] == pytest.approx(np.arcsinh(4.0))
    assert signals["chr21"][2] == 0
    assert signals["chrY"].max() == 0          # skipped like the original
    assert signals["chr1"].max() == 0          # absent from the file


# ---------------------------------------------------------------------------
# BENGI pair construction
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_genome(tmp_path):
    tss = tmp_path / "tss.bed.gz"
    write_gz(tss, [
        "chr1\t100000\t100000\tENST1\t.\t+\tG1",
        "chr1\t100100\t100100\tENST1b\t.\t-\tG1",
        "chr1\t160000\t160000\tENST2\t.\t+\tG2",
        "chr1\t900000\t900000\tENST3\t.\t+\tG3",
        "chr1\t50000\t50000\tENST4\t.\t+\tG4",
    ])
    index = build_pairs.TSSIndex(tss)
    ccre = tmp_path / "ccre.bed.gz"
    write_gz(ccre, [
        "chr1\t130000\t130300\tEH1\t0\t.\t130000\t130300\t255,205,0",
        "chr1\t101000\t101200\tEHprox\t0\t.\t101000\t101200\t255,205,0",
        "chr1\t140000\t140300\tEH2\t0\t.\t140000\t140300\t255,0,0",
    ])
    return index, build_pairs.cell_enhancers(ccre, index)


def test_enhancers_are_distal_els_only(toy_genome):
    _, enh = toy_genome
    assert set(enh.coords) == {"EH1"}    # EHprox is within 2 kb of a TSS, EH2 is a promoter


def test_pairs_follow_bengi_rules(toy_genome):
    tss, enh = toy_genome
    loops = [(("chr1", 129000, 131000), ("chr1", 99000, 99500)),   # EH1 - G1
             (("chr1", 129000, 131000), ("chr2", 99000, 99500))]   # crosses chromosomes
    links = build_pairs.link_pairs(loops, enh, tss)
    assert links == [("EH1", "G1")]
    rows, report = build_pairs.build("TOY", links, enh, tss)
    by_gene = {r["fields"][9].split("|")[2]: r for r in rows}
    # cutoff = the one link's distance, 29,900 bp (EH1 start to the G1 TSS at 100,100)
    assert report["cutoff_bp"] == pytest.approx(29900)
    assert by_gene["G1"]["label"] == 1
    assert "G2" in by_gene and by_gene["G2"]["label"] == 0     # 29,700 bp away
    assert "G3" not in by_gene and "G4" not in by_gene         # beyond the cutoff
    fields = by_gene["G1"]["fields"]
    assert fields[5] == "chr1:130000-130300|TOY|EH1"
    # the closest TSS of G1 to the enhancer centre is the minus-strand one
    assert fields[9] == "chr1:100100-100101|TOY|G1|ENST1b|-"
    assert (fields[7], fields[8]) == ("99601", "101601")


def test_rows_parse_with_the_training_parser(toy_genome, tmp_path):
    sys.path.insert(0, str(ROOT))
    from src.bengi import read_rows
    tss, enh = toy_genome
    links = [("EH1", "G1")]
    rows, _ = build_pairs.build("TOY", links, enh, tss)
    path = tmp_path / "TOY.HiC-Benchmark.v3.tsv.gz"
    build_pairs.write_rows(rows, path)
    parsed = read_rows([str(path)])
    assert [r["label"] for r in parsed] == [r["label"] for r in rows]
    assert parsed[0]["cell"] == "TOY" and parsed[0]["assay"] == "HiC"


# ---------------------------------------------------------------------------
# liftOver bookkeeping
# ---------------------------------------------------------------------------
def test_lift_bedpe_keeps_loops_whose_anchors_stay_together(monkeypatch, tmp_path):
    def fake_liftover(cmd, check, capture_output):
        src, _chain, dst, _bad = cmd[1:5]
        with open(src) as handle, open(dst, "w") as out:
            for line in handle:
                chrom, start, end, row = line.split()
                row = int(row)
                if row == 1:                     # unmapped
                    continue
                if row == 2:                     # moves chromosome
                    chrom = "chr9"
                if row == 4:                     # split into two pieces
                    out.write(f"{chrom}\t{start}\t{int(start) + 10}\t{row}\n")
                out.write(f"{chrom}\t{int(start) + 100}\t{int(end) + 100}\t{row}\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(liftover.subprocess, "run", fake_liftover)
    monkeypatch.setattr(liftover, "liftover_binary", lambda: "liftOver")
    pairs = [(("chr1", 0, 1000), ("chr1", 5000, 6000)),
             (("chr1", 0, 1000), ("chr1", 7000, 8000)),
             (("chr2", 0, 1000), ("chr2", 7000, 8000))]
    # rows 0-2 are first anchors, 3-5 second anchors
    kept, report = liftover.lift_bedpe(pairs, chain=tmp_path / "x.chain")
    assert [i for i, _, _ in kept] == [0]
    assert kept[0][1] == ("chr1", 100, 1100)
    assert report["anchors"]["unmapped"] == 1
    assert report["anchors"]["moved_chrom"] == 1
    assert report["anchors"]["split"] == 1


# ---------------------------------------------------------------------------
# The test-cell rule
# ---------------------------------------------------------------------------
def test_audit_fails_on_a_test_cell_in_training(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    bengi = tmp_path / "bengi"
    row = ("1\t50000\tchr1\t130000\t130300\tchr1:130000-130300|{cell}|EH1\t"
           "chr1\t99601\t101601\tchr1:100100-100101|{cell}|ENSG1.1|ENST1|-")
    write_gz(bengi / "HMEC.HiC-Benchmark.v3.tsv.gz", [row.format(cell="HMEC")])
    write_gz(bengi / "NHEK.HiC-Benchmark.v3.tsv.gz", [row.format(cell="NHEK")])
    write_gz(out / "BENGI_extended" / "Bcell.HiC-Benchmark.v3.tsv.gz",
             [row.format(cell="HMEC").replace("EH1", "EH2")])
    env = dict(os.environ, SLIM_RAW=str(raw), SLIM_OUT=str(out))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "audit_new_data.py"),
         "--test-bengi", str(bengi), "--out", str(tmp_path / "audit.json")],
        env=env, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test cell lines ['HMEC']" in result.stdout
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["fatal"]


def test_audit_removes_repeats_of_test_pairs(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    bengi = tmp_path / "bengi"
    row = ("{label}\t50000\tchr1\t130000\t130300\tchr1:130000-130300|{cell}|{enh}\t"
           "chr1\t99601\t101601\tchr1:100100-100101|{cell}|ENSG1.1|ENST1|-")
    write_gz(bengi / "HMEC.HiC-Benchmark.v3.tsv.gz",
             [row.format(label=1, cell="HMEC", enh="EH1")])
    write_gz(bengi / "NHEK.HiC-Benchmark.v3.tsv.gz",
             [row.format(label=0, cell="NHEK", enh="EH9")])
    path = out / "BENGI_extended" / "Bcell.HiC-Benchmark.v3.tsv.gz"
    write_gz(path, [row.format(label=1, cell="Bcell", enh="EH1"),
                    row.format(label=0, cell="Bcell", enh="EH2")])
    env = dict(os.environ, SLIM_RAW=str(raw), SLIM_OUT=str(out))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "audit_new_data.py"),
         "--test-bengi", str(bengi), "--out", str(tmp_path / "audit.json")],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    remaining = gzip.open(path, "rt").read().splitlines()
    assert len(remaining) == 1 and "EH2" in remaining[0]

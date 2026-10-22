"""Tests of the extra test sets (scripts/data/build_extra_tests.py) and their
scoring (scripts/eval_extra.py), on small synthetic files."""

import gzip
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import build_extra_tests as build  # noqa: E402
import eval_extra  # noqa: E402


def bengi_line(label, cell, enh, start, end, tss, gene, transcript, chrom="chr1"):
    dist = abs((start + end) // 2 - tss)
    return (f"{label}\t{dist}\t{chrom}\t{start}\t{end}\t{chrom}:{start}-{end}|{cell}|{enh}\t"
            f"{chrom}\t{tss - 1000}\t{tss + 1000}\t{chrom}:{tss}-{tss + 1}|{cell}|{gene}.1|"
            f"{transcript}|+\n")


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.writelines(lines)


def make_data(tmp: Path):
    """Two enhancers in a training cell's file: E1 [1000, 1200) with genes A, B
    (B on two transcripts), E2 [5000, 5200) with genes C, D. A Javierre file
    for monocytes repeats one of them."""
    write(tmp / "pairs" / "BENGI" / "GM12878.HiC-Benchmark.v3.tsv.gz", [
        bengi_line(1, "GM12878", "E1", 1000, 1200, 20000, "GA", "T1"),
        bengi_line(0, "GM12878", "E1", 1000, 1200, 60000, "GB", "T2"),
        bengi_line(0, "GM12878", "E1", 1000, 1200, 61000, "GB", "T3"),
        bengi_line(0, "GM12878", "E2", 5000, 5200, 30000, "GC", "T4"),
        bengi_line(1, "GM12878", "E2", 5000, 5200, 90000, "GD", "T5"),
    ])
    write(tmp / "pairs" / "BENGI_extended" / "Monocyte.PCHiC_javierre2016-Benchmark.v3.tsv.gz", [
        bengi_line(1, "Monocyte", "E1", 1000, 1200, 20000, "GA", "T1"),
        bengi_line(0, "Monocyte", "E2", 5000, 5200, 30000, "GC", "T4"),
    ])
    (tmp / "pairs" / "BENGI_extra_test").mkdir(parents=True)
    # Study QTD000021 (monocytes): a variant in E1 for gene GB, one in E2 for
    # a gene that E2 is never paired with, one outside every enhancer.
    eqtl = tmp / "eval" / "eqtl"
    eqtl.mkdir(parents=True)
    with gzip.open(eqtl / "QTD000021.pip0.5.hg19.tsv.gz", "wt") as handle:
        handle.write("chrom\tpos\tgene_id\tpip\tcs_id\tvariant\n")
        handle.write("chr1\t1100\tGB\t0.9\tGB_L1\tchr1_1_A_C\n")
        handle.write("chr1\t5100\tGZ\t0.8\tGZ_L1\tchr1_2_A_C\n")
        handle.write("chr1\t9000\tGA\t0.7\tGA_L1\tchr1_3_A_C\n")


def test_variant_boundaries():
    # 0-based half-open enhancer [100, 200) holds 1-based positions 101..200.
    variants = {"chr1": ([100, 101, 200, 201], ["a", "b", "c", "d"])}
    assert build.genes_in(variants, "chr1", 100, 200, 0) == {"b", "c"}
    assert build.genes_in(variants, "chr1", 100, 200, 1) == {"a", "b", "c", "d"}
    assert build.genes_in(variants, "chr2", 100, 200, 0) == set()


def test_eqtl_rows_labels_and_cell(tmp_path):
    make_data(tmp_path)
    report = build.build_eqtl(tmp_path, tmp_path / "out", 0, build.Coverage(None),
                              seen={("E1", "GA")})
    stats = report["studies"]["QTD000021"]
    assert stats["cell"] == "Monocyte"
    assert stats["enhancers_with_variant"] == 2
    assert stats["enhancers_without_egene_candidate"] == 1     # E2: GZ is no candidate
    assert stats["enhancers"] == 1
    path = tmp_path / "out" / "eqtl_pairs" / "Monocyte.eQTL_QTD000021-Benchmark.v3.tsv.gz"
    rows = [l.split("\t") for l in gzip.open(path, "rt")]
    # E1's three transcripts (GA once, GB twice); GB is the eGene.
    assert sorted((r[9].split("|")[2].split(".")[0], r[0]) for r in rows) == [
        ("GA", "0"), ("GB", "1"), ("GB", "1")]
    assert all(r[5].split("|")[1] == "Monocyte" and r[9].split("|")[1] == "Monocyte" for r in rows)
    assert stats["rows_seen_in_training"] == 1


def test_javierre_keeps_unseen_pairs(tmp_path, monkeypatch):
    make_data(tmp_path)
    monkeypatch.setattr(build, "JAVIERRE_CELLS", ["Monocyte"])
    report = build.build_javierre(tmp_path, tmp_path / "out", build.Coverage(None),
                                  {("E1", "GA")})
    assert report["Monocyte"]["rows"] == 2
    assert report["Monocyte"]["unseen_rows"] == 1
    path = tmp_path / "out" / "javierre_unseen" / "Monocyte.PCHiC_javierre2016unseen-Benchmark.v3.tsv.gz"
    assert "GC" in gzip.open(path, "rt").read()


def test_eqtl_metrics_gene_level_and_top1():
    # Enhancer E1: genes GA (two transcripts) and GB; GA is the eGene. E2: GC
    # (eGene, far) and GD (near). The model ranks GA first in E1 and GC first
    # in E2; the nearest gene is GA in E1 and GD in E2.
    rows = {}
    for key, enh, gene, label, dist in [
            (("chr1", 1, 10), "E1", "GA", 1, 10), (("chr1", 1, 11), "E1", "GA", 1, 11),
            (("chr1", 1, 50), "E1", "GB", 0, 50),
            (("chr1", 2, 90), "E2", "GC", 1, 90), (("chr1", 2, 20), "E2", "GD", 0, 20)]:
        rows[key] = [{"enh_id": enh, "gene": gene, "label": label, "dist": dist}]
    scores = {("chr1", 1, 10): 0.2, ("chr1", 1, 11): 0.9, ("chr1", 1, 50): 0.5,
              ("chr1", 2, 90): 0.8, ("chr1", 2, 20): 0.1}
    pairs = eval_extra.gene_table(rows, scores, seen={("E2", "GD")})
    assert pairs[("E1", "GA")]["score"] == 0.9 and pairs[("E1", "GA")]["dist"] == 10
    m = eval_extra.eqtl_metrics(pairs)
    assert m["pairs"] == 4 and m["positives"] == 2
    assert m["top1"] == {"enhancers": 2, "model": 1.0, "nearest_gene": 0.5}
    assert m["model"]["auroc"] == 1.0
    # Unseen: E2-GD is seen, so E2 leaves the top-1 set and its pair the pool.
    assert m["unseen"]["pairs"] == 3 and m["unseen"]["top1"]["enhancers"] == 1


def test_summary_averages_seeds():
    per_run = {"ga_seed0": {"model": {"aupr": 0.4}}, "ga_seed1": {"model": {"aupr": 0.6}},
               "baseline_seed0": {"model": {"aupr": 0.3}}}
    s = eval_extra.summarise(per_run)
    assert s["ga"]["seeds"] == 2 and np.isclose(s["ga"]["model.aupr"][0], 0.5)
    assert s["baseline"]["model.aupr"] == [0.3, 0.0]

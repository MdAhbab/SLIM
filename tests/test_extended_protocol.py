"""
Tests for the extended training protocol (held-out validation cells, several
data folders, cell-balanced sampling, presence channels, mark dropout and
per-cell quantile normalisation).

Each switch is off by default, so the configurations of the earlier runs are
unaffected; the end-to-end tests in test_end_to_end.py cover that. These check
that each switch does what it says, on small files written here.
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from src.bengi import PairTable, deduplicate, read_rows, split_assay  # noqa: E402
from src.dataset import CellBalancedSampler, EPIDataset, shift_tracks  # noqa: E402
from src.epi_data_pipeline import EPIGenomicDataset  # noqa: E402

MARKS = ["CTCF", "DNase", "H3K27ac", "H3K27me3",
         "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]


def bengi_row(label, cell, enh, gene, start=200_000, tss=220_000, chrom="chr1"):
    return "\t".join(map(str, [
        label, float(abs(tss - start)), chrom, start, start + 300,
        f"{chrom}:{start}-{start + 300}|{cell}|{enh}", chrom, tss - 1499, tss + 501,
        f"{chrom}:{tss}-{tss + 1}|{cell}|{gene}.1|ENST1|+"]))


def write_gz(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------
def test_assay_and_source_come_from_the_file_name():
    assert split_assay("GM12878.HiC-Benchmark.v3.tsv.gz") == ("HiC", "")
    assert split_assay("GM12878.HiC_encode-Benchmark.v3.tsv.gz") == ("HiC", "encode")
    assert split_assay("H1.PCHiC_jung2019-Benchmark.v3.tsv.gz") == ("PCHiC", "jung2019")
    assert split_assay("HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz") == ("CTCF-ChIAPET", "")


def test_pair_table_reads_and_merges_like_the_dict_reader(tmp_path):
    a = tmp_path / "A.HiC-Benchmark.v3.tsv.gz"
    b = tmp_path / "A.CTCF-ChIAPET-Benchmark.v3.tsv.gz"
    write_gz(a, [bengi_row(0, "A", "E1", "G1"), bengi_row(1, "A", "E2", "G1", start=300_000),
                 bengi_row(0, "A", "E3", "G2", start=400_000)])
    # E1-G1 repeats with the other label; E3-G2 repeats with the same label.
    write_gz(b, [bengi_row(1, "A", "E1", "G1"), bengi_row(0, "A", "E3", "G2", start=400_000),
                 bengi_row(1, "A", "E4", "G3", start=500_000)])
    files = [str(a), str(b)]
    rows, report = deduplicate(read_rows(files), "union")
    table, table_report = PairTable.from_files(files).deduplicate("union")
    assert table_report == report
    assert report["pairs_with_conflicting_labels"] == 1
    for i, row in enumerate(rows):
        got = table.row(i)
        for key in ("label", "chrom", "enh_start", "enh_end", "enh_coord",
                    "tss_coord", "cell", "enh_id", "gene", "assay"):
            assert got[key] == row[key], (i, key)
    assert table.row(0)["assay"] == "CTCF-ChIAPET+HiC" and table.row(0)["label"] == 1
    assert table.pair_mask({("E2", "G1"), ("E9", "G9")}).tolist() == [False, True, False, False]
    assert table.pairs() == {("E1", "G1"), ("E2", "G1"), ("E3", "G2"), ("E4", "G3")}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_cell_balanced_sampler_shares_rows_fairly_and_repeats_by_epoch():
    cells = np.array(["small"] * 5 + ["mid"] * 50 + ["big"] * 500)
    sampler = CellBalancedSampler(cells, 90, seed=3)
    sampler.set_epoch(1)
    first = list(sampler)
    assert len(first) == len(set(first)) == 90
    drawn = cells[first]
    assert (drawn == "small").sum() == 5                 # every row of a small cell
    assert abs((drawn == "mid").sum() - (drawn == "big").sum()) <= 1
    sampler.set_epoch(1)
    assert list(sampler) == first                        # a resumed epoch is the same
    sampler.set_epoch(2)
    assert list(sampler) != first


# ---------------------------------------------------------------------------
# Tracks: presence channels, mark dropout, quantile normalisation
# ---------------------------------------------------------------------------
def make_cells(tmp_path, cells, scale=None, drop=None):
    """Toy track files and a feature list. `drop` maps a cell to marks it lacks."""
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.0, size=4_000).astype(np.float32)
    config = {"_location": str(tmp_path)}
    for cell in cells:
        config[cell] = {}
        for mark in MARKS:
            if mark in (drop or {}).get(cell, []):
                continue
            values = base * (scale or {}).get(cell, 1.0)
            torch.save({"chr1": torch.from_numpy(values.copy())},
                       tmp_path / f"{cell}_{mark}.pt")
            config[cell][mark] = f"{cell}_{mark}.pt"
    path = tmp_path / "features.json"
    path.write_text(json.dumps(config))
    return path


def toy_dataset(tmp_path, cells, **options):
    rows = [bengi_row(1, c, f"E{i}", f"G{i}", start=600_000, tss=640_000)
            for i, c in enumerate(cells)]
    bengi = tmp_path / "toy.HiC-Benchmark.v3.tsv.gz"
    write_gz(bengi, rows)
    return EPIGenomicDataset(bengi_paths=[str(bengi)], seq_len=100_000, bin_size=500,
                             **options)


def toy_config(presence, dropout=0.0):
    return {"data": {"sequence_length": 600, "epigenetic_bins": 200,
                     "n_epigenetic_features": 17 if presence else 9, "kmer_size": 3,
                     "modalities": "epi", "feats_order": MARKS,
                     "presence_channels": presence},
            "augmentation": {"mark_dropout": dropout, "epi_noise_std": 0.0,
                             "shift_max_bins": 3, "shift_mode": "tracks",
                             "rc_mode": "segment"}}


class _NoEncoder:
    k = 3


def test_presence_channels_mark_missing_tracks(tmp_path):
    feats = make_cells(tmp_path, ["A", "B"], drop={"B": ["CTCF"]})
    ds = toy_dataset(tmp_path, ["A", "B"], feats_config_path=str(feats),
                     presence_channels=True)
    a, b = ds[0]["epi"], ds[1]["epi"]
    assert a.shape == b.shape == (17, 200)
    assert torch.all(a[9:] == 1)                         # A has every track
    assert torch.all(b[9] == 0) and torch.all(b[10:] == 1)
    assert torch.all(b[1] == 0)                          # B's CTCF bins are zero


def test_mark_dropout_hides_tracks_and_their_presence(tmp_path):
    feats = make_cells(tmp_path, ["A"])
    ds = toy_dataset(tmp_path, ["A"], feats_config_path=str(feats), presence_channels=True)
    wrapped = EPIDataset(toy_config(True, dropout=1.0), _NoEncoder(), source_dataset=ds)
    wrapped.augment = True
    epi = wrapped[0]["epi"]
    assert torch.all(epi[1:] == 0)                       # every track and flag hidden
    assert torch.equal(epi[0], ds[0]["epi"][0])          # position channel kept
    with pytest.raises(ValueError):
        EPIDataset(toy_config(False, dropout=0.5), _NoEncoder(), source_dataset=ds)


def test_shift_leaves_presence_channels_alone():
    epi = torch.arange(3 * 10, dtype=torch.float).reshape(3, 10)
    epi[2] = 1.0                                         # one presence channel
    moved = shift_tracks(epi, 2, "tracks", n_tracks=1)
    assert torch.equal(moved[0], epi[0]) and torch.equal(moved[2], epi[2])
    assert torch.equal(moved[1, 2:], epi[1, :-2]) and torch.all(moved[1, :2] == 0)


def test_quantile_normalisation_puts_cells_on_one_scale(tmp_path):
    import track_quantiles
    feats = make_cells(tmp_path, ["A", "B"], scale={"B": 3.0})
    quantiles = {c: {m: track_quantiles.cell_quantiles(str(tmp_path / f"{c}_{m}.pt"))
                     for m in MARKS if m != "CTCF"} for c in ("A", "B")}
    qpath = tmp_path / "quantiles.json"
    qpath.write_text(json.dumps(quantiles))
    raw = toy_dataset(tmp_path, ["A", "B"], feats_config_path=str(feats),
                      normalize_epi=False)
    mapped = toy_dataset(tmp_path, ["A", "B"], feats_config_path=str(feats),
                         normalize_epi=False, track_quantiles=str(qpath),
                         quantile_reference_cells=["A"])
    dnase = MARKS.index("DNase") + 1
    assert torch.allclose(raw[1]["epi"][dnase], 3 * raw[0]["epi"][dnase])
    assert torch.allclose(mapped[1]["epi"][dnase], mapped[0]["epi"][dnase], atol=1e-3)
    ctcf = MARKS.index("CTCF") + 1                       # no quantiles: unchanged
    assert torch.equal(mapped[1]["epi"][ctcf], raw[1]["epi"][ctcf])


# ---------------------------------------------------------------------------
# Rows for held-out validation cells
# ---------------------------------------------------------------------------
def test_extended_rows_split_by_cell_and_drop_eval_pairs(tmp_path):
    import train
    write_gz(tmp_path / "T1.HiC-Benchmark.v3.tsv.gz",
             [bengi_row(1, "T1", "E1", "G1"), bengi_row(0, "T1", "E2", "G2", start=300_000),
              bengi_row(0, "T1", "E5", "G5", start=500_000)])
    write_gz(tmp_path / "T1.PCHiC_jung2019-Benchmark.v3.tsv.gz",
             [bengi_row(1, "T1", "E6", "G6", start=600_000)])
    write_gz(tmp_path / "V.HiC-Benchmark.v3.tsv.gz", [bengi_row(1, "V", "E1", "G1")])
    write_gz(tmp_path / "V.intactHiC-Benchmark.v3.tsv.gz",
             [bengi_row(1, "V", "E7", "G7", start=700_000)])
    test_file = tmp_path / "test" / "X.HiC_rao2014-Benchmark.v3.tsv.gz"
    write_gz(test_file, [bengi_row(1, "X", "E2", "G2", start=300_000)])
    config = {"data": {"train_assays": ["HiC", "PCHiC"], "dedup": "union",
                       "exclude_eval_pairs": True},
              "training": {"valid_cells": ["V"], "valid_assays": ["HiC"]},
              "paths": {"exclusion_files": [str(tmp_path / "test" / "*.tsv.gz")]}}
    files = sorted(str(p) for p in tmp_path.glob("*.tsv.gz"))
    table, train_idx, val_idx, report = train.extended_rows(config, files)
    kept = {table.row(i)["enh_id"] for i in train_idx}
    assert kept == {"E5", "E6"}            # E1 repeats validation, E2 repeats test
    assert [table.row(i)["enh_id"] for i in val_idx] == ["E1"]   # intact Hi-C left out
    assert report["rows_removed_as_eval_pairs"] == 2


def test_all_training_cells_never_include_held_out_ones(tmp_path):
    import train
    for cell in ("A", "B", "V", "HMEC", "HUVEC"):
        write_gz(tmp_path / f"{cell}.HiC-Benchmark.v3.tsv.gz", [bengi_row(1, cell, "E", "G")])

    class Args:
        train_cells = ["A"]
        test_cells = ["HMEC"]

    config = {"data": {"train_cells": "all", "never_train": ["HUVEC"]},
              "training": {"valid_cells": ["V"]}}
    assert train.training_cells(config, Args, [str(tmp_path)]) == ["A", "B"]


# ---------------------------------------------------------------------------
# A whole run under the extended protocol
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_training_runs_under_the_extended_protocol(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_end_to_end import run_training, write_config, write_dataset
    bengi_dir, feats = write_dataset(tmp_path / "data")
    config_path = tmp_path / "config.yaml"
    write_config(config_path, bengi_dir, feats, epochs=2)
    config = yaml.safe_load(config_path.read_text())
    config["paths"]["bengi_dirs"] = [str(bengi_dir)]
    config["paths"]["exclusion_files"] = [str(bengi_dir / "HMEC.*")]
    config["data"].update(train_cells="all", exclude_eval_pairs=True,
                          presence_channels=True, n_epigenetic_features=17)
    config["training"].update(valid_cells=["IMR90"], valid_samples=30,
                              samples_per_epoch=64, log_test_each_epoch=True)
    config["augmentation"]["mark_dropout"] = 0.2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = run_training(config_path, tmp_path / "run")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Validation on held-out cells: ['IMR90']" in result.stdout
    assert "Each epoch: 64 rows drawn from 3 training cells" in result.stdout
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 2 and history[0]["test_aupr"] is not None
    results = np.load(tmp_path / "run" / "eval_results.npz", allow_pickle=True)
    assert len(results["val_labels"]) == 30

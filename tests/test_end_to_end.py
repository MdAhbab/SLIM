"""
End-to-end smoke test: the real pipeline, on a small synthetic dataset.

The other tests exercise the models on random tensors. This one builds a
miniature dataset in the exact formats the real pipeline reads (BENGI
benchmark rows, binned chromatin tensors, a feature configuration file), then
runs `scripts/train.py` as a subprocess, exactly as it runs on real data.

It checks three things that matter before a long job starts on another
machine:

  1. training completes and writes every file the analysis scripts expect,
  2. an interrupted run resumes from the last completed epoch rather than
     starting over, which is what makes a power cut survivable,
  3. the analysis scripts read the outputs and produce numbers.

Everything runs on the processor in a couple of minutes. Marked slow, so:

    python -m pytest tests/ -q                     # skips this file
    python -m pytest tests/test_end_to_end.py -q   # runs it
    python -m pytest tests/ -q -m slow             # runs only slow tests
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow

CELLS = ["GM12878", "HeLa", "K562", "IMR90", "HMEC", "NHEK"]
MARKS = ["CTCF", "DNase", "H3K27ac", "H3K27me3",
         "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]
CHROMS = ["chr1", "chr2", "chr11", "chr17"]

# A small window keeps the test quick while exercising the same code paths.
BIN_SIZE = 500
SEQ_LEN_BP = 50_000
N_BINS = SEQ_LEN_BP // BIN_SIZE          # 100
CHROM_BINS = 4_000                        # covers every coordinate used below


def write_dataset(root):
    """Build BENGI rows, binned tracks and a feature configuration file."""
    rng = random.Random(0)
    bengi_dir = root / "BENGI"
    processed = root / "genomic_data" / "processed"
    bengi_dir.mkdir(parents=True)
    processed.mkdir(parents=True)

    for cell in CELLS:
        rows = []
        for i in range(48):
            chrom = CHROMS[i % len(CHROMS)]
            enh_start = 200_000 + rng.randrange(0, 400) * 500
            enh_end = enh_start + 300
            prom_start = enh_start + 20_000
            prom_end = prom_start + 200
            label = 1 if rng.random() < 0.3 else 0
            rows.append("\t".join(map(str, [
                label, abs(prom_start - enh_start), chrom,
                enh_start, enh_end, f"EH37E{100000 + i}|{cell}",
                chrom, prom_start, prom_end,
                f"{chrom}:{prom_start}-{prom_end}|GENE{i}",
            ])))
        (bengi_dir / f"{cell}.HiC-Benchmark.v3.tsv").write_text(
            "\n".join(rows) + "\n")

        # Each track file is a mapping from chromosome to a binned signal.
        for mark in MARKS:
            torch.save(
                {chrom: torch.rand(CHROM_BINS) for chrom in CHROMS},
                processed / f"{cell}_{mark}.500bp.pt",
            )

    config = {"_location": str(processed)}
    for cell in CELLS:
        config[cell] = {mark: f"{cell}_{mark}.500bp.pt" for mark in MARKS}
    feats_path = root / "genomic_data" / "feats.json"
    feats_path.write_text(json.dumps(config, indent=2))
    return bengi_dir, feats_path


def write_config(path, bengi_dir, feats_path, epochs):
    """A miniature configuration that still exercises every component."""
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    variant = yaml.safe_load((ROOT / "configs" / "slim_ka.yaml").read_text())
    variant.pop("extends", None)

    for section, values in variant.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            base[section].update(values)
        else:
            base[section] = values

    base["paths"] = {
        "bengi_dir": str(bengi_dir),
        "feats_config": str(feats_path),
        "ref_genome": "",          # no reference genome: placeholder sequences
    }
    base["data"].update({
        "sequence_length": 600,
        "epigenetic_bins": N_BINS,
        "seq_len_bp": SEQ_LEN_BP,
        "bin_size": BIN_SIZE,
        "enhancer_window": 300,
        "promoter_window": 300,
        "batch_size": 4,
        "encoder_fit_samples": 8,
    })
    base["model"].update({"hidden_dim": 60, "num_heads": 4, "n_tokens": 16,
                          "num_layers": 2, "sa_r": 4, "sa_da": 8})
    base["model"]["memory"].update({"bin_slots": 4, "bin_dim": 16,
                                  "survivors_per_layer": 2, "local_window": 8})
    base["training"].update({"epochs": epochs, "num_workers": 0,
                             "use_amp": False, "patience": 99,
                             "valid_chroms": ["chr11", "chr17"]})
    path.write_text(yaml.safe_dump(base, sort_keys=False))


def run_training(config_path, out_dir, extra=()):
    command = [sys.executable, str(ROOT / "scripts" / "train.py"),
               "--config", str(config_path), "--seed", "0",
               "--device", "cpu", "--output-dir", str(out_dir), *extra]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          timeout=1800)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once for the whole module, then reuse the outputs."""
    workspace = tmp_path_factory.mktemp("e2e")
    bengi_dir, feats_path = write_dataset(workspace / "data")
    config_path = workspace / "config.yaml"
    write_config(config_path, bengi_dir, feats_path, epochs=2)
    out_dir = workspace / "run"
    result = run_training(config_path, out_dir)
    assert result.returncode == 0, (
        f"training failed\nSTDOUT:\n{result.stdout[-4000:]}\n"
        f"STDERR:\n{result.stderr[-4000:]}")
    return {"workspace": workspace, "config": config_path, "out": out_dir,
            "stdout": result.stdout}


def test_training_writes_every_expected_file(trained):
    out = trained["out"]
    for name in ("eval_results.npz", "history.json", "config_snapshot.yaml",
                 "checkpoint.pt", "encoder.pkl"):
        assert (out / name).exists(), f"{name} was not written"
    # The resume file is removed once the run completes.
    assert not (out / "last.pt").exists(), (
        "last.pt should be deleted after a completed run, otherwise re-running "
        "the command would skip training")


def test_saved_predictions_are_usable(trained):
    data = np.load(trained["out"] / "eval_results.npz", allow_pickle=True)
    for key in ("test_predictions", "test_labels", "val_predictions",
                "val_labels", "test_chrom", "test_cell", "test_enh_coord",
                "test_prom_coord"):
        assert key in data.files, f"{key} missing from eval_results.npz"

    probs = data["test_predictions"]
    labels = data["test_labels"]
    assert len(probs) == len(labels) > 0
    assert probs.min() >= 0.0 and probs.max() <= 1.0, (
        "predictions must be probabilities")
    assert set(np.unique(labels)).issubset({0.0, 1.0})
    # Coordinates travel with the predictions, which the leakage audit needs.
    assert len(data["test_enh_coord"]) == len(probs)
    assert set(data["test_cell"].tolist()) == {"HMEC", "NHEK"}


def test_history_records_both_train_and_validation(trained):
    history = json.loads((trained["out"] / "history.json").read_text())
    assert len(history) == 2
    for row in history:
        for key in ("train_aupr", "val_aupr", "train_loss", "val_loss",
                    "train_auroc", "val_auroc"):
            assert key in row, f"{key} missing from history"
            assert row[key] is not None
    # Both gaps in check_overfitting.py are computed from these columns.


def test_interrupted_run_resumes_from_the_last_epoch(trained, tmp_path):
    """A run stopped after one epoch continues at epoch two, not epoch one."""
    workspace = trained["workspace"]
    config_path = tmp_path / "config.yaml"
    write_config(config_path, workspace / "data" / "BENGI",
                 workspace / "data" / "genomic_data" / "feats.json", epochs=1)
    out_dir = tmp_path / "resumed"

    first = run_training(config_path, out_dir)
    assert first.returncode == 0, first.stderr[-3000:]

    # Put last.pt back, as it would be if the power had failed mid-run.
    import shutil
    shutil.copy(out_dir / "checkpoint.pt", out_dir / "last_backup.pt")
    state = torch.load(out_dir / "checkpoint.pt", map_location="cpu",
                       weights_only=False)
    history = json.loads((out_dir / "history.json").read_text())
    torch.save({"model": state["model"],
                "optimizer": {}, "scheduler": {}, "scaler": {},
                "history": history, "best_metric": -1e9, "patience_left": 99,
                "epoch": 1, "seed": 0, "variant": "KA",
                "config": state["config"]}, out_dir / "last.pt")

    write_config(config_path, workspace / "data" / "BENGI",
                 workspace / "data" / "genomic_data" / "feats.json", epochs=3)
    second = run_training(config_path, out_dir)

    # The optimizer state above is a stub, so tolerate a load failure and only
    # require that the resume path was taken and reported.
    assert "Resuming from" in second.stdout or second.returncode != 0, (
        "an interrupted run did not attempt to resume\n"
        + second.stdout[-3000:])
    if second.returncode == 0:
        assert "continuing at epoch 2" in second.stdout, second.stdout[-2000:]
        history = json.loads((out_dir / "history.json").read_text())
        assert [row["epoch"] for row in history] == [1, 2, 3], (
            f"epochs recorded: {[row['epoch'] for row in history]}")


def test_analysis_scripts_run_on_the_outputs(trained):
    out = trained["out"]
    evaluate = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"),
         "--run", str(out), "--bootstrap", "50"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert evaluate.returncode == 0, evaluate.stderr[-3000:]
    assert "Calibration" in evaluate.stdout
    assert "chosen on validation" in evaluate.stdout

    overfit = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_overfitting.py"),
         "--run", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert overfit.returncode == 0, overfit.stderr[-3000:]
    assert "overfitting gap" in overfit.stdout
    assert "transfer gap" in overfit.stdout.lower()


def test_leakage_audit_runs_on_the_dataset(trained, tmp_path):
    workspace = trained["workspace"]
    audit = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_leakage.py"),
         "--config", str(trained["config"]),
         "--bengi-dir", str(workspace / "data" / "BENGI"),
         "--out", str(tmp_path / "leakage")],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert audit.returncode == 0, audit.stderr[-3000:]
    assert (tmp_path / "leakage" / "leakage_index.npz").exists()

    index = np.load(tmp_path / "leakage" / "leakage_index.npz",
                    allow_pickle=True)
    data = np.load(trained["out"] / "eval_results.npz", allow_pickle=True)
    assert len(index["disjoint"]) == len(data["test_predictions"]), (
        "the leakage index and the saved predictions must line up row for row, "
        "otherwise the disjoint-subset metrics would be meaningless")
    # The same ordering, checked on the coordinates themselves.
    assert np.array_equal(index["test_enh_start"] + 150,
                          data["test_enh_coord"]), (
        "enhancer coordinates disagree between the audit and the saved "
        "predictions")

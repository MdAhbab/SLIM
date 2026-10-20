"""
Tests for masked pretraining (scripts/pretrain.py) and fine-tuning from it
(train.py --init).
"""

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
sys.path.insert(0, str(ROOT / "tests"))

import pretrain  # noqa: E402


def test_spans_cover_the_requested_share():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = pretrain.span_mask(rng, 500, 0.15, 4)
        assert 75 <= mask.sum() <= 78
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0])))[::2]
        assert runs.min() >= 1


class _Windows:
    """Two windows of a position channel plus three tracks of 40 bins."""

    def __init__(self):
        torch.manual_seed(0)
        self.items = [torch.rand(4, 40) for _ in range(2)]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {"epi": self.items[i].clone()}


def test_hidden_values_are_marked_and_targets_come_from_the_originals():
    source = _Windows()
    windows = pretrain.MaskedWindows(source, n_tracks=3, pool=10, fraction=0.25,
                                     max_span=1, hide_mark=True, fixed_seed=7)
    item = windows[0]
    original = source.items[0]
    hidden_bins = item["hidden"].repeat_interleave(10, dim=1)
    tracks = item["epi"][1:]
    assert torch.all(tracks[hidden_bins] == pretrain.MASKED)
    assert torch.equal(tracks[~hidden_bins], original[1:][~hidden_bins])
    assert torch.equal(item["epi"][0], original[0])          # position channel kept
    assert torch.allclose(item["target"], original[1:].reshape(3, 4, 10).mean(dim=2))
    assert item["hidden"].all(dim=1).sum() >= 1              # one whole track hidden
    assert torch.equal(windows[0]["hidden"], item["hidden"])  # fixed per index


@pytest.mark.slow
def test_pretraining_then_fine_tuning_from_it(tmp_path):
    from test_end_to_end import run_training, write_config, write_dataset
    bengi_dir, feats = write_dataset(tmp_path / "data")
    config_path = tmp_path / "config.yaml"
    write_config(config_path, bengi_dir, feats, epochs=1)
    config = yaml.safe_load(config_path.read_text())
    config["paths"]["bengi_dirs"] = [str(bengi_dir)]
    config["data"].update(train_cells="all", modalities="epi")
    config["training"].update(valid_cells=["IMR90"], valid_samples=30,
                              samples_per_epoch=64)
    config["pretrain"] = {"windows_per_epoch": 32, "epochs": 2, "monitor_windows": 16,
                          "mask_fraction": 0.2, "max_span": 2}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    for variant in ("KA", "baseline"):
        out = tmp_path / f"pre_{variant}"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "pretrain.py"), "--config",
             str(config_path), "--variant", variant, "--output-dir", str(out),
             "--device", "cpu"], cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert "chromatin of 3 epigenomes" in result.stdout     # not IMR90, HMEC, NHEK
        saved = torch.load(out / "pretrained.pt", weights_only=False)
        assert saved["epoch"] == 2 and len(saved["history"]) == 2
        assert saved["corpus_cells"] == ["GM12878", "HeLa", "K562"]
        assert all(k.startswith(pretrain.PRETRAINED_PREFIXES) for k in saved["model"])

    result = run_training(config_path, tmp_path / "run",
                          extra=("--init", str(tmp_path / "pre_KA" / "pretrained.pt")))
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Initialised" in result.stdout
    wrong = run_training(config_path, tmp_path / "run_wrong",
                         extra=("--init", str(tmp_path / "pre_baseline" / "pretrained.pt")))
    assert wrong.returncode != 0 and "variant baseline" in wrong.stdout + wrong.stderr

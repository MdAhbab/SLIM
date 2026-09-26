"""
Smoke tests: every model builds, runs and produces usable numbers.

These use random tensors of the true input shapes, so they need neither the
dataset nor a GPU and finish in a couple of minutes on a laptop. Run them
before pushing, and on any machine before starting a long training job.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.slim_model import build_epi_channel_mask, build_model
from src.metrics import (
    bootstrap_ci, expected_calibration_error, metrics_at_threshold,
    select_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "baseline": "configs/baseline.yaml",
    "A": "configs/slim_a.yaml",
    "KA": "configs/slim_ka.yaml",
    "GA": "configs/slim_ga.yaml",
    "KA_legacy": "configs/slim_ka_legacy.yaml",
}
BATCH = 2


def small_config(name):
    """Load a config and shrink the window so the test runs quickly."""
    config = load_config(str(ROOT / CONFIGS[name]))
    config["data"]["sequence_length"] = 600
    config["data"]["epigenetic_bins"] = 500
    return config


def random_batch(config):
    seq_len = config["data"]["sequence_length"]
    n_bins = config["data"]["epigenetic_bins"]
    n_feats = config["data"]["n_epigenetic_features"]
    return (
        torch.randn(BATCH, 64, seq_len - 2),
        torch.rand(BATCH, n_feats, n_bins),
        torch.randint(0, n_bins, (BATCH, 1)).float(),
        torch.randint(0, n_bins, (BATCH, 1)).float(),
    )


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_forward_pass_shapes(name):
    config = small_config(name)
    model = build_model(config).eval()
    with torch.no_grad():
        out = model(*random_batch(config))

    cls_out, reg_out, attention = out[0], out[1], out[2]
    assert cls_out.shape == (BATCH, 1)
    assert reg_out.shape == (BATCH, 1)
    assert torch.isfinite(cls_out).all(), "classification logits are not finite"
    assert torch.isfinite(reg_out).all(), "distance output is not finite"

    penalty = model.attention_penalty(attention)
    assert torch.isfinite(penalty), "the pooling penalty is not finite"

    if name != "baseline":
        aux = out[3]
        for key in ("gate_mean", "gate_entropy", "slot_diversity_penalty"):
            assert key in aux, f"{key} missing from the auxiliary terms"
            assert torch.isfinite(aux[key]), f"{key} is not finite"
        assert torch.isfinite(model.memory_auxiliary_loss(aux))


@pytest.mark.parametrize("name", ["A", "KA", "GA", "KA_legacy"])
def test_backward_pass_reaches_every_parameter(name):
    """A gradient must reach every trainable weight, including the memory."""
    config = small_config(name)
    model = build_model(config).train()
    out = model(*random_batch(config))
    loss = out[0].sum() + out[1].sum() + model.attention_penalty(out[2])
    loss = loss + model.memory_auxiliary_loss(
        out[3], prediction_weight=config["training"]["prediction_loss_weight"])
    loss.backward()

    without_gradient = [n for n, p in model.named_parameters()
                        if p.requires_grad and p.grad is None]
    assert not without_gradient, (
        f"no gradient reached: {without_gradient[:10]}")


def test_write_log_is_available_and_in_range():
    config = small_config("KA")
    model = build_model(config).eval()
    with torch.no_grad():
        out = model(*random_batch(config), return_trace=True)
    trace = out[-1]
    assert len(trace["tokens"]) == model.memory_config.num_layers
    assert len(trace["slots"]) == model.memory_config.num_layers
    n_tokens = model.n_seq_tokens + model.n_epi_tokens
    for tokens, slots in zip(trace["tokens"], trace["slots"]):
        assert tokens.shape[0] == BATCH
        assert int(tokens.min()) >= 0
        assert int(tokens.max()) < n_tokens
        assert slots.shape == tokens.shape
        assert int(slots.max()) < model.memory_config.bin_slots


def test_sequence_only_hides_every_chromatin_channel():
    config = small_config("KA")
    config["data"]["modalities"] = "seq"
    model = build_model(config).eval()
    assert float(model.epi_channel_mask.sum()) == 0.0

    seq, epi, enh, prom = random_batch(config)
    with torch.no_grad():
        first = model(seq, epi, enh, prom)[0]
        second = model(seq, torch.rand_like(epi) * 9.0, enh, prom)[0]
    assert torch.allclose(first, second), (
        "changing the chromatin input changed the output in sequence-only "
        "mode, so the mask is not being applied")


def test_sequence_plus_position_keeps_only_the_position_channel():
    config = small_config("KA")
    config["data"]["modalities"] = "seq+pos"
    model = build_model(config).eval()
    mask = model.epi_channel_mask.view(-1)
    assert float(mask[0]) == 1.0
    assert float(mask[1:].sum()) == 0.0


def test_channel_mask_can_keep_named_tracks():
    mask = build_epi_channel_mask(9, "all", keep_tracks=["CTCF", "DNase"])
    assert mask.view(-1).tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        build_epi_channel_mask(9, "all", keep_tracks=["NotATrack"])


def test_gate_terms_can_be_switched_off_individually():
    for off in ("use_learned_score", "use_novelty", "use_prediction_error"):
        config = small_config("KA")
        config["model"]["memory"][off] = False
        model = build_model(config).eval()
        with torch.no_grad():
            out = model(*random_batch(config))
        assert torch.isfinite(out[0]).all(), f"turning off {off} broke the model"


def test_every_gate_term_off_is_rejected():
    config = small_config("KA")
    for key in ("use_learned_score", "use_novelty", "use_prediction_error"):
        config["model"]["memory"][key] = False
    with pytest.raises(ValueError):
        build_model(config)


def test_threshold_selection_beats_the_fixed_threshold():
    rng = np.random.default_rng(0)
    labels = (rng.random(3000) < 0.11).astype(int)
    probs = np.clip(rng.normal(0.25 + 0.3 * labels, 0.2), 1e-4, 1 - 1e-4)

    threshold, value = select_threshold(labels, probs, metric="mcc")
    assert 0.0 < threshold < 1.0
    at_fixed = metrics_at_threshold(labels, probs, 0.5)["mcc"]
    assert value >= at_fixed - 1e-9, (
        "the selected threshold scores worse than 0.5, which cannot happen "
        "when 0.5 is inside the search grid")


def test_calibration_and_bootstrap_return_sane_values():
    rng = np.random.default_rng(1)
    labels = (rng.random(2000) < 0.2).astype(int)
    probs = np.clip(rng.normal(0.3 + 0.3 * labels, 0.15), 1e-4, 1 - 1e-4)

    calibration = expected_calibration_error(labels, probs, n_bins=10)
    assert 0.0 <= calibration["ece"] <= 1.0
    assert len(calibration["bins"]) == 10

    interval = bootstrap_ci(labels, probs, metric="aupr", n_resamples=100)
    assert interval["lower"] <= interval["point"] <= interval["upper"]


def test_config_inheritance_merges_without_losing_defaults():
    merged = load_config(str(ROOT / "configs/slim_ga.yaml"))
    base = load_config(str(ROOT / "configs/base.yaml"))
    assert merged["model"]["variant"] == "GA"
    assert merged["training"]["epochs"] == base["training"]["epochs"]
    assert merged["data"]["n_epigenetic_features"] == 9
    assert merged["model"]["memory"]["bin_slots"] == 16

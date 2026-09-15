"""
Guard the claim that the variants differ only in the feed-forward sublayer.

The manuscript compares variants A, KA and GA and attributes the differences
between them to the feed-forward sublayer alone. An earlier version of this
code kept the three models in three files, and they drifted: one of them ended
up with different pooling, a different pooled width and different prediction
heads, so its comparison measured four changes at once rather than one.

These tests make that failure impossible to reintroduce quietly. They compare
the models by structure rather than by reading the source, so any future edit
that touches one variant and not the others fails here.

The reference parameter count pins the current KA model to the one that
produced the published results. It was taken from the archived implementation
and confirmed to give bit-identical outputs under the same weights.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.slim_model import build_model

# Parameter count of the KA model that produced the published cross-cell
# results. Changing this number means changing the published architecture.
KA_REFERENCE_PARAMETERS = 4_251_155
BASELINE_REFERENCE_PARAMETERS = 3_529_796

VARIANT_CONFIG = {
    "A": "configs/slim_a.yaml",
    "KA": "configs/slim_ka.yaml",
    "GA": "configs/slim_ga.yaml",
}
ROOT = Path(__file__).resolve().parents[1]


def build(variant):
    return build_model(load_config(str(ROOT / VARIANT_CONFIG[variant])))


def shapes_outside_ffn(model):
    """Every parameter that is not part of the feed-forward sublayer."""
    return {name: tuple(param.shape)
            for name, param in model.named_parameters()
            if ".ffn." not in name}


def test_ka_matches_the_published_architecture():
    model = build("KA")
    total = sum(p.numel() for p in model.parameters())
    assert total == KA_REFERENCE_PARAMETERS, (
        f"KA now has {total:,} parameters but the published model had "
        f"{KA_REFERENCE_PARAMETERS:,}. If this change is deliberate, the "
        f"published results no longer describe this model and must be "
        f"regenerated.")


def test_baseline_matches_the_published_architecture():
    model = build_model(load_config(str(ROOT / "configs/baseline.yaml")))
    total = sum(p.numel() for p in model.parameters())
    assert total == BASELINE_REFERENCE_PARAMETERS, (
        f"The baseline now has {total:,} parameters but the published model "
        f"had {BASELINE_REFERENCE_PARAMETERS:,}.")


def test_variants_agree_outside_the_feed_forward_sublayer():
    """A, KA and GA must share every parameter outside the feed-forward layer."""
    reference = shapes_outside_ffn(build("KA"))
    for variant in ("A", "GA"):
        other = shapes_outside_ffn(build(variant))
        assert set(other) == set(reference), (
            f"variant {variant} and KA have different parameters outside the "
            f"feed-forward sublayer.\n"
            f"  only in {variant}: {sorted(set(other) - set(reference))}\n"
            f"  only in KA: {sorted(set(reference) - set(other))}")
        mismatched = {k: (reference[k], other[k])
                      for k in reference if reference[k] != other[k]}
        assert not mismatched, (
            f"variant {variant} and KA share parameter names whose shapes "
            f"differ outside the feed-forward sublayer: {mismatched}")


def test_only_the_feed_forward_sublayer_differs_in_size():
    """The variants' parameter counts may differ only by their feed-forward layers."""
    counts = {}
    for variant in VARIANT_CONFIG:
        model = build(variant)
        total = sum(p.numel() for p in model.parameters())
        ffn = sum(p.numel() for name, p in model.named_parameters()
                  if ".ffn." in name)
        counts[variant] = (total, ffn)
    baselines = {variant: total - ffn for variant, (total, ffn) in counts.items()}
    assert len(set(baselines.values())) == 1, (
        f"the variants differ outside the feed-forward sublayer: {baselines}")


def test_gated_and_rectified_layers_hold_comparable_capacity():
    """A and GA are sized to match, so their comparison is not about capacity."""
    sizes = {}
    for variant in ("A", "GA"):
        model = build(variant)
        sizes[variant] = sum(p.numel() for name, p in model.named_parameters()
                             if ".ffn." in name)
    larger, smaller = max(sizes.values()), min(sizes.values())
    assert (larger - smaller) / larger < 0.01, (
        f"the rectified and gated feed-forward layers differ by more than one "
        f"percent in size: {sizes}. A comparison between them would confound "
        f"the layer type with its capacity.")


@pytest.mark.parametrize("variant", sorted(VARIANT_CONFIG))
def test_pooled_width_is_shared(variant):
    """Every variant pools into the same 720-dimensional vector."""
    model = build(variant)
    assert model.fc_linear.in_features == 4 * model.d_model


def test_the_baseline_has_no_survival_gate():
    """The baseline must remain a genuine global-attention control."""
    model = build_model(load_config(str(ROOT / "configs/baseline.yaml")))
    assert not hasattr(model, "encoder"), (
        "the baseline should carry the global KAN-Transformer, not the "
        "survival-gated encoder")

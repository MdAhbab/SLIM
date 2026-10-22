"""
Tests for the Phase 2 switches: the augmentation noise off the position
channel (R0), runs without the language-model table, the chromatin groups
(R4), the loop-extrusion module (R3) and the sequence path (R2), all in
src/pair_modules.py, and their wiring into SLIM and the baseline.
"""

import copy
import hashlib
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from src.config import load_config  # noqa: E402
from src.dataset import EPIDataset  # noqa: E402
from src.pair_modules import (SequencePairPath, build_epi_stem,  # noqa: E402
                              build_extrusion, chromatin_groups)
from src.slim_model import build_model  # noqa: E402

TRACKS = ["CTCF", "DNase", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
          "H3K4me3", "H3K9me3", "CTCFmotif_plus", "CTCFmotif_minus",
          "YY1motif_plus", "YY1motif_minus"]
GROUPS = ["accessibility", "active", "promoter", "transcription", "repressive",
          "architecture"]
NEW_MODULES = ("epi_cnn.", "pair_features.", "fc_extra.")

# Parameter counts and SHA-256 of the ordered state-dict keys and shapes,
# recorded before the Phase 2 switches existed (commit a701e83). With every
# switch off these configurations must build exactly that model; phase2_r0
# changes the augmentation only.
PINNED = {
    ("phase2_dna_both.yaml", "GA"):
        (4_333_055, "0252713a640855cfed03d9774255aee66b560891758e976b10099d30ab944428"),
    ("phase2_dna_both.yaml", "baseline"):
        (3_470_036, "b01271acbf481469f7be981e2a7664813c061540a14cef07f21c075f64e07af5"),
    ("phase2_p1a.yaml", "GA"):
        (4_392_815, "bf426321c511e879418a887b71849be37b6fd18582705780a5b9504c54780d26"),
    ("phase2_p1a.yaml", "baseline"):
        (3_529_796, "43f93dad553c366f1180ab7cf9fe214b724b712f5d4665a76137e82adfaa4212"),
    ("phase2_r0.yaml", "GA"):
        (4_333_055, "0252713a640855cfed03d9774255aee66b560891758e976b10099d30ab944428"),
    ("phase2_r0.yaml", "baseline"):
        (3_470_036, "b01271acbf481469f7be981e2a7664813c061540a14cef07f21c075f64e07af5"),
}


def config_for(name, variant=None, **model):
    config = load_config(str(ROOT / "configs" / name))
    if variant is not None:
        config["model"]["variant"] = variant
    config["model"].update(model)
    return config


def small(config, bins=500):
    """A shorter chromatin window, so the full models run quickly."""
    config["data"]["epigenetic_bins"] = bins
    return config


def batch(config, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = config["data"]
    if data.get("dna_input") == "embedding":
        seq = torch.randn(2, 2 * data["dna_segments"], data["dna_embedding_dim"], generator=g)
    else:
        seq = torch.randn(2, 64, data["sequence_length"] - 2, generator=g)
    epi = torch.rand(2, data["n_epigenetic_features"], data["epigenetic_bins"], generator=g)
    return seq, epi, torch.tensor([[100.0], [260.0]]), torch.tensor([[180.0], [200.0]])


def key_hash(model):
    keys = "\n".join(f"{k}:{tuple(v.shape)}" for k, v in model.state_dict().items())
    return hashlib.sha256(keys.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dataset: noise off the position channel, no table without the DNA branch
# ---------------------------------------------------------------------------

class SignedPair:
    """One pair whose position channel runs from -3 to 3, as outside and
    between the anchors."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.epi = torch.cat([torch.linspace(-3, 3, 20)[None],
                              torch.rand(8, 20, generator=g)])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"epi": self.epi.clone(), "enhancer_seq": "AAACCC",
                "promoter_seq": "GGGTTA", "chrom": "chr1",
                "label": torch.tensor([1.0]), "dist": torch.tensor([1.0]),
                "enh_idx": torch.tensor([5.0]), "prom_idx": torch.tensor([9.0])}


class ZeroEncoder:
    k = 3

    def transform(self, sequence, chrom=None):
        return torch.zeros(64, len(sequence) - 2)


def noisy_dataset(**augmentation):
    config = {"data": {"sequence_length": 12, "epigenetic_bins": 20,
                       "n_epigenetic_features": 9},
              "augmentation": {"rc_prob": 0.0, "epi_noise_std": 0.5,
                               "shift_max_bins": 0, **augmentation}}
    dataset = EPIDataset(config, ZeroEncoder(), source_dataset=SignedPair())
    dataset.augment = True
    return dataset


def test_track_noise_leaves_the_position_channel_untouched():
    source = SignedPair().epi
    epi = noisy_dataset(noise_mode="tracks")[0]["epi"]
    assert torch.equal(epi[0], source[0])                 # negative values kept
    assert not torch.allclose(epi[1:], source[1:])
    assert (epi[1:] >= 0).all()


@pytest.mark.parametrize("augmentation", [{}, {"noise_mode": "all"}])
def test_the_original_noise_rule_is_unchanged(augmentation):
    source = SignedPair().epi
    torch.manual_seed(5)
    epi = noisy_dataset(**augmentation)[0]["epi"]
    torch.manual_seed(5)
    expected = (source + torch.randn_like(source) * 0.5).clamp(min=0)
    assert torch.equal(epi, expected)
    assert (epi[0] >= 0).all()           # the clamp removes the negative positions


def test_noise_mode_is_checked():
    with pytest.raises(ValueError, match="noise_mode"):
        noisy_dataset(noise_mode="position")


def test_without_the_dna_branch_no_embedding_table_is_read(tmp_path):
    config = {"data": {"sequence_length": 12, "epigenetic_bins": 20,
                       "n_epigenetic_features": 9, "modalities": "epi",
                       "dna_input": "embedding", "dna_segments": 4,
                       "dna_embedding_dim": 16},
              "paths": {"dna_embeddings": str(tmp_path / "absent")},
              "augmentation": {}}
    dataset = EPIDataset(config, None, source_dataset=SignedPair())
    item = dataset[0]
    assert dataset.embeddings is None
    assert item["seq"].shape == (8, 16) and not item["seq"].any()
    config["data"]["modalities"] = "all"              # now the table is needed
    with pytest.raises(FileNotFoundError):
        EPIDataset(config, None, source_dataset=SignedPair())


# ---------------------------------------------------------------------------
# Chromatin groups
# ---------------------------------------------------------------------------

def group_config(feats_order, groups=None, presence=False):
    n = len(feats_order)
    config = {"data": {"feats_order": list(feats_order), "presence_channels": presence,
                       "n_epigenetic_features": 1 + n * (2 if presence else 1)},
              "model": {"chromatin_stem": "groups"}}
    if groups is not None:
        config["model"]["chromatin_groups"] = groups
    return config


def test_chromatin_groups_follow_the_feature_list():
    groups = chromatin_groups(group_config(TRACKS[:10]))      # without YY1
    assert list(groups) == GROUPS
    assert groups["accessibility"] == [0, 2]                  # position, DNase
    assert groups["architecture"] == [0, 1, 9, 10]            # CTCF and its motifs
    assert list(chromatin_groups(group_config(["CTCF", "DNase"]))) == [
        "accessibility", "architecture"]                      # empty groups dropped
    assert chromatin_groups(group_config(["CTCF", "DNase"], presence=True)) == {
        "accessibility": [0, 2, 4], "architecture": [0, 1, 3]}


def test_group_errors():
    with pytest.raises(ValueError, match="no group"):
        build_epi_stem(group_config(TRACKS[:8] + ["ZNF143motif_plus"]), 16, 10)
    with pytest.raises(ValueError, match="both"):
        chromatin_groups(group_config(["CTCF"], {"a": ["CTCF"], "b": ["CTCF"]}))
    wrong = group_config(TRACKS[:8])
    wrong["data"]["n_epigenetic_features"] = 12
    with pytest.raises(ValueError, match="n_epigenetic_features"):
        build_epi_stem(wrong, 16, 10)
    with pytest.raises(ValueError, match="chromatin_stem"):
        build_epi_stem(dict(wrong, model={"chromatin_stem": "many"}), 16, 10)


def test_the_gate_starts_at_the_sum_of_the_stems_and_weights_sum_to_one():
    torch.manual_seed(0)
    stem = build_epi_stem(group_config(TRACKS), 16, 10).eval()
    epi = torch.rand(2, 13, 100)
    assert len(stem.stems) == 6
    assert torch.allclose(stem(epi), sum(stem.stem_outputs(epi)), atol=1e-6)
    assert torch.allclose(stem.last_alpha, torch.full((2, 6, 10), 1 / 6))
    with torch.no_grad():
        stem.gate.weight.normal_()
        stem.gate.bias.normal_()
    out = stem(epi)
    assert torch.allclose(stem.last_alpha.sum(dim=1), torch.ones(2, 10))
    assert not torch.allclose(out, sum(stem.stem_outputs(epi)), atol=1e-3)


# ---------------------------------------------------------------------------
# Loop extrusion
# ---------------------------------------------------------------------------

def extrusion_config(**extrusion):
    return {"data": {"feats_order": TRACKS, "n_epigenetic_features": 13, "bin_size": 500},
            "model": {"extrusion": {"enabled": True, **extrusion}}}


def make_extrusion(seed=0, bias=None, **extrusion):
    torch.manual_seed(seed)
    module = build_extrusion(extrusion_config(**extrusion))
    if bias is not None:
        with torch.no_grad():
            module.hazard_net[2].bias.fill_(bias)
    return module


def brute_force(module, epi, enh, prom):
    """The extrusion features by direct loops over the bins, in float64."""
    module = copy.deepcopy(module).double()
    x = epi.double()[:, module.channels]
    batch_size, _, n = x.shape
    k_heads, m, reach = module.heads, module.margin_bins, module.anchor_pool_bins
    with torch.no_grad():
        base = F.softplus(module.beta)
        if module.barriers:
            learned_l = F.softplus(module.hazard_net(x))
            learned_r = F.softplus(module.hazard_net(x[:, module.swap].flip(-1))).flip(-1)
        else:
            learned_l = learned_r = torch.zeros(batch_size, k_heads, n, dtype=torch.float64)
    features = torch.zeros(batch_size, 7, k_heads, dtype=torch.float64)
    for i in range(batch_size):
        a, b = sorted((int(enh[i]), int(prom[i])))
        for k in range(k_heads):
            def hazard(learned):
                return [float(base[k]) + (0.0 if abs(j - a) <= m or abs(j - b) <= m
                                          else float(learned[i, k, j])) for j in range(n)]
            h_l, h_r = hazard(learned_l), hazard(learned_r)
            arrive = [math.exp(-sum(h_l[a + 1:l]) - sum(h_r[l + 1:b])) for l in range(a + 1, b)]
            log_q = math.log(sum(arrive) / len(arrive)) if arrive else 0.0

            def strength(learned, c):
                return max(float(learned[i, k, j])
                           for j in range(max(0, c - reach), min(n - 1, c + reach) + 1))
            values = [log_q, sum(h_l[a + 1:b]), sum(h_r[a + 1:b]),
                      strength(learned_l, a), strength(learned_r, a),
                      strength(learned_l, b), strength(learned_r, b)]
            for f, value in enumerate(values):
                features[i, f, k] = value
    return features.view(batch_size, -1)


ENH = torch.tensor([[10.0], [45.0], [0.0], [20.0], [30.0], [2.0]])
PROM = torch.tensor([[40.0], [5.0], [59.0], [21.0], [30.0], [8.0]])


@pytest.mark.parametrize("bias", [None, 0.0])
@pytest.mark.parametrize("barriers", [True, False])
def test_extrusion_matches_a_brute_force_loop(bias, barriers):
    module = make_extrusion(bias=bias if barriers else None, barriers=barriers)
    epi = torch.rand(6, 13, 60, generator=torch.Generator().manual_seed(2))
    expected = brute_force(module, epi, ENH, PROM)
    exact = copy.deepcopy(module).double().terms(epi.double(), ENH, PROM)["features"]
    assert torch.allclose(exact, expected, rtol=1e-10, atol=1e-10)
    single = module.terms(epi, ENH, PROM)["features"]
    assert single.dtype == torch.float32
    assert torch.allclose(single.double(), expected, rtol=1e-4, atol=1e-4)


def test_extrusion_starts_at_the_configured_processivity_and_swaps_strands():
    module = make_extrusion(processivity_kb=300)
    assert torch.allclose(module.processivity_kb, torch.tensor([300.0, 300.0]))
    read = [TRACKS[c - 1] for c in module.channels.tolist()]
    assert read == ["CTCF", "CTCFmotif_plus", "CTCFmotif_minus", "YY1motif_plus",
                    "YY1motif_minus", "DNase", "H3K4me3", "H3K36me3"]
    assert [read[j] for j in module.swap.tolist()] == [
        "CTCF", "CTCFmotif_minus", "CTCFmotif_plus", "YY1motif_minus",
        "YY1motif_plus", "DNase", "H3K4me3", "H3K36me3"]
    with pytest.raises(ValueError, match="both strands"):
        build_extrusion(extrusion_config(tracks=["CTCFmotif_plus"]))
    with pytest.raises(ValueError, match="none of its tracks"):
        build_extrusion(extrusion_config(tracks=["ZNF143"]))
    assert not build_extrusion(extrusion_config(tracks=["ZNF143"], barriers=False)).barriers


def test_a_barrier_between_the_anchors_lowers_the_contact_term():
    module = make_extrusion(tracks=["CTCF"])
    with torch.no_grad():                 # hazard rises with CTCF, at its own bin only
        first, second = module.hazard_net[0], module.hazard_net[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[:, 0, 1] = 1.0
        second.weight.fill_(1.0)
        second.bias.fill_(-8.0)
    a, b = 15, 45

    def terms(site=None):
        epi = torch.zeros(1, 13, 60)
        if site is not None:
            epi[0, 1, site] = 5.0                          # CTCF is channel 1
        return module.terms(epi, torch.tensor([[a]]), torch.tensor([[b]]))

    free = terms()
    inside = terms(30)
    assert (inside["log_q"] < free["log_q"] - 1).all()
    assert (inside["total_left"] > free["total_left"] + 10).all()
    for site in (5, 52, a + 2, b - 3):                     # outside, or in the margin
        assert torch.allclose(terms(site)["log_q"], free["log_q"])


def mirror_channels():
    """Channel order with the two strands of every stranded track exchanged."""
    order = list(range(13))
    for name in TRACKS:
        if name.endswith("_plus"):
            i, j = 1 + TRACKS.index(name), 1 + TRACKS.index(name[:-5] + "_minus")
            order[i], order[j] = j, i
    return order


def test_extrusion_is_strand_equivariant_and_ignores_the_pair_order():
    module = make_extrusion(bias=0.0)
    n = 80
    epi = torch.rand(3, 13, n, generator=torch.Generator().manual_seed(1))
    enh, prom = torch.tensor([[10.0], [70.0], [30.0]]), torch.tensor([[50.0], [20.0], [33.0]])
    blocks = module.terms(epi, enh, prom)["features"].view(3, 7, -1)
    mirrored = epi[:, mirror_channels()].flip(-1)
    other = module.terms(mirrored, n - 1 - enh, n - 1 - prom)["features"].view(3, 7, -1)
    assert torch.allclose(other[:, 0], blocks[:, 0], atol=1e-4)      # logQ unchanged
    # T^L <-> T^R; and the left anchor's strengths (left leg, right leg)
    # become the right anchor's with the legs exchanged, so the four
    # strengths come in reverse order.
    assert torch.allclose(other[:, 1:3], blocks[:, 1:3].flip(1), atol=1e-4)
    assert torch.allclose(other[:, 3:], blocks[:, 3:].flip(1), atol=1e-4)
    assert torch.equal(module(epi, prom, enh), module(epi, enh, prom))


@pytest.mark.parametrize("barriers", [True, False])
def test_extrusion_is_finite_at_the_edge_cases(barriers):
    module = make_extrusion(barriers=barriers)
    epi = torch.rand(4, 13, 50).half()
    enh = torch.tensor([[20.0], [20.0], [0.0], [49.0]])
    prom = torch.tensor([[20.0], [21.0], [49.0], [48.0]])
    out = module(epi, enh, prom)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    log_q = module.terms(epi, enh, prom)["log_q"]
    assert torch.equal(log_q[[0, 1, 3]], torch.zeros(3, module.heads))  # same or adjacent bins
    out.sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert module(epi.float(), enh, prom).dtype == torch.float32


# ---------------------------------------------------------------------------
# Sequence path
# ---------------------------------------------------------------------------

def test_sequence_path_needs_the_language_model_embeddings():
    with pytest.raises(ValueError, match="dna_input embedding"):
        build_model(config_for("phase2_p1a.yaml", "GA", seq_compat={"enabled": True}))


def test_sequence_path_reads_the_mean_of_each_element():
    torch.manual_seed(0)
    path = SequencePairPath(16, 4, rank=8, unary=3)
    seq = torch.randn(2, 8, 16)
    out = path(seq)
    assert out.shape == (2, 14) and path.out_dim == 14
    shuffled = seq[:, [3, 1, 0, 2, 6, 7, 5, 4]]          # parts reordered within each
    assert torch.allclose(path(shuffled), out, atol=1e-6)
    changed = seq.clone()
    changed[:, 4:] += torch.randn(2, 1, 16)               # only the promoter
    moved = path(changed)
    assert torch.allclose(moved[:, 8:11], out[:, 8:11])  # the enhancer projection
    assert not torch.allclose(moved[:, :8], out[:, :8])  # the product term


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,variant", sorted(PINNED))
def test_switches_off_build_the_model_from_before_phase_2(name, variant):
    model = build_model(config_for(name, variant))
    count, keys = PINNED[(name, variant)]
    assert sum(p.numel() for p in model.parameters()) == count
    assert key_hash(model) == keys
    assert model.pair_features is None and not hasattr(model, "fc_extra")


ARMS = {
    "phase2_r1_yy1.yaml": (False, False, False),
    "phase2_r2_compat.yaml": (False, False, True),
    "phase2_r3_extrusion.yaml": (False, True, False),
    "phase2_r4_groups.yaml": (True, False, False),
    "phase2_c.yaml": (True, True, True),
    "phase2_c_nolm.yaml": (True, True, False),
    "phase2_c_null.yaml": (True, True, True),
}


@pytest.mark.parametrize("name", sorted(ARMS))
def test_each_arm_builds_its_modules_in_both_models(name):
    groups, extrusion, compat = ARMS[name]
    shapes = {}
    for variant in ("GA", "baseline"):
        model = build_model(config_for(name, variant))
        assert hasattr(model.epi_cnn, "gate") == groups
        pair = model.pair_features
        assert (pair is not None and pair.extrusion is not None) == extrusion
        assert (pair is not None and pair.seq_compat is not None) == compat
        shapes[variant] = {k: tuple(v.shape) for k, v in model.state_dict().items()
                           if k.startswith(NEW_MODULES)}
    assert shapes["GA"] == shapes["baseline"]
    if name == "phase2_c_null.yaml":
        assert not model.pair_features.extrusion.barriers
    if name == "phase2_c_nolm.yaml":
        assert float(model.seq_input_scale) == 0.0            # the DNA branch is off


@pytest.mark.parametrize("variant", ["GA", "baseline"])
def test_every_parameter_gets_a_gradient_with_every_switch_on(variant):
    config = small(config_for("phase2_c.yaml", variant))
    torch.manual_seed(0)
    model = build_model(config).train()
    with torch.no_grad():               # zero at the start; not after a few steps
        model.fc_extra.weight.normal_(std=0.1)
    out = model(*batch(config))
    loss = out[0].sum() + out[1].sum() + model.attention_penalty(out[2])
    if variant != "baseline":
        loss = loss + model.memory_auxiliary_loss(out[3], prediction_weight=1.0)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing[:10]
    silent = [n for n, p in model.named_parameters()
              if n.startswith(NEW_MODULES) and not p.grad.abs().sum() > 0]
    assert not silent, silent


@pytest.mark.parametrize("variant", ["GA", "baseline"])
def test_with_fresh_weights_the_pair_features_change_nothing(variant):
    plain_config = small(config_for("phase2_r1_yy1.yaml", variant))
    torch.manual_seed(0)
    plain = build_model(plain_config).eval()
    torch.manual_seed(0)
    extended = build_model(small(config_for("phase2_c.yaml", variant,
                                            chromatin_stem="single"))).eval()
    before, after = plain.state_dict(), extended.state_dict()
    # Built last: every earlier weight is initialised as without them.
    assert list(after)[:len(before)] == list(before)
    assert all(torch.allclose(before[k], after[k], atol=1e-6) for k in before)
    assert extended.fc_extra.weight.abs().sum() == 0
    extended.load_state_dict(before, strict=False)
    args = batch(plain_config)
    with torch.no_grad():
        assert torch.equal(extended(*args)[0], plain(*args)[0])
        assert torch.equal(extended(*args)[1], plain(*args)[1])


def test_a_checkpoint_round_trips_through_evaluate_external(tmp_path):
    import evaluate_external
    config = small(config_for("phase2_c.yaml", "GA"))
    torch.manual_seed(0)
    model = build_model(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.01 * torch.randn_like(param))
    torch.save({"model": model.state_dict(), "config": config, "epoch": 1},
               tmp_path / "checkpoint.pt")
    (tmp_path / "config_snapshot.yaml").write_text(
        yaml.safe_dump(dict(config, _run={"seed": 0}), sort_keys=False))
    with open(tmp_path / "encoder.pkl", "wb") as handle:
        pickle.dump(None, handle)
    _, loaded, _ = evaluate_external.load_run(tmp_path, torch.device("cpu"))
    args = batch(config)
    with torch.no_grad():
        assert torch.equal(loaded(*args)[0], model(*args)[0])


# ---------------------------------------------------------------------------
# One epoch with every switch on
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_training_with_every_switch_on(tmp_path):
    pytest.importorskip("pyfaidx")
    from test_dna_inputs import build_embeddings, write_genome
    from test_end_to_end import CELLS, CHROMS, run_training, write_config, write_dataset
    bengi_dir, feats = write_dataset(tmp_path / "data")
    genome = tmp_path / "toy.fa"
    write_genome(genome, CHROMS)
    out = tmp_path / "emb"
    build_embeddings(bengi_dir, genome, out)

    # Four genome-wide motif tracks, the same files for every cell.
    motifs = ["CTCFmotif_plus", "CTCFmotif_minus", "YY1motif_plus", "YY1motif_minus"]
    listed = json.loads(feats.read_text())
    for motif in motifs:
        torch.save({c: torch.rand(4_000) for c in CHROMS}, tmp_path / "data" / f"{motif}.pt")
        for cell in CELLS:
            listed[cell][motif] = str(tmp_path / "data" / f"{motif}.pt")
    feats.write_text(json.dumps(listed))

    config_path = tmp_path / "config.yaml"
    write_config(config_path, bengi_dir, feats, epochs=1)
    config = yaml.safe_load(config_path.read_text())
    config["paths"].update(bengi_dirs=[str(bengi_dir)], dna_embeddings=str(out))
    config["data"].update(
        train_cells="all", modalities="all", dna_input="embedding", dna_segments=4,
        dna_embedding_dim=16, n_epigenetic_features=13,
        feats_order=config["data"]["feats_order"] + motifs)
    config["training"].update(valid_cells=["IMR90"], valid_samples=30, samples_per_epoch=64)
    config["augmentation"]["noise_mode"] = "tracks"
    phase2 = load_config(str(ROOT / "configs" / "phase2_c.yaml"))["model"]
    config["model"].update({k: phase2[k] for k in ("seq_compat", "extrusion",
                                                   "chromatin_stem", "chromatin_groups")})
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    for variant in ("GA", "baseline"):
        run = tmp_path / f"run_{variant}"
        result = run_training(config_path, run, extra=("--variant", variant))
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        saved = torch.load(run / "checkpoint.pt", weights_only=False)["model"]
        assert "epi_cnn.gate.weight" in saved and "fc_extra.weight" in saved
        assert any(k.startswith("pair_features.extrusion.hazard_net") for k in saved)
        assert any(k.startswith("pair_features.seq_compat.") for k in saved)

    test_file = str(bengi_dir / "HMEC.HiC-Benchmark.v3.tsv")
    scored = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_external.py"), "--runs",
         str(tmp_path / "run_GA"), "--tests", test_file, "--feats-config", str(feats),
         "--abc-dir", str(tmp_path / "no_abc"), "--out", str(tmp_path / "external"),
         "--device", "cpu", "--dna-embeddings", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert scored.returncode == 0, scored.stdout[-3000:] + scored.stderr[-3000:]
    inspected = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_modules.py"), "--run",
         str(tmp_path / "run_baseline"), "--test", test_file, "--feats-config", str(feats),
         "--out", str(tmp_path / "inspect"), "--device", "cpu", "--examples", "2",
         "--dna-embeddings", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert inspected.returncode == 0, inspected.stdout[-3000:] + inspected.stderr[-3000:]
    summary = json.loads((tmp_path / "inspect" / "modules.json").read_text())
    assert summary["groups"]["names"] == GROUPS
    assert abs(sum(summary["groups"]["alpha_mean"]["all"].values()) - 1) < 1e-4
    assert len(summary["extrusion"]["processivity_kb"]) == 2

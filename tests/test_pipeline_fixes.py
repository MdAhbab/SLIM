"""
Tests for the data and training pipeline fixes.

Each fix sits behind a configuration switch whose original setting reproduces
the old behaviour, so these tests check both sides: the corrected rule does
what it says, and the original rule is unchanged. None of them needs the real
dataset or a GPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.bengi import deduplicate, parse_line, select_rows
from src.config import load_config
from src.dataset import EPIDataset, _reverse_complement, shift_tracks
from src.encoding import CrossFitEncoder, POCD_ND_Encoder, chromosome_half
from src.slim_model import build_model


# ---------------------------------------------------------------------------
# Reading and merging BENGI rows
# ---------------------------------------------------------------------------

def bengi_line(label, chrom="chr1", enh=(1000, 1400), tss=5000, cell="GM12878",
               gene="ENSG1.1", transcript="ENST1.1"):
    return "\t".join(map(str, [
        label, abs(tss - enh[0]), chrom, enh[0], enh[1],
        f"{chrom}:{enh[0]}-{enh[1]}|{cell}|EH37E1",
        chrom, tss - 1000, tss + 1000,
        f"{chrom}:{tss}-{tss + 1}|{cell}|{gene}|{transcript}|+"]))


def test_parse_line_reads_every_identifier():
    row = parse_line(bengi_line(1), assay="HiC")
    assert row["cell"] == "GM12878"
    assert row["enh_id"] == "EH37E1"
    assert row["gene"] == "ENSG1"            # version dropped
    assert row["transcript"] == "ENST1.1"
    assert row["enh_coord"] == 1200
    assert row["tss_coord"] == 5000
    assert row["assay"] == "HiC"


def test_union_dedup_merges_repeats_and_keeps_any_positive():
    rows = [parse_line(bengi_line(0), "HiC"),
            parse_line(bengi_line(1, tss=9000), "HiC"),
            parse_line(bengi_line(1), "RNAPII-ChIAPET"),     # repeats row 0
            parse_line(bengi_line(0), "CTCF-ChIAPET")]      # repeats row 0
    merged, report = deduplicate(rows, "union")
    assert len(merged) == 2
    assert merged[0]["label"] == 1                          # union of 0, 1, 0
    assert merged[0]["assay"] == "CTCF-ChIAPET+HiC+RNAPII-ChIAPET"
    assert merged[1]["tss_coord"] == 9000                   # order kept
    assert report["duplicate_rows"] == 2
    assert report["pairs_with_conflicting_labels"] == 1


def test_no_dedup_returns_rows_unchanged():
    rows = [parse_line(bengi_line(0), "HiC"), parse_line(bengi_line(1), "HiC")]
    kept, report = deduplicate(rows, "none")
    assert kept is rows
    assert report["duplicate_rows"] == 1


def test_assay_filters_apply_to_their_own_split():
    rows = [parse_line(bengi_line(1, chrom="chr1"), "HiC"),
            parse_line(bengi_line(1, chrom="chr1"), "RNAPII-ChIAPET"),
            parse_line(bengi_line(1, chrom="chr11"), "HiC"),
            parse_line(bengi_line(1, chrom="chr11"), "CTCF-ChIAPET")]
    kept, _ = select_rows(rows, held={"chr11"}, train_assays=None,
                          valid_assays=["HiC"])
    assert [(r["chrom"], r["assay"]) for r in kept] == [
        ("chr1", "HiC"), ("chr1", "RNAPII-ChIAPET"), ("chr11", "HiC")]
    kept, _ = select_rows(rows, held={"chr11"}, train_assays=["HiC"],
                          valid_assays=None)
    assert [(r["chrom"], r["assay"]) for r in kept] == [
        ("chr1", "HiC"), ("chr11", "HiC"), ("chr11", "CTCF-ChIAPET")]


# ---------------------------------------------------------------------------
# The cross-fitted sequence encoding
# ---------------------------------------------------------------------------

def test_chromosome_halves():
    assert [chromosome_half(c) for c in ("chr1", "chr11", "chr17", "chr21")] == [0] * 4
    assert [chromosome_half(c) for c in ("chr2", "chr22", "chrX", "chrY")] == [1] * 4


class RandomSequences:
    """Random DNA with random labels, the interface fit_sequence_encoder uses."""

    def __init__(self, n=600, length=120, seed=0):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 2, n)
        self.chroms = np.array([f"chr{1 + i % 6}" for i in range(n)])
        self.seqs = ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]

    def get_labels(self):
        return self.labels

    def get_chrom_groups(self):
        return self.chroms

    def sequence_pair(self, i):
        half = len(self.seqs[i]) // 2
        return self.seqs[i][:half], self.seqs[i][half:]


def leak_auroc(rule):
    """AUROC of the log-ratio score on rows the encoder was fitted on.

    The labels are random, so any separation is the encoder handing each
    fitted row its own label.
    """
    from sklearn.metrics import roc_auc_score
    from train import fit_sequence_encoder
    source = RandomSequences()
    config = {"data": {"kmer_size": 3, "sequence_length": 120,
                       "encoder_fit_samples": 1000, "encoder_fit": rule}}
    encoder = fit_sequence_encoder(source, list(range(len(source.seqs))),
                                   config, seed=0)
    fitted = encoder.fit_rows
    scores = []
    for i in fitted:
        seq = "".join(source.sequence_pair(i))
        scores.append(encoder.log_ratio_score(seq, source.chroms[i])
                      if isinstance(encoder, CrossFitEncoder)
                      else encoder.log_ratio_score(seq))
    return roc_auc_score(source.labels[fitted], scores)


def test_training_fit_leaks_labels_and_crossfit_does_not():
    leaked = leak_auroc("train")
    clean = leak_auroc("crossfit")
    assert leaked > 0.9, f"expected the original rule to leak, AUROC {leaked:.3f}"
    assert abs(clean - 0.5) < 0.1, f"cross-fitting should not leak, AUROC {clean:.3f}"


def test_crossfit_encodes_with_the_other_half():
    source = RandomSequences(n=200)
    encoder = CrossFitEncoder(k=3)
    pos = [[s for s, y, c in zip(source.seqs, source.labels, source.chroms)
            if y == 1 and chromosome_half(c) == h] for h in (0, 1)]
    neg = [[s for s, y, c in zip(source.seqs, source.labels, source.chroms)
            if y == 0 and chromosome_half(c) == h] for h in (0, 1)]
    encoder.fit(pos, neg, 120)
    seq = source.seqs[0]
    assert torch.equal(encoder.transform(seq, "chr1"),
                       encoder.encoders[1].transform(seq))
    assert torch.equal(encoder.transform(seq, "chr2"),
                       encoder.encoders[0].transform(seq))
    with pytest.raises(ValueError):
        encoder.transform(seq)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class RecordingEncoder:
    """Stands in for the sequence encoder and records what it was given."""
    k = 3

    def __init__(self):
        self.seen = []

    def transform(self, sequence, chrom=None):
        self.seen.append(sequence)
        return torch.zeros(64, len(sequence) - 2)


class OnePair:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"epi": torch.rand(9, 20), "enhancer_seq": "AAACCC",
                "promoter_seq": "GGGTTA", "chrom": "chr1",
                "label": torch.tensor([1.0]), "dist": torch.tensor([1.0]),
                "enh_idx": torch.tensor([5.0]), "prom_idx": torch.tensor([9.0])}


def augmenting_dataset(rc_mode):
    config = {"data": {"sequence_length": 12, "epigenetic_bins": 20,
                       "n_epigenetic_features": 9},
              "augmentation": {"rc_prob": 1.0, "epi_noise_std": 0.0,
                               "shift_max_bins": 0, "rc_mode": rc_mode}}
    encoder = RecordingEncoder()
    dataset = EPIDataset(config, encoder, source_dataset=OnePair())
    dataset.augment = True
    return dataset, encoder


def test_segment_reverse_complement_keeps_the_enhancer_first():
    dataset, encoder = augmenting_dataset("segment")
    dataset[0]
    assert encoder.seen[-1] == _reverse_complement("AAACCC") + _reverse_complement("GGGTTA")
    assert encoder.seen[-1][:6] == "GGGTTT"                  # the enhancer, flipped


def test_original_reverse_complement_swaps_the_halves():
    dataset, encoder = augmenting_dataset("concat")
    dataset[0]
    assert encoder.seen[-1] == _reverse_complement("AAACCCGGGTTA")
    assert encoder.seen[-1][:6] == "TAACCC"                  # the promoter, flipped


def test_track_shift_leaves_the_position_channel_and_fills_with_zeros():
    epi = torch.arange(3 * 8, dtype=torch.float32).view(3, 8) + 1
    moved = shift_tracks(epi, 2, "tracks")
    assert torch.equal(moved[0], epi[0])
    assert torch.equal(moved[1:, 2:], epi[1:, :-2])
    assert float(moved[1:, :2].abs().sum()) == 0.0
    moved = shift_tracks(epi, -3, "tracks")
    assert torch.equal(moved[1:, :-3], epi[1:, 3:])
    assert float(moved[1:, -3:].abs().sum()) == 0.0
    assert torch.equal(shift_tracks(epi, 2, "all"), torch.roll(epi, 2, dims=1))


# ---------------------------------------------------------------------------
# Models: input ablations, memory ablations, stochastic depth
# ---------------------------------------------------------------------------

def small(name, **data):
    config = load_config(str(ROOT / "configs" / name))
    config["data"].update({"sequence_length": 600, "epigenetic_bins": 500, **data})
    return config


def batch(config, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = config["data"]["epigenetic_bins"]
    return (torch.randn(2, 64, config["data"]["sequence_length"] - 2, generator=g),
            torch.rand(2, config["data"]["n_epigenetic_features"], n, generator=g),
            torch.tensor([[100.0], [200.0]]), torch.tensor([[300.0], [320.0]]))


@pytest.mark.parametrize("name", ["slim_ka.yaml", "baseline.yaml"])
def test_without_dna_the_sequence_input_is_ignored(name):
    torch.manual_seed(0)
    model = build_model(small(name, modalities="epi")).eval()
    seq, epi, enh, prom = batch(model_config := small(name, modalities="epi"))
    with torch.no_grad():
        first = model(seq, epi, enh, prom)[0]
        second = model(seq * 7.0 + 3.0, epi, enh, prom)[0]
    assert torch.allclose(first, second)


def test_baseline_honours_the_chromatin_ablation_too():
    torch.manual_seed(0)
    model = build_model(small("baseline.yaml", modalities="seq")).eval()
    seq, epi, enh, prom = batch(small("baseline.yaml"))
    with torch.no_grad():
        first = model(seq, epi, enh, prom)[0]
        second = model(seq, epi * 5.0, enh, prom)[0]
    assert torch.allclose(first, second)


def test_new_memory_switches_default_to_the_current_encoder():
    config = small("slim_ka.yaml")
    explicit = small("slim_ka.yaml")
    explicit["model"]["memory"].update({"selection": "learned", "read_back": True})
    torch.manual_seed(0)
    a = build_model(config).eval()
    torch.manual_seed(0)
    b = build_model(explicit).eval()
    # The spline layers are initialised by a least-squares solve whose CPU
    # result, like the forward kernels, varies by ~1e-8 between two builds of
    # one config, so both are compared to a tolerance well above that.
    state_a, state_b = a.state_dict(), b.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.allclose(state_a[k].float(), state_b[k].float(), atol=1e-6, rtol=0)
               for k in state_a)
    inputs = batch(config)
    with torch.no_grad():
        assert torch.allclose(a(*inputs)[0], b(*inputs)[0], atol=1e-6, rtol=0)


def test_random_selection_reproduces_in_evaluation():
    config = small("slim_ka_random_select.yaml")
    torch.manual_seed(0)
    model = build_model(config).eval()
    inputs = batch(config)
    with torch.no_grad():
        first = model(*inputs, return_trace=True)[-1]["tokens"]
        second = model(*inputs, return_trace=True)[-1]["tokens"]
    for x, y in zip(first, second):
        assert torch.equal(x, y)


def test_without_read_back_the_memory_does_not_reach_the_output():
    config = small("slim_ka_no_readback.yaml")
    torch.manual_seed(0)
    model = build_model(config).eval()
    inputs = batch(config)
    with torch.no_grad():
        before = model(*inputs)[0]
        for block in model.encoder.blocks:
            for param in block.bin_cross_attention.parameters():
                param.add_(1.0)
        after = model(*inputs)[0]
    assert torch.equal(before, after)


def test_baseline_reads_stochastic_depth_from_the_config():
    from src.baseline_model import DropPath
    config = small("baseline.yaml")
    config["model"]["drop_path_rate"] = 0.1
    model = build_model(config)
    assert isinstance(model.transformer.blocks[-1].drop_path, DropPath)


def test_parameter_counts_do_not_depend_on_the_ablation_switches():
    counts = {}
    for name in ("slim_ka.yaml", "slim_ka_random_select.yaml",
                 "slim_ka_no_readback.yaml", "slim_ka_learned_only.yaml",
                 "slim_ka_old_recipe.yaml"):
        model = build_model(small(name))
        counts[name] = sum(p.numel() for p in model.parameters())
    assert len(set(counts.values())) == 1, counts


# ---------------------------------------------------------------------------
# Training recipe
# ---------------------------------------------------------------------------

def test_warmup_cosine_schedule_shape():
    from train import build_scheduler
    config = {"training": {"scheduler": "warmup_cosine", "epochs": 4,
                           "warmup_fraction": 0.1, "min_lr_ratio": 0.05,
                           "lr": 1.0}}
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler, per_batch = build_scheduler(optimizer, config, steps_per_epoch=25)
    assert per_batch
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < 0.2                                  # warming up
    assert rates[9] == pytest.approx(1.0)                  # peak after warm-up
    assert all(a >= b - 1e-12 for a, b in zip(rates[9:], rates[10:]))
    assert rates[-1] == pytest.approx(0.05, abs=0.01)      # floor at the end


def test_original_scheduler_is_unchanged():
    from train import build_scheduler
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler, per_batch = build_scheduler(
        optimizer, {"training": {"use_cosine_scheduler": False, "epochs": 5}}, 10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert not per_batch


def test_weight_decay_spares_one_dimensional_parameters():
    from train import build_optimizer
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(model, {"training": {
        "lr": 1e-3, "weight_decay": 0.05, "no_decay_1d": True}})
    decays = {len(g["params"]): g["weight_decay"] for g in optimizer.param_groups}
    assert decays == {1: 0.05, 3: 0.0}
    optimizer = build_optimizer(model, {"training": {
        "lr": 1e-3, "weight_decay": 1e-4}})
    assert len(optimizer.param_groups) == 1


def test_old_recipe_config_restores_the_original_optimisation():
    config = load_config(str(ROOT / "configs" / "slim_ka_old_recipe.yaml"))
    t = config["training"]
    assert (t["scheduler"], t["weight_decay"], t["ema_decay"], t["epochs"]) == \
        ("plateau", 0.0001, 0.0, 5)
    assert config["model"]["drop_path_rate"] == 0.0
    # The data side stays corrected.
    assert config["data"]["encoder_fit"] == "crossfit"
    assert config["data"]["dedup"] == "union"


# ---------------------------------------------------------------------------
# Protocol choice
# ---------------------------------------------------------------------------

def write_pilot(root, name, train_assays, modalities, val_aupr, val_rows=100):
    run = root / name
    run.mkdir()
    np.savez(run / "eval_results.npz", val_aupr=np.float64(val_aupr),
             val_auroc=np.float64(0.8), test_aupr=np.float64(0.99),
             val_rows=np.int64(val_rows))
    (run / "config_snapshot.yaml").write_text(yaml.safe_dump({
        "data": {"train_assays": train_assays, "modalities": modalities},
        "training": {"valid_assays": ["HiC"]}}))
    return str(run)


def run_choose(pilots, out, *extra):
    import subprocess
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "choose_protocol.py"),
                           "--pilots", *pilots, "--out", str(out), *extra],
                          capture_output=True, text=True)


def test_protocol_is_chosen_on_validation(tmp_path):
    pilots = [write_pilot(tmp_path, "hic_dna", ["HiC"], "all", 0.30),
              write_pilot(tmp_path, "hic_nodna", ["HiC"], "epi", 0.35),
              write_pilot(tmp_path, "all_dna", None, "all", 0.20)]
    out = tmp_path / "protocol.json"
    result = run_choose(pilots, out)
    assert result.returncode == 0, result.stderr
    protocol = json.loads(out.read_text())
    assert protocol["chosen"] == "hic_nodna"
    assert (protocol["train_assays"], protocol["modalities"]) == (["HiC"], "epi")
    result = run_choose(pilots, out, "--keep-dna")
    assert json.loads(out.read_text())["chosen"] == "hic_dna"


def test_protocol_refuses_pilots_scored_on_different_rows(tmp_path):
    pilots = [write_pilot(tmp_path, "a", ["HiC"], "all", 0.3, val_rows=100),
              write_pilot(tmp_path, "b", None, "all", 0.4, val_rows=120)]
    result = run_choose(pilots, tmp_path / "protocol.json")
    assert result.returncode != 0


def test_every_stage_after_the_choice_uses_the_protocol():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    plan = run.build_plan([0, 1, 2])
    names = [stage["name"] for stage in plan]
    chosen = names.index("choose_protocol")
    for stage in plan:
        command = stage["command"]
        if "scripts/train.py" not in command:
            continue
        uses = "--protocol" in command
        if stage["name"].startswith("pilot_"):
            assert not uses and names.index(stage["name"]) < chosen
        else:
            assert uses, stage["name"]
            assert names.index(stage["name"]) > chosen

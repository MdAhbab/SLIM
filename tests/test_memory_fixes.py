"""
The memory-encoder fixes, checked on real preprocessed data.

Each test builds its inputs from the BENGI pairs, the binned chromatin tracks
and hg19 named in configs/base.yaml, rather than from random tensors, so the
tokens the encoder sees have the statistics of the real task. The module is
skipped when that data is not on the machine.

What is checked, fix by fix:

  sliding window     block-wise local attention equals masked full attention
  content addressing survivors reach slots beyond the first eight, and never
                     share a slot within a layer
  decay              an unwritten slot shrinks by exactly bin_decay
  prediction         one prediction per token, trained by its own loss only
  gate gradient      with gate_gradient="all" (an option, off in the
                     configurations) the task loss reaches every score, and
                     the forward pass is the same either way
  seams              the local window does not cross the DNA/chromatin seams

and that the legacy configuration keeps the original behaviour in each case.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.slim_model import build_model

N_SAMPLES = 16


def _data_available(config):
    paths = config["paths"]
    bengi = os.path.join(paths["bengi_dir"], "NHEK.HiC-Benchmark.v3.tsv.gz")
    return all(os.path.exists(p) for p in
               (bengi, paths["feats_config"], paths["ref_genome"]))


@pytest.fixture(scope="module")
def real_batch():
    """Sixteen real NHEK pairs, spread evenly through the benchmark file."""
    config = load_config(str(ROOT / "configs" / "slim_ka.yaml"))
    if not _data_available(config):
        pytest.skip("the preprocessed dataset named in configs/base.yaml is "
                    "not on this machine")
    from src.dataset import EPIDataset
    from src.encoding import POCD_ND_Encoder
    from src.epi_data_pipeline import EPIGenomicDataset

    paths, data = config["paths"], config["data"]
    genomic = EPIGenomicDataset(
        bengi_paths=[os.path.join(paths["bengi_dir"],
                                  "NHEK.HiC-Benchmark.v3.tsv.gz")],
        feats_config_path=paths["feats_config"],
        feats_order=data["feats_order"], seq_len=data["seq_len_bp"],
        bin_size=data["bin_size"], enhancer_window=data["enhancer_window"],
        promoter_window=data["promoter_window"],
        ref_genome_path=paths["ref_genome"])
    labels = np.array(genomic.get_labels())
    # Fit the sequence encoder on real sequences, as training does.
    pos = np.where(labels == 1)[0][:100]
    neg = np.where(labels == 0)[0][:100]
    join = lambda i: genomic[int(i)]["enhancer_seq"] + genomic[int(i)]["promoter_seq"]
    encoder = POCD_ND_Encoder(k=data["kmer_size"])
    encoder.fit([join(i) for i in pos], [join(i) for i in neg],
                data["sequence_length"])
    dataset = EPIDataset(config, encoder, source_dataset=genomic)
    dataset.augment = False
    picks = np.linspace(0, len(dataset) - 1, N_SAMPLES).astype(int)
    items = [dataset[int(i)] for i in picks]
    batch = {key: torch.stack([torch.as_tensor(item[key]) for item in items]).float()
             for key in ("seq", "epi", "enh_idx", "prom_idx", "label")}
    return batch


def build(name, seed=0, **memory):
    """Build a configuration, optionally overriding `model.memory` keys."""
    config = load_config(str(ROOT / "configs" / name))
    config["model"]["memory"].update(memory)
    # Stochastic depth drops a whole residual branch for some samples in
    # training mode, which zeroes their gate gradients by design. These tests
    # measure how gradients are routed, so it is switched off here.
    config["model"]["drop_path_rate"] = 0.0
    torch.manual_seed(seed)
    return build_model(config)


def model_inputs(batch):
    return batch["seq"], batch["epi"], batch["enh_idx"], batch["prom_idx"]


def encoder_tokens(model, batch):
    """The 628 real tokens the encoder receives, from the model's branches."""
    with torch.no_grad():
        seq = model.seq_bilstm(model.seq_cnn(batch["seq"]).permute(0, 2, 1))[0]
        epi = batch["epi"] * model.epi_channel_mask
        epi = model.epi_bilstm(model.epi_cnn(epi).permute(0, 2, 1))[0]
        return model.pos_enc(torch.cat([seq, epi], dim=1))


def dense_local_attention(module, x, segment_ids=None):
    """Reference: full score matrix, then the window (and segment) mask."""
    batch, seq_len, dim = x.shape
    q, k, v = module.qkv(x).chunk(3, dim=-1)
    q, k, v = (t.view(batch, seq_len, module.num_heads, module.head_dim)
               .transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / module.head_dim ** 0.5
    pos = torch.arange(seq_len)
    mask = (pos[:, None] - pos[None, :]).abs() <= module.radius
    if segment_ids is not None:
        mask = mask & (segment_ids[:, None] == segment_ids[None, :])
    scores = scores.masked_fill(~mask, -1e4)
    weights = F.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    y = (weights @ v).transpose(1, 2).reshape(batch, seq_len, dim)
    return module.out(y)


# --------------------------------------------------------------- sliding window

def test_sliding_window_equals_masked_full_attention(real_batch):
    model = build("slim_ka.yaml").eval()
    tokens = encoder_tokens(model, real_batch)
    attention = model.encoder.blocks[0].local_attention
    segments = model.encoder.segment_ids
    with torch.no_grad():
        for segment_ids in (None, segments):
            fast = attention(tokens, segment_ids)
            reference = dense_local_attention(attention, tokens, segment_ids)
            assert torch.allclose(fast, reference, atol=1e-5, rtol=1e-4), (
                f"block-wise attention differs from the masked reference by "
                f"{(fast - reference).abs().max():.2e}")


def test_sliding_window_score_tensor_grows_linearly(real_batch):
    """The block mask, and so the score tensor, is linear in sequence length."""
    attention = build("slim_ka.yaml").encoder.blocks[0].local_attention
    r = attention.radius
    sizes = []
    for length in (628, 1256, 2512):
        blocks = -(-length // r)
        sizes.append(attention._block_mask(length, blocks, torch.device("cpu"),
                                           None).numel())
    assert sizes[1] / sizes[0] < 2.1 and sizes[2] / sizes[1] < 2.1
    assert sizes[0] < 628 * 628 / 5


# ----------------------------------------------------------- content addressing

def slots_written(model, batch):
    model.eval()
    with torch.no_grad():
        trace = model(*model_inputs(batch), return_trace=True)[-1]
    return trace["slots"]


def test_content_addressing_is_not_confined_to_the_first_slots(real_batch):
    """The original rule wrote survivor j to slot j, so slots 8-15 sat idle.

    Which slots content picks depends on the inputs, so a small sample need
    not touch all sixteen; what must hold is that the writes are no longer
    confined to slots 0-7.
    """
    per_layer = slots_written(build("slim_ka.yaml"), real_batch)
    used = torch.unique(torch.cat([s.flatten() for s in per_layer]))
    assert (used >= 8).any(), f"only slots {used.tolist()} were written"
    assert used.numel() > 8, f"only slots {used.tolist()} were written"
    for slots in per_layer:
        for row in slots:
            assert row.unique().numel() == row.numel(), (
                "two survivors were written to the same slot in one layer")


def test_legacy_order_addressing_writes_only_the_first_slots(real_batch):
    per_layer = slots_written(build("slim_ka_legacy.yaml"), real_batch)
    used = torch.unique(torch.cat([s.flatten() for s in per_layer]))
    assert used.tolist() == list(range(8))


# ------------------------------------------------------------------------ decay

def test_unwritten_slots_decay_by_bin_decay(real_batch):
    fixed = build("slim_ka.yaml").eval()
    legacy = build("slim_ka_legacy.yaml").eval()
    for model, should_decay in ((fixed, True), (legacy, False)):
        block = model.encoder.blocks[1]
        # A real memory state: the memory after the first layer.
        with torch.no_grad():
            tokens = encoder_tokens(model, real_batch)
            state = model.encoder.initial_bin[None].expand(N_SAMPLES, -1, -1)
            _, state, _ = model.encoder.blocks[0](
                tokens, state, model.encoder.segment_ids)
            nothing = torch.zeros(N_SAMPLES, 16, 1)
            aged = block._gru_write(state, torch.zeros_like(state), nothing)
        ratio = aged.norm(dim=-1) / state.norm(dim=-1)
        if should_decay:
            assert torch.allclose(ratio, torch.full_like(ratio, 0.92), atol=1e-5)
        else:
            assert (ratio - 0.92).abs().min() > 0.01, (
                "the legacy layer norm should undo the decay")


# ------------------------------------------------------------------- prediction

def test_prediction_is_per_token_and_trains_only_the_predictor(real_batch):
    model = build("slim_ka.yaml").train()
    out = model(*model_inputs(real_batch))
    aux = out[3]
    assert "prediction_loss" in aux and torch.isfinite(aux["prediction_loss"])
    params = dict(model.named_parameters())
    grads = torch.autograd.grad(aux["prediction_loss"], list(params.values()),
                                allow_unused=True)
    reached = {name for (name, _), g in zip(params.items(), grads)
               if g is not None and g.abs().sum() > 0}
    assert reached, "the prediction loss trains nothing"
    stray = {n for n in reached
             if "position_query" not in n and "predict_from_bin" not in n}
    assert not stray, f"the prediction loss leaked into {sorted(stray)[:5]}"

    gate = model.encoder.blocks[1].survival_gate
    with torch.no_grad():
        tokens = encoder_tokens(model, real_batch)
        memory = torch.randn(N_SAMPLES, 16, 96)
        query = gate.position_query(gate.positions[:tokens.shape[1]])
        weights = F.softmax(query @ memory.transpose(1, 2) / 96 ** 0.5, dim=-1)
        prediction = gate.predict_from_bin(weights @ memory)
    assert prediction.shape == tokens.shape
    assert prediction.std(dim=1).mean() > 0, "prediction ignores position"


def test_legacy_prediction_has_no_loss(real_batch):
    model = build("slim_ka_legacy.yaml").eval()
    with torch.no_grad():
        aux = model(*model_inputs(real_batch))[3]
    assert "prediction_loss" not in aux


# ---------------------------------------------------------------- gate gradient

def score_gradients(model, batch):
    model.train()
    scores = []

    def keep(module, inputs, output):
        output[3].retain_grad()
        scores.append(output[3])

    hooks = [block.survival_gate.register_forward_hook(keep)
             for block in model.encoder.blocks]
    try:
        logits = model(*model_inputs(batch))[0]
        F.binary_cross_entropy_with_logits(logits, batch["label"]).backward()
    finally:
        for hook in hooks:
            hook.remove()
    return [s.grad for s in scores]


def test_dense_gate_gradient_reaches_every_survival_score(real_batch):
    model = build("slim_ka.yaml", gate_gradient="all")
    for grad in score_gradients(model, real_batch):
        assert (grad != 0).float().mean() > 0.99, (
            "the task gradient still misses tokens that were not written")


def test_legacy_task_gradient_reaches_only_the_survivors(real_batch):
    for grad in score_gradients(build("slim_ka_legacy.yaml"), real_batch):
        assert ((grad != 0).sum(dim=1) <= 8).all()


def test_gate_gradient_switch_leaves_the_forward_pass_unchanged(real_batch):
    dense = build("slim_ka.yaml", gate_gradient="all").eval()
    sparse = build("slim_ka.yaml", gate_gradient="selected").eval()
    sparse.load_state_dict(dense.state_dict())
    with torch.no_grad():
        a = dense(*model_inputs(real_batch))[0]
        b = sparse(*model_inputs(real_batch))[0]
    assert torch.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------------------ seams

def test_local_window_stops_at_the_seams(real_batch):
    """Changing the chromatin tokens must not reach the DNA tokens locally."""
    for name, isolated in (("slim_ka.yaml", True), ("slim_ka_legacy.yaml", False)):
        model = build(name).eval()
        attention = model.encoder.blocks[0].local_attention
        tokens = encoder_tokens(model, real_batch)
        changed = tokens.clone()
        changed[:, 128:] = tokens[:, 128:].flip(0)   # another pair's chromatin
        with torch.no_grad():
            before = attention(tokens, model.encoder.segment_ids)[:, 96:128]
            after = attention(changed, model.encoder.segment_ids)[:, 96:128]
        moved = (before - after).abs().max()
        if isolated:
            assert moved < 1e-6, f"DNA tokens moved by {moved:.2e}"
        else:
            assert moved > 1e-3, "legacy windows should cross the seam"


# ------------------------------------------------------------ old checkpoints

def test_a_checkpoint_from_before_the_fixes_loads_as_legacy():
    path = ROOT / "results" / "ka_legacy" / "seed0" / "checkpoint.pt"
    if not path.exists():
        pytest.skip("no legacy KA checkpoint on this machine")
    state = torch.load(path, map_location="cpu", weights_only=False)
    memory = state["config"]["model"].get("memory", {})
    if memory.get("slot_addressing", "order") != "order":
        pytest.skip("this checkpoint was trained with the fixes")
    model = build_model(state["config"])
    model.load_state_dict(state["model"], strict=True)
    assert model.memory_config.slot_addressing == "order"
    assert model.memory_config.memory_norm == "all"

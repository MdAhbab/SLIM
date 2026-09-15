"""
Survival-gated bounded-memory encoder.

The encoder replaces global self-attention with three steps per layer:

  1. Local windowed self-attention, so each token mixes only with neighbours
     inside a window of `local_window` tokens.
  2. A survival gate that scores every token and writes the highest-scoring
     ones into a bounded memory of `bin_slots` slots through a gated recurrent
     update with decay.
  3. Cross-attention from every token back to the memory, so information from
     distant positions reaches the whole sequence through the memory rather
     than through all-to-all attention.

The feed-forward sublayer is selected by `ffn_type` and is the ONLY difference
between the model variants:

    ffn_type="gelu"  variant A   rectified two-layer network, hidden 4*d
    ffn_type="kan"   variant KA  Kolmogorov-Arnold layer with cubic B-splines
    ffn_type="glu"   variant GA  gated linear unit, hidden 8/3*d

Hidden widths follow the usual parameter-matching convention for gated units,
so the rectified and gated sublayers hold the same number of parameters to
within 0.1 percent. Every other module in the block is shared, so a comparison
between variants isolates the feed-forward sublayer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model_layers import KAN


@dataclass
class MemoryConfig:
    """Configuration for the survival-gated memory encoder."""

    model_dim: int = 180
    bin_dim: int = 96
    num_layers: int = 3
    num_heads: int = 6
    local_window: int = 64
    bin_slots: int = 16
    survivors_per_layer: int = 8
    dropout: float = 0.1
    gate_temperature: float = 1.0
    novelty_weight: float = 0.35
    prediction_error_weight: float = 0.20
    bin_decay: float = 0.92
    bin_update: str = "gru"
    use_straight_through_gate: bool = True
    max_length: int = 640

    # Feed-forward sublayer: the only difference between variants A, KA and GA.
    ffn_type: str = "kan"
    kan_hidden: int = 64
    ffn_multiplier: int = 4

    # Gate-component switches, for isolating the three scoring terms.
    use_learned_score: bool = True
    use_novelty: bool = True
    use_prediction_error: bool = True

    def __post_init__(self) -> None:
        if self.bin_update not in {"fifo", "gru", "attention"}:
            raise ValueError("bin_update must be one of: fifo, gru, attention")
        if self.ffn_type not in {"gelu", "kan", "glu"}:
            raise ValueError("ffn_type must be one of: gelu, kan, glu")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if not (self.use_learned_score or self.use_novelty
                or self.use_prediction_error):
            raise ValueError("the survival gate needs at least one scoring term")


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class LocalSelfAttention(nn.Module):
    """Multi-head self-attention restricted to a local window."""

    def __init__(self, model_dim: int, num_heads: int, local_window: int,
                 dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.local_window = local_window
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._local_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)

    def _local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        radius = max(1, self.local_window // 2)
        positions = torch.arange(seq_len, device=device)
        distance = (positions[:, None] - positions[None, :]).abs()
        return (distance <= radius).view(1, 1, seq_len, seq_len)


class BinCrossAttention(nn.Module):
    """Every token reads from the bounded bounded memory."""

    def __init__(self, model_dim: int, bin_dim: int, num_heads: int,
                 dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.bin_dim = bin_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(bin_dim, model_dim, bias=False)
        self.v = nn.Linear(bin_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, bin_state: Tensor) -> Tensor:
        batch, seq_len, _ = tokens.shape
        slots = bin_state.shape[1]
        q = self.q(tokens).view(batch, seq_len, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k(bin_state).view(batch, slots, self.num_heads,
                                   self.head_dim).transpose(1, 2)
        v = self.v(bin_state).view(batch, slots, self.num_heads,
                                   self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = self.dropout(F.softmax(scores, dim=-1))
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.model_dim)
        return self.out(y)


# ---------------------------------------------------------------------------
# Survival gate
# ---------------------------------------------------------------------------


class SurvivalGate(nn.Module):
    """Scores tokens for survival into the bounded memory.

    The score of token i combines three terms:

        s_i = g(x_i, mean(B)) + alpha * novelty_i + beta * prediction_error_i

    where g is a learned two-layer network, novelty is one minus the highest
    cosine similarity between the compressed token and any memory slot, and
    prediction error is the squared error between the token and a linear
    prediction made from the memory, normalised by its batch mean. Each term
    can be switched off independently to isolate its contribution.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        self.compress = nn.Linear(config.model_dim, config.bin_dim)
        self.predict_from_bin = nn.Linear(config.bin_dim, config.model_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(config.model_dim + config.bin_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def forward(self, tokens: Tensor, bin_state: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        batch, seq_len, _ = tokens.shape
        compressed = self.compress(tokens)
        bin_pool = bin_state.mean(dim=1)
        expanded_pool = bin_pool[:, None, :].expand(batch, seq_len, -1)

        learned_score = self.score_mlp(
            torch.cat([tokens, expanded_pool], dim=-1)).squeeze(-1)
        novelty = self._novelty(compressed, bin_state)
        prediction_error = self._prediction_error(tokens, bin_pool)

        score = torch.zeros_like(learned_score)
        if self.config.use_learned_score:
            score = score + learned_score
        if self.config.use_novelty:
            score = score + self.config.novelty_weight * novelty
        if self.config.use_prediction_error:
            score = score + self.config.prediction_error_weight * prediction_error

        gate, gate_probability = self._topk_gate(score)
        debug = {
            "score": score.detach(),
            "novelty": novelty.detach(),
            "prediction_error": prediction_error.detach(),
            "gate": gate.detach(),
            "gate_probability": gate_probability.detach(),
        }
        return compressed, gate, gate_probability, score, debug

    def _novelty(self, candidates: Tensor, bin_state: Tensor) -> Tensor:
        candidate_norm = F.normalize(candidates, dim=-1)
        bin_norm = F.normalize(bin_state, dim=-1)
        similarity = torch.matmul(candidate_norm, bin_norm.transpose(1, 2))
        max_similarity = similarity.max(dim=-1).values
        return (1.0 - max_similarity).clamp(min=0.0, max=2.0)

    def _prediction_error(self, tokens: Tensor, bin_pool: Tensor) -> Tensor:
        prediction = self.predict_from_bin(bin_pool)[:, None, :]
        error = (tokens - prediction).pow(2).mean(dim=-1)
        return error / (error.detach().mean(dim=-1, keepdim=True) + 1e-6)

    def _topk_gate(self, score: Tensor) -> Tuple[Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, score.shape[-1])
        topk = torch.topk(score, k=k, dim=-1).indices
        hard = torch.zeros_like(score).scatter(1, topk, 1.0)
        soft = torch.sigmoid(score / max(self.config.gate_temperature, 1e-4))
        if not self.config.use_straight_through_gate:
            return hard, soft
        # Straight-through: the hard choice goes forward, the sigmoid relaxation
        # carries the gradient backward.
        return hard.detach() - soft.detach() + soft, soft


# ---------------------------------------------------------------------------
# Feed-forward sublayers (the only difference between variants)
# ---------------------------------------------------------------------------


class KANFeedForward(nn.Module):
    """Kolmogorov-Arnold layer with cubic B-splines on edges (variant KA)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.kan = KAN([dim, hidden, dim])

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        return self.kan(x.reshape(-1, d)).reshape(b, t, d)


class GELUFeedForward(nn.Module):
    """Rectified two-layer network (variant A)."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GLUFeedForward(nn.Module):
    """Gated linear unit (variant GA).

    One projection produces a value branch and a gate branch. The gate passes
    through a sigmoid-weighted linear unit and multiplies the value branch, so
    the layer can suppress its own output position by position.
    """

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, hidden * 2)
        self.out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(self.drop(value * F.silu(gate)))


def build_ffn(config: MemoryConfig) -> nn.Module:
    """Build the feed-forward sublayer named by `config.ffn_type`.

    Widths follow the usual convention: 4*d for the rectified unit, and
    8/3*d for the gated unit, which carries three weight matrices instead of
    two. The two therefore hold the same number of parameters.
    """
    dim = config.model_dim
    if config.ffn_type == "kan":
        return KANFeedForward(dim, config.kan_hidden)
    if config.ffn_type == "gelu":
        return GELUFeedForward(dim, dim * config.ffn_multiplier, config.dropout)
    hidden = int(dim * config.ffn_multiplier * 2 / 3)
    return GLUFeedForward(dim, hidden, config.dropout)


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------


def drop_path(x: Tensor, drop_prob: float = 0.0,
              training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Encoder block and stack
# ---------------------------------------------------------------------------


class MemoryBlock(nn.Module):
    """One survival-gated bounded-memory block, pre-normalised throughout."""

    def __init__(self, config: MemoryConfig, drop_path_rate: float = 0.0):
        super().__init__()
        dim = config.model_dim
        self.config = config

        self.norm_local = nn.LayerNorm(dim, eps=1e-5)
        self.local_attention = LocalSelfAttention(
            dim, config.num_heads, config.local_window, config.dropout)

        self.survival_gate = SurvivalGate(config)

        self.norm_bin = nn.LayerNorm(dim, eps=1e-5)
        self.bin_cross_attention = BinCrossAttention(
            dim, config.bin_dim, config.num_heads, config.dropout)

        self.norm_ffn = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = build_ffn(config)

        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0
                          else nn.Identity())
        self.dropout = nn.Dropout(config.dropout)

        self.slot_gru = nn.GRUCell(config.bin_dim, config.bin_dim)
        self.bin_norm = nn.LayerNorm(config.bin_dim)
        if config.bin_update == "attention":
            self.write_gate = nn.Linear(config.bin_dim * 2, config.bin_dim)
            self.write_value = nn.Linear(config.bin_dim, config.bin_dim)

    def forward(self, x: Tensor, bin_state: Tensor
                ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        # 1. Local windowed self-attention.
        local = self.local_attention(self.norm_local(x))
        x = x + self.drop_path(self.dropout(local))

        # 2. Score tokens, write the survivors into memory.
        candidates, gate, gate_prob, score, _ = self.survival_gate(x, bin_state)
        selected, selected_gates, selected_indices = self._select_candidates(
            candidates, gate, score)
        selected = selected * selected_gates.unsqueeze(-1)
        next_bin = self._update_bin(bin_state, selected)

        # 3. Read the memory back into every token.
        bin_read = self.bin_cross_attention(self.norm_bin(x), next_bin)
        x = x + self.drop_path(self.dropout(bin_read))

        # 4. Feed-forward sublayer, the only part that differs between variants.
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))

        aux = self._auxiliary_terms(gate, gate_prob, next_bin)
        aux["selected_indices"] = selected_indices.detach()
        return x, next_bin, aux

    def _select_candidates(self, candidates: Tensor, gate: Tensor,
                           score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, candidates.shape[1],
                self.config.bin_slots)
        selected_indices = torch.topk(score, k=k, dim=-1).indices
        gather_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, candidates.shape[-1])
        selected = torch.gather(candidates, dim=1, index=gather_index)
        selected_gate = torch.gather(gate, dim=1, index=selected_indices)
        return selected, selected_gate, selected_indices

    def _update_bin(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        if self.config.bin_update == "fifo":
            return self._fifo_update(bin_state, selected)
        if self.config.bin_update == "gru":
            return self._gru_update(bin_state, selected)
        return self._attention_update(bin_state, selected)

    def _fifo_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        aged = bin_state * self.config.bin_decay
        return torch.cat([selected, aged], dim=1)[:, :self.config.bin_slots, :]

    def _gru_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        batch, slots, dim = bin_state.shape
        writes = torch.zeros_like(bin_state)
        write_count = min(selected.shape[1], slots)
        writes[:, :write_count, :] = selected[:, :write_count, :]
        aged = bin_state * self.config.bin_decay
        updated = self.slot_gru(
            writes.reshape(batch * slots, dim),
            aged.reshape(batch * slots, dim),
        ).view(batch, slots, dim)
        write_mask = torch.zeros(batch, slots, 1, device=bin_state.device,
                                 dtype=bin_state.dtype)
        write_mask[:, :write_count, :] = 1.0
        next_bin = write_mask * updated + (1.0 - write_mask) * aged
        return self.bin_norm(next_bin)

    def _attention_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        if selected.shape[1] == 0:
            return bin_state * self.config.bin_decay
        aged = bin_state * self.config.bin_decay
        scores = torch.matmul(aged, selected.transpose(1, 2)) / (
            self.config.bin_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        write = torch.matmul(weights, selected)
        candidate = torch.tanh(self.write_value(write))
        gate = torch.sigmoid(self.write_gate(torch.cat([aged, candidate], dim=-1)))
        return self.bin_norm((1.0 - gate) * aged + gate * candidate)

    def _auxiliary_terms(self, gate: Tensor, gate_probability: Tensor,
                         bin_state: Tensor) -> Dict[str, Tensor]:
        denom = max(gate.numel(), 1)
        gate_mean = gate.sum() / denom
        p = gate_probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        gate_entropy = entropy.sum() / denom
        norm_bin = F.normalize(bin_state, dim=-1)
        similarity = torch.matmul(norm_bin, norm_bin.transpose(1, 2)).pow(2)
        slots = similarity.shape[-1]
        eye = torch.eye(slots, device=similarity.device, dtype=torch.bool)[None]
        slot_diversity_penalty = similarity.masked_fill(eye, 0.0).sum() / max(
            similarity.shape[0] * slots * max(slots - 1, 1), 1)
        return {
            "gate_mean": gate_mean,
            "gate_entropy": gate_entropy,
            "slot_diversity_penalty": slot_diversity_penalty,
        }


class MemoryEncoder(nn.Module):
    """Stack of survival-gated blocks sharing one bounded memory."""

    def __init__(self, config: MemoryConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.config = config
        depth = config.num_layers
        rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.initial_bin = nn.Parameter(
            torch.randn(config.bin_slots, config.bin_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [MemoryBlock(config, drop_path_rate=rates[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(config.model_dim, eps=1e-5)

    def forward(self, x: Tensor, return_trace: bool = False
                ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        """
        Args:
            x: (B, S, model_dim) token embeddings.
            return_trace: also return the token indices written to memory at
                each layer, for the memory write-log analysis.

        Returns:
            x: (B, S, model_dim) encoded tokens.
            auxiliary_terms: scalar regularisation terms averaged over layers.
            trace: list of (B, k) written token indices per layer, or None.
        """
        batch = x.shape[0]
        bin_state = self.initial_bin[None, :, :].expand(batch, -1, -1)

        aux_by_layer: List[Dict[str, Tensor]] = []
        trace: List[Tensor] = []
        for block in self.blocks:
            x, bin_state, aux = block(x, bin_state)
            aux_by_layer.append(aux)
            if return_trace:
                trace.append(aux["selected_indices"])

        x = self.norm(x)

        keys = [k for k in aux_by_layer[0] if k != "selected_indices"]
        auxiliary_terms = {
            key: torch.stack([a[key] for a in aux_by_layer]).mean()
            for key in keys
        }
        return x, auxiliary_terms, (trace if return_trace else None)

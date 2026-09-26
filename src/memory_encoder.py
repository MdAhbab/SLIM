"""
Survival-gated bounded-memory encoder.

The encoder replaces global self-attention with three steps per layer:

  1. Sliding-window self-attention, so each token mixes only with neighbours
     inside a window of `local_window` tokens. Scores are computed block by
     block, so the cost grows with sequence length times window rather than
     with the square of the length.
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

Memory behaviour
----------------
Five options correct places where the original encoder did not do what its
design describes. Each defaults to the original behaviour, so a checkpoint
trained before the option existed rebuilds and loads exactly. The variant
configurations switch on all of them except `gate_gradient`, for the reason
given there; `configs/slim_ka_legacy.yaml` switches them all off, for a run
that measures what the fixes changed.

    slot_addressing    "order": survivor j is written to slot j, so slots
                       k..bin_slots-1 never receive anything.
                       "content": survivors, in score order, each claim the
                       free slot whose content is most similar to theirs.
    gate_gradient      "selected": the task loss reaches the survival score
                       only at the tokens that were written.
                       "all": the write is a sum over every token weighted by
                       its straight-through gate. The forward pass is
                       unchanged, and the gradient reaches every score.
                       Needs content addressing. Not used by the variant
                       configurations: in short runs on the real data it made
                       the gate converge on the same positions for every input
                       faster than "selected" did.
    prediction_target  "pooled": one vector per sequence, decoded from the mean
                       slot and compared with every token; nothing trains it.
                       "position": each token is predicted from the memory by
                       a query built from its position alone, and a prediction
                       loss trains the predictor.
    memory_norm        "all": a LayerNorm over every slot after the update,
                       which undoes the decay applied to unwritten slots.
                       "written": only freshly written slots are normalised,
                       so an unwritten slot keeps shrinking by `bin_decay`.
    segment_lengths    None: the local window runs across the whole row.
                       A tuple of lengths: tokens attend only inside their own
                       segment, so the window stops at the seam between the
                       enhancer and promoter DNA and at the seam between DNA
                       and chromatin.

One further option addresses a failure the original design anticipated:

    selection_noise    0: the survivors are the top-k scores. A positive value
                       adds Gumbel noise of that scale to the scores used for
                       choosing survivors, in training only, which samples k
                       tokens without replacement from softmax(score / scale).
                       Tokens just outside the top-k are then written some of
                       the time, so the gate keeps receiving evidence about
                       them instead of locking onto the positions it chose
                       first. Evaluation always takes the plain top-k. In short
                       runs on the real data it did not reduce positional
                       collapse, so the variant configurations leave it at 0.

Two options exist only for ablations that take a part of the mechanism away:

    selection          "learned": survivors are the top-k survival scores.
                       "random": survivors are k tokens drawn uniformly at
                       random, a fresh draw for every input in training and a
                       fixed seeded draw in evaluation, so test predictions
                       reproduce. Tests whether the learned choice matters.
    read_back          True: every token reads the memory back.
                       False: the read-back is skipped, so no information
                       travels through the memory and the encoder reduces to
                       local attention plus the feed-forward sublayer. The
                       modules stay in place, so checkpoint shapes match.
"""

from __future__ import annotations

import math
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

    # Memory behaviour; see the module docstring. Defaults are the original
    # behaviour, so older checkpoints rebuild exactly.
    slot_addressing: str = "order"
    gate_gradient: str = "selected"
    prediction_target: str = "pooled"
    memory_norm: str = "all"
    segment_lengths: Optional[Tuple[int, ...]] = None
    selection_noise: float = 0.0

    # Ablations that remove part of the mechanism; see the module docstring.
    selection: str = "learned"
    read_back: bool = True

    def __post_init__(self) -> None:
        if self.bin_update not in {"fifo", "gru", "attention"}:
            raise ValueError("bin_update must be one of: fifo, gru, attention")
        if self.slot_addressing not in {"order", "content"}:
            raise ValueError("slot_addressing must be one of: order, content")
        if self.gate_gradient not in {"selected", "all"}:
            raise ValueError("gate_gradient must be one of: selected, all")
        if self.prediction_target not in {"pooled", "position"}:
            raise ValueError("prediction_target must be one of: pooled, position")
        if self.memory_norm not in {"all", "written"}:
            raise ValueError("memory_norm must be one of: all, written")
        if self.bin_update != "gru" and (self.slot_addressing == "content"
                                         or self.memory_norm == "written"):
            raise ValueError("content addressing and memory_norm='written' "
                             "apply to the gru update only")
        if self.gate_gradient == "all" and self.slot_addressing != "content":
            raise ValueError("gate_gradient='all' needs slot_addressing='content'")
        if self.selection_noise < 0:
            raise ValueError("selection_noise must be zero or positive")
        if self.selection not in {"learned", "random"}:
            raise ValueError("selection must be one of: learned, random")
        if self.segment_lengths is not None:
            self.segment_lengths = tuple(int(n) for n in self.segment_lengths)
            if not self.segment_lengths or min(self.segment_lengths) <= 0:
                raise ValueError("segment_lengths must be positive lengths")
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
    """Multi-head self-attention restricted to a sliding window.

    Token i attends to every token j with |i - j| <= local_window // 2, and,
    when segment ids are given, only to tokens in its own segment. Queries are
    taken in blocks of `radius` tokens, and each block is scored against the
    `3 * radius` keys its members' windows can reach. The score tensor
    therefore holds about 3 * radius * seq_len entries per head rather than
    seq_len squared, and the result equals masked full attention to
    floating-point precision.
    """

    def __init__(self, model_dim: int, num_heads: int, local_window: int,
                 dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.local_window = local_window
        self.radius = max(1, local_window // 2)
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, segment_ids: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, dim = x.shape
        radius = self.radius
        block = radius
        n_blocks = -(-seq_len // block)
        padded = n_blocks * block
        span = block + 2 * radius

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, seq_len, self.num_heads, self.head_dim)
                   .transpose(1, 2) for t in (q, k, v))
        # Queries padded at the end to whole blocks; keys and values padded by
        # one radius on each side, then cut into overlapping spans of keys.
        q = F.pad(q, (0, 0, 0, padded - seq_len)).reshape(
            batch, self.num_heads, n_blocks, block, self.head_dim)
        pad = (0, 0, radius, padded - seq_len + radius)
        k = F.pad(k, pad).unfold(2, span, block).transpose(-1, -2)
        v = F.pad(v, pad).unfold(2, span, block).transpose(-1, -2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._block_mask(seq_len, n_blocks, x.device, segment_ids)
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        y = torch.matmul(weights, v)
        y = y.reshape(batch, self.num_heads, padded, self.head_dim)[:, :, :seq_len]
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)

    def _block_mask(self, seq_len: int, n_blocks: int, device: torch.device,
                    segment_ids: Optional[Tensor]) -> Tensor:
        """(n_blocks, block, span) mask of the key positions each query keeps."""
        radius = self.radius
        block = radius
        starts = torch.arange(n_blocks, device=device)[:, None, None] * block
        query = starts + torch.arange(block, device=device)[None, :, None]
        key = starts - radius + torch.arange(
            block + 2 * radius, device=device)[None, None, :]
        mask = (((query - key).abs() <= radius) & (key >= 0) & (key < seq_len)
                & (query < seq_len))
        if segment_ids is not None:
            last = seq_len - 1
            mask = mask & (segment_ids[query.clamp(0, last)]
                           == segment_ids[key.clamp(0, last)])
        return mask


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


def sinusoidal_table(length: int, dim: int) -> Tensor:
    """(length, dim) sinusoidal position table, as in the input encoding."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


class SurvivalGate(nn.Module):
    """Scores tokens for survival into the bounded memory.

    The score of token i combines three terms:

        s_i = g(x_i, mean(B)) + alpha * novelty_i + beta * prediction_error_i

    where g is a learned two-layer network and novelty is one minus the highest
    cosine similarity between the compressed token and any memory slot. Each
    term can be switched off independently to isolate its contribution.

    The prediction error depends on `prediction_target`:

      "pooled"    the squared error between the token and a single vector
                  decoded from the mean slot, normalised by its mean over the
                  sequence. The original behaviour: the same vector is
                  compared with every token and no loss trains the decoder.
      "position"  the squared error between the token and a prediction read
                  from the memory by a query built from the token's position
                  alone, so it asks how poorly the current memory predicts
                  what sits at that position. The predictor is trained by
                  `prediction_loss`, returned alongside the score.
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
        if config.prediction_target == "position":
            self.position_query = nn.Linear(config.model_dim, config.bin_dim,
                                            bias=False)
            self.register_buffer(
                "positions", sinusoidal_table(config.max_length, config.model_dim),
                persistent=False)

    def forward(self, tokens: Tensor, bin_state: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor],
                           Dict[str, Tensor]]:
        batch, seq_len, _ = tokens.shape
        compressed = self.compress(tokens)
        bin_pool = bin_state.mean(dim=1)
        expanded_pool = bin_pool[:, None, :].expand(batch, seq_len, -1)

        learned_score = self.score_mlp(
            torch.cat([tokens, expanded_pool], dim=-1)).squeeze(-1)
        novelty = self._novelty(compressed, bin_state)
        prediction_loss = None
        if self.config.prediction_target == "position":
            prediction_error, prediction_loss = self._positional_prediction(
                tokens, bin_state)
        else:
            prediction_error = self._prediction_error(tokens, bin_pool)

        score = torch.zeros_like(learned_score)
        if self.config.use_learned_score:
            score = score + learned_score
        if self.config.use_novelty:
            score = score + self.config.novelty_weight * novelty
        if self.config.use_prediction_error:
            score = score + self.config.prediction_error_weight * prediction_error

        # The scores that choose the survivors. Gumbel noise in training
        # samples the survivors instead of always taking the same top-k.
        ranking = score
        if self.config.selection == "random":
            ranking = self._random_ranking(score)
        elif self.training and self.config.selection_noise > 0:
            uniform = torch.rand_like(score).clamp(1e-6, 1.0 - 1e-6)
            ranking = score - self.config.selection_noise * torch.log(
                -torch.log(uniform))

        gate, gate_probability = self._topk_gate(ranking, score)
        debug = {
            "score": score.detach(),
            "novelty": novelty.detach(),
            "prediction_error": prediction_error.detach(),
            "gate": gate.detach(),
            "gate_probability": gate_probability.detach(),
        }
        return (compressed, gate, gate_probability, ranking, prediction_loss,
                debug)

    def _random_ranking(self, score: Tensor) -> Tensor:
        """Scores that pick k tokens uniformly at random.

        A fresh draw per input in training. In evaluation the draw comes from
        a generator seeded the same way at every call, so the selection is
        random with respect to the input but identical between runs.
        """
        if self.training:
            return torch.rand_like(score, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(0)
        draw = torch.rand(score.shape, generator=generator)
        return draw.to(score.device)

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

    def _positional_prediction(self, tokens: Tensor, bin_state: Tensor
                               ) -> Tuple[Tensor, Tensor]:
        """Predict each token from the memory, queried by its position alone.

        Returns the normalised per-token error, used as a score term, and the
        mean error, used as the predictor's training loss.

        The memory and the tokens are detached here. The loss then trains only
        the predictor, so it cannot pull the memory or the tokens toward
        whatever is easy to predict, and the score term is a fixed signal of
        surprise rather than something the gate can game. The target is the
        token after a parameter-free layer norm, which keeps the loss on the
        same scale however the residual stream grows.
        """
        seq_len = tokens.shape[1]
        memory = bin_state.detach()
        query = self.position_query(self.positions[:seq_len])
        weights = F.softmax(torch.matmul(query, memory.transpose(1, 2))
                            / (self.config.bin_dim ** 0.5), dim=-1)
        prediction = self.predict_from_bin(torch.matmul(weights, memory))
        target = F.layer_norm(tokens.detach(), (tokens.shape[-1],))
        error = (target - prediction).pow(2).mean(dim=-1)
        surprise = error.detach()
        return (surprise / (surprise.mean(dim=-1, keepdim=True) + 1e-6),
                error.mean())

    def _topk_gate(self, ranking: Tensor, score: Tensor
                   ) -> Tuple[Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, score.shape[-1])
        topk = torch.topk(ranking, k=k, dim=-1).indices
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

    def forward(self, x: Tensor, bin_state: Tensor,
                segment_ids: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        # 1. Sliding-window self-attention.
        local = self.local_attention(self.norm_local(x), segment_ids)
        x = x + self.drop_path(self.dropout(local))

        # 2. Score tokens, write the survivors into memory. `ranking` is the
        # survival score, plus selection noise when that is switched on.
        (candidates, gate, gate_prob, ranking, prediction_loss,
         _) = self.survival_gate(x, bin_state)
        if self.config.slot_addressing == "content":
            next_bin, selected_indices, slot_indices = self._content_write(
                bin_state, candidates, gate, ranking)
        else:
            selected, selected_gates, selected_indices = self._select_candidates(
                candidates, gate, ranking)
            selected = selected * selected_gates.unsqueeze(-1)
            next_bin = self._update_bin(bin_state, selected)
            slot_indices = self._order_slots(selected_indices)

        # 3. Read the memory back into every token.
        if self.config.read_back:
            bin_read = self.bin_cross_attention(self.norm_bin(x), next_bin)
            x = x + self.drop_path(self.dropout(bin_read))

        # 4. Feed-forward sublayer, the only part that differs between variants.
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))

        aux = self._auxiliary_terms(gate, gate_prob, next_bin)
        if prediction_loss is not None:
            aux["prediction_loss"] = prediction_loss
        aux["selected_indices"] = selected_indices.detach()
        aux["slot_indices"] = slot_indices.detach()
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

    def _order_slots(self, selected_indices: Tensor) -> Tensor:
        """Slot taken by each survivor under order addressing, -1 if none."""
        batch, k = selected_indices.shape
        if self.config.bin_update == "attention":
            # Every slot is written softly, so no survivor owns a slot.
            return torch.full_like(selected_indices, -1)
        slots = torch.arange(k, device=selected_indices.device).expand(batch, k)
        return torch.where(slots < self.config.bin_slots, slots,
                           torch.full_like(slots, -1))

    def _gru_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        batch, slots, dim = bin_state.shape
        writes = torch.zeros_like(bin_state)
        write_count = min(selected.shape[1], slots)
        writes[:, :write_count, :] = selected[:, :write_count, :]
        write_mask = torch.zeros(batch, slots, 1, device=bin_state.device,
                                 dtype=bin_state.dtype)
        write_mask[:, :write_count, :] = 1.0
        return self._gru_write(bin_state, writes, write_mask)

    def _gru_write(self, bin_state: Tensor, writes: Tensor,
                   write_mask: Tensor) -> Tensor:
        """Gated recurrent update of the written slots; the rest only age.

        memory_norm="all" normalises every slot afterwards. That is the
        original behaviour, and it rescales the aged slots straight back,
        undoing the decay. memory_norm="written" normalises only the freshly
        written slots, so a slot that is not written keeps shrinking by
        `bin_decay` at every layer and contributes less when tokens read it.
        """
        batch, slots, dim = bin_state.shape
        aged = bin_state * self.config.bin_decay
        updated = self.slot_gru(
            writes.reshape(batch * slots, dim),
            aged.reshape(batch * slots, dim),
        ).view(batch, slots, dim)
        if self.config.memory_norm == "written":
            return write_mask * self.bin_norm(updated) + (1.0 - write_mask) * aged
        next_bin = write_mask * updated + (1.0 - write_mask) * aged
        return self.bin_norm(next_bin)

    def _content_write(self, bin_state: Tensor, candidates: Tensor,
                       gate: Tensor, score: Tensor
                       ) -> Tuple[Tensor, Tensor, Tensor]:
        """Write the survivors into the slots their content picks.

        Survivors are taken in score order, and each claims the free slot whose
        current content is most similar to its own (cosine). No two survivors
        share a slot within a layer, and any of the `bin_slots` slots can be
        written.

        The write is a sum over every token, weighted by its straight-through
        gate and a one-hot slot address. A token that did not survive has gate
        exactly 0 in the forward pass, so the result equals writing the
        survivors alone. Under gate_gradient="all" the backward pass still
        reaches the scores of the tokens that were not written, each through
        the slot it would have taken, which is what lets the gate learn from
        tokens it did not pick. Under "selected" those contributions carry no
        gradient, as in the original encoder.

        Returns the next memory, the survivor token indices (B, k) in score
        order, and the slot each survivor was written to (B, k).
        """
        batch, seq_len, dim = candidates.shape
        slots = bin_state.shape[1]
        k = min(self.config.survivors_per_layer, seq_len, slots)
        selected_indices = torch.topk(score, k=k, dim=-1).indices

        with torch.no_grad():
            similarity = torch.matmul(
                F.normalize(candidates.float(), dim=-1),
                F.normalize(bin_state.float(), dim=-1).transpose(1, 2))
            # Every token's preferred slot. Survivors are then reassigned so
            # that each takes a slot no earlier survivor has claimed.
            address = similarity.argmax(dim=-1)
            wanted = torch.gather(
                similarity, 1,
                selected_indices.unsqueeze(-1).expand(-1, -1, slots))
            taken = torch.zeros(batch, slots, dtype=torch.bool,
                                device=candidates.device)
            rows = torch.arange(batch, device=candidates.device)
            slot_indices = torch.empty(batch, k, dtype=torch.long,
                                       device=candidates.device)
            for j in range(k):
                pick = wanted[:, j].masked_fill(taken, float("-inf")).argmax(-1)
                slot_indices[:, j] = pick
                taken[rows, pick] = True
            address = address.scatter(1, selected_indices, slot_indices)
            one_hot = F.one_hot(address, slots).to(candidates.dtype)
            survived = torch.zeros(batch, seq_len, dtype=candidates.dtype,
                                   device=candidates.device).scatter(
                1, selected_indices, 1.0)
            # Survivors per slot, 0 or 1 since survivors never share a slot.
            count = (one_hot * survived.unsqueeze(-1)).sum(dim=1).clamp_min(1.0)

        weight = gate if self.config.gate_gradient == "all" else gate * survived
        routed = weight.to(candidates.dtype).unsqueeze(-1) * one_hot
        writes = torch.matmul(routed.transpose(1, 2), candidates) / count.unsqueeze(-1)
        write_mask = (routed.sum(dim=1) / count).unsqueeze(-1)
        next_bin = self._gru_write(bin_state, writes, write_mask)
        return next_bin, selected_indices, slot_indices

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

        # Segment of every token, for keeping the local window inside it.
        segment_ids = None
        if config.segment_lengths is not None:
            lengths = torch.tensor(config.segment_lengths)
            segment_ids = torch.repeat_interleave(
                torch.arange(len(lengths)), lengths)
        self.register_buffer("segment_ids", segment_ids, persistent=False)

    def forward(self, x: Tensor, return_trace: bool = False
                ) -> Tuple[Tensor, Dict[str, Tensor],
                           Optional[Dict[str, List[Tensor]]]]:
        """
        Args:
            x: (B, S, model_dim) token embeddings.
            return_trace: also return what each layer wrote to memory, for the
                memory write-log analysis.

        Returns:
            x: (B, S, model_dim) encoded tokens.
            auxiliary_terms: scalar regularisation terms averaged over layers.
            trace: None, or {"tokens": [...], "slots": [...]} with one (B, k)
                tensor per layer: the token indices written, and the slot each
                was written to (-1 where no single slot owns the write).
        """
        batch, seq_len = x.shape[:2]
        segment_ids = self.segment_ids
        if segment_ids is not None and segment_ids.numel() != seq_len:
            raise ValueError(
                f"segment_lengths {self.config.segment_lengths} cover "
                f"{segment_ids.numel()} tokens but the input has {seq_len}")
        bin_state = self.initial_bin[None, :, :].expand(batch, -1, -1)

        aux_by_layer: List[Dict[str, Tensor]] = []
        trace: Dict[str, List[Tensor]] = {"tokens": [], "slots": []}
        for block in self.blocks:
            x, bin_state, aux = block(x, bin_state, segment_ids)
            aux_by_layer.append(aux)
            if return_trace:
                trace["tokens"].append(aux["selected_indices"])
                trace["slots"].append(aux["slot_indices"])

        x = self.norm(x)

        traced = {"selected_indices", "slot_indices"}
        keys = [k for k in aux_by_layer[0] if k not in traced]
        auxiliary_terms = {
            key: torch.stack([a[key] for a in aux_by_layer]).mean()
            for key in keys
        }
        return x, auxiliary_terms, (trace if return_trace else None)

"""
Phase 2 additions to the chromatin branch and the interaction head, shared by
SLIM (src/slim_model.py) and the baseline (src/baseline_model.py).

Each is switched on under `model:` and is off by default, so a configuration
that does not name them builds the same model as before: the same parameters,
the same state-dict keys in the same order, and the same initialisation.

    chromatin_stem: groups   the chromatin tracks are read by one convolution
                             per functional group (accessibility, active marks,
                             promoter, transcription, repressive marks,
                             architecture) instead of one over every channel,
                             and a per-token gate weighs the groups
                             (GroupedChromatinStem). The gate's weights are
                             kept for interpretation.
    extrusion.enabled        a loop-extrusion model of the pair (LoopExtrusion):
                             learned barrier hazards for the two legs of a
                             cohesin loaded between the anchors, the chance
                             that both legs reach the anchors, the barrier
                             totals between them and the barrier strength at
                             each anchor.
    seq_compat.enabled       a direct path from the language-model embeddings
                             of the enhancer and the promoter to the head: a
                             low-rank product of the two plus one projection
                             of each (SequencePairPath).

The stem keeps the attribute name `epi_cnn` in both models, so masked
pretraining (scripts/pretrain.py) still finds it. The pair features enter the
interaction head through a linear layer without bias, initialised to zero and
added to the head's first linear layer, so a model with fresh weights scores
exactly as it would without them.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Functional groups of the chromatin tracks (model.chromatin_groups). A name
# that is not in data.feats_order is skipped, and a group left empty is
# dropped; a track that no group names is an error.
DEFAULT_CHROMATIN_GROUPS = {
    "accessibility": ["DNase"],
    "active": ["H3K27ac", "H3K4me1"],
    "promoter": ["H3K4me3"],
    "transcription": ["H3K36me3"],
    "repressive": ["H3K27me3", "H3K9me3"],
    "architecture": ["CTCF", "CTCFmotif_plus", "CTCFmotif_minus",
                     "YY1motif_plus", "YY1motif_minus"],
}

# Tracks the extrusion hazards may read (model.extrusion.tracks); those in
# data.feats_order are used. Stranded tracks come in _plus and _minus pairs.
DEFAULT_EXTRUSION_TRACKS = (
    "CTCF", "CTCFmotif_plus", "CTCFmotif_minus", "YY1motif_plus",
    "YY1motif_minus", "DNase", "H3K4me3", "H3K36me3",
)

# Value given to loading positions outside the anchors before the logsumexp.
# Finite, so no gradient becomes NaN; it also floors the contact term at
# about -1e4, far below any value a trained model reaches.
MASKED_SCORE = -1e4


def track_names(config: dict) -> List[str]:
    """Names of the chromatin tracks in channel order, from channel 1."""
    names = config["data"].get("feats_order")
    if names:
        return list(names)
    from src.slim_model import CHROMATIN_TRACKS
    return list(CHROMATIN_TRACKS[:config["data"]["n_epigenetic_features"] - 1])


def check_channel_count(config: dict, switch: str) -> None:
    """The new modules address tracks by name, so the channel count must be
    the position channel, the tracks and, with data.presence_channels, one
    presence channel per track."""
    data = config["data"]
    n_tracks = len(track_names(config))
    expected = 1 + n_tracks * (2 if data.get("presence_channels", False) else 1)
    if data["n_epigenetic_features"] != expected:
        raise ValueError(
            f"{switch} reads tracks by name, but data.n_epigenetic_features is "
            f"{data['n_epigenetic_features']} while the position channel and "
            f"{n_tracks} tracks in data.feats_order make {expected}")


# ---------------------------------------------------------------------------
# Chromatin stem
# ---------------------------------------------------------------------------

def chromatin_groups(config: dict) -> Dict[str, List[int]]:
    """Input channels of each chromatin group, in group order.

    Every group also reads channel 0, the position channel, and, with
    data.presence_channels, the presence channel of each of its tracks.
    """
    names = track_names(config)
    n_tracks = len(names)
    presence = config["data"].get("presence_channels", False)
    groups = config["model"].get("chromatin_groups") or DEFAULT_CHROMATIN_GROUPS
    owner = {}
    for group, members in groups.items():
        for name in members:
            if name in owner:
                raise ValueError(f"model.chromatin_groups puts {name} in both "
                                 f"{owner[name]} and {group}")
            owner[name] = group
    unassigned = [name for name in names if name not in owner]
    if unassigned:
        raise ValueError(f"model.chromatin_groups gives no group to {unassigned}; "
                         f"every track in data.feats_order needs one")
    channels = {}
    for group, members in groups.items():
        tracks = [names.index(name) for name in members if name in names]
        if not tracks:
            continue                       # none of its tracks is read here
        channels[group] = ([0] + [1 + j for j in tracks]
                           + ([1 + n_tracks + j for j in tracks] if presence else []))
    return channels


class GroupedChromatinStem(nn.Module):
    """One convolution per functional group of chromatin tracks, combined by a
    per-token gate.

    Each stem is the original stem (convolution, batch norm, leaky ReLU, max
    pooling) over its group's channels. The gate is a 1x1 convolution over the
    stacked stem outputs giving one weight per group and token, a softmax over
    the groups; the output is G * sum_g alpha_g s_g. The gate starts at zero,
    so every alpha is 1/G and the output is the sum of the stems. The last
    weights are kept, detached, in `last_alpha` (B, G, tokens).
    """

    def __init__(self, groups: Dict[str, List[int]], d_model: int, pool: int,
                 kernel_size: int = 11):
        super().__init__()
        self.group_names = list(groups)
        self.group_channels = [list(c) for c in groups.values()]
        for i, channels in enumerate(self.group_channels):
            self.register_buffer(f"channels_{i}", torch.tensor(channels),
                                 persistent=False)
        self.stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(len(channels), d_model, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.LeakyReLU(),
                nn.MaxPool1d(pool),
            )
            for channels in self.group_channels)
        n_groups = len(self.group_channels)
        self.gate = nn.Conv1d(n_groups * d_model, n_groups, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_alpha: Optional[Tensor] = None

    def stem_outputs(self, epi: Tensor) -> List[Tensor]:
        """Each group's stem output, (B, d_model, tokens)."""
        return [stem(epi.index_select(1, getattr(self, f"channels_{i}")))
                for i, stem in enumerate(self.stems)]

    def forward(self, epi: Tensor) -> Tensor:
        outs = self.stem_outputs(epi)
        alpha = torch.softmax(self.gate(torch.cat(outs, dim=1)), dim=1)
        self.last_alpha = alpha.detach()
        stacked = torch.stack(outs, dim=1)                     # (B, G, d, T)
        return len(outs) * (alpha.unsqueeze(2) * stacked).sum(dim=1)


def build_epi_stem(config: dict, d_model: int, pool: int) -> nn.Module:
    """The chromatin stem named by model.chromatin_stem.

    "single" (the default) is the original convolution over every channel,
    built exactly as before; "groups" is GroupedChromatinStem.
    """
    mode = config["model"].get("chromatin_stem", "single")
    if mode == "single":
        n_epi = config["data"]["n_epigenetic_features"]
        return nn.Sequential(
            nn.Conv1d(n_epi, d_model, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(pool),
        )
    if mode != "groups":
        raise ValueError("model.chromatin_stem must be single or groups")
    check_channel_count(config, "model.chromatin_stem groups")
    return GroupedChromatinStem(chromatin_groups(config), d_model, pool)


# ---------------------------------------------------------------------------
# Loop extrusion
# ---------------------------------------------------------------------------

def strand_partner(name: str) -> Optional[str]:
    """The other strand of a stranded track (CTCFmotif_plus and
    CTCFmotif_minus), or None for a track without a strand."""
    if name.endswith("_plus"):
        return name[:-len("_plus")] + "_minus"
    if name.endswith("_minus"):
        return name[:-len("_minus")] + "_plus"
    return None


class LoopExtrusion(nn.Module):
    """Loop extrusion between the two anchors of a pair, as pair features.

    A cohesin loaded at bin l between the anchors a < b extrudes with two
    legs: the left leg moves towards a, the right leg towards b. Each bin
    stops a leg with a hazard, a uniform base rate softplus(beta) (one per
    head, starting at the processivity given) plus a learned barrier term
    read from the chromatin and motif tracks:

        h^L = softplus(beta) + softplus(f(x))
        h^R = softplus(beta) + flip(softplus(f(flip(swap(x)))))

    where f is a small convolution (K heads) and swap exchanges the _plus and
    _minus tracks. A rightward leg is a leftward one on the other strand, read
    backwards, so the module is strand-equivariant: swapping the strands and
    reversing the window swaps the legs and the anchors. The learned term is
    zeroed within margin_bins of either anchor, where the anchors' own sites
    sit. With prefix sums C, a leg loaded at l survives to its anchor with
    probability exp(-H), with

        H^L_l = C^L_{l-1} - C^L_a,    H^R_l = C^R_{b-1} - C^R_l,

    and the contact term is the log mean, over uniform loading, of both legs
    arriving: logQ = logsumexp_l(-H^L_l - H^R_l) - log(b - a - 1). Anchors in
    the same or adjacent bins have logQ 0. Per head the features are logQ,
    the barrier totals T^L and T^R over the bins between the anchors, and the
    largest learned hazard of each leg within anchor_pool_bins of each anchor
    (convergent against divergent sites): 7 per head. The totals and strengths
    are taken as logs, so all features are on similar scales, and each
    feature is standardised on its own (BatchNorm) before Linear and GELU;
    a LayerNorm across them would let the largest feature swamp the anchor
    strengths.

    The anchors are taken as left and right, not as enhancer and promoter, so
    the output does not depend on which of the two comes first. Everything is
    O(B K N) and runs in the parameters' precision (fp32) with autocast off,
    the prefix sums in double precision.
    With barriers false, f is not built and the hazard is the base rate alone:
    a distance-decay control.
    """

    def __init__(self, channels: Sequence[int], swap: Sequence[int],
                 heads: int = 2, hidden: int = 16, barriers: bool = True,
                 margin_bins: int = 3, anchor_pool_bins: int = 5,
                 processivity_kb: float = 250.0, bin_kb: float = 0.5,
                 out_dim: int = 32):
        super().__init__()
        self.register_buffer("channels", torch.tensor(list(channels), dtype=torch.long),
                             persistent=False)
        self.register_buffer("swap", torch.tensor(list(swap), dtype=torch.long),
                             persistent=False)
        self.heads = heads
        self.barriers = barriers
        self.margin_bins = margin_bins
        self.anchor_pool_bins = anchor_pool_bins
        self.bin_kb = bin_kb
        self.out_dim = out_dim
        base = bin_kb / processivity_kb                # hazard per bin
        start = math.log(math.expm1(base))             # softplus(start) = base
        if barriers:
            self.hazard_net = nn.Sequential(
                nn.Conv1d(len(channels), hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, heads, kernel_size=1),
            )
            # The learned term starts at about the base rate, so the module
            # starts close to a uniform distance decay.
            nn.init.constant_(self.hazard_net[2].bias, start)
        self.beta = nn.Parameter(torch.full((heads,), start))
        n_features = 7 * heads
        self.norm = nn.BatchNorm1d(n_features)
        self.proj = nn.Linear(n_features, out_dim)
        self.act = nn.GELU()

    @property
    def processivity_kb(self) -> Tensor:
        """Learned processivity per head, bin size / base hazard, in kb."""
        return self.bin_kb / F.softplus(self.beta.detach())

    def learned_hazards(self, epi: Tensor):
        """The learned barrier hazards of the left and right legs, (B, K, N)
        each, before the anchor margin; zeros without barriers."""
        x = epi.index_select(1, self.channels).to(self.beta.dtype)
        if not self.barriers:
            zeros = x.new_zeros(x.shape[0], self.heads, x.shape[2])
            return zeros, zeros
        left = F.softplus(self.hazard_net(x))
        right = F.softplus(self.hazard_net(x.index_select(1, self.swap).flip(-1))).flip(-1)
        return left, right

    def terms(self, epi: Tensor, enh_idx: Tensor, prom_idx: Tensor) -> Dict[str, Tensor]:
        """Every intermediate: the hazards, the anchors and the raw features.

        Returns hazard_left and hazard_right (B, K, N, base plus the masked
        learned term), learned_left and learned_right (B, K, N, unmasked),
        left and right (B,) anchor bins, log_q, total_left and total_right
        (B, K), and features (B, 7K): logQ, T^L, T^R, then the left anchor's
        strengths for the left and right legs, then the right anchor's.
        """
        with torch.autocast(device_type=epi.device.type, enabled=False):
            batch, _, n_bins = epi.shape
            k = self.heads
            enh = enh_idx.reshape(batch).long().clamp(0, n_bins - 1)
            prom = prom_idx.reshape(batch).long().clamp(0, n_bins - 1)
            a, b = torch.minimum(enh, prom), torch.maximum(enh, prom)

            learned_left, learned_right = self.learned_hazards(epi)
            base = F.softplus(self.beta).view(1, k, 1)
            pos = torch.arange(n_bins, device=epi.device)
            near = (((pos - a[:, None]).abs() <= self.margin_bins)
                    | ((pos - b[:, None]).abs() <= self.margin_bins))
            keep = (~near).unsqueeze(1).to(base.dtype)                 # (B, 1, N)
            hazard_left = base + learned_left * keep
            hazard_right = base + learned_right * keep

            # Prefix sums with a leading zero: c[..., i] sums the bins before
            # i, so C_i (bins 0 to i) is c[..., i + 1]. They are taken in
            # double precision, so a strong barrier outside the anchors costs
            # no precision in the differences between them.
            dtype = hazard_left.dtype
            c_left = F.pad(hazard_left.double().cumsum(-1), (1, 0))
            c_right = F.pad(hazard_right.double().cumsum(-1), (1, 0))

            def at(c, index):
                return c.gather(-1, index.view(batch, 1, 1).expand(batch, k, 1)).squeeze(-1)

            leg_left = (c_left[..., :n_bins] - at(c_left, a + 1).unsqueeze(-1)
                        ).clamp(min=0).to(dtype)
            leg_right = (at(c_right, b).unsqueeze(-1) - c_right[..., 1:]
                         ).clamp(min=0).to(dtype)
            inside = ((pos > a[:, None]) & (pos < b[:, None])).unsqueeze(1)
            score = torch.where(inside, -(leg_left + leg_right),
                                torch.full_like(leg_left, MASKED_SCORE))
            n_inside = (b - a - 1).clamp(min=0).unsqueeze(-1)          # (B, 1)
            log_q = (torch.logsumexp(score, dim=-1)
                     - torch.log(n_inside.clamp(min=1).to(score.dtype)))
            log_q = torch.where(n_inside > 0, log_q, torch.zeros_like(log_q))
            total_left = (at(c_left, b) - at(c_left, a + 1)).clamp(min=0).to(dtype)
            total_right = (at(c_right, b) - at(c_right, a + 1)).clamp(min=0).to(dtype)

            reach = torch.arange(-self.anchor_pool_bins, self.anchor_pool_bins + 1,
                                 device=epi.device)

            def strength(learned, anchor):
                window = (anchor[:, None] + reach).clamp(0, n_bins - 1)
                return learned.gather(-1, window.unsqueeze(1).expand(batch, k, -1)).amax(-1)

            features = torch.cat([
                log_q, total_left, total_right,
                strength(learned_left, a), strength(learned_right, a),
                strength(learned_left, b), strength(learned_right, b),
            ], dim=1)
        return dict(hazard_left=hazard_left, hazard_right=hazard_right,
                    learned_left=learned_left, learned_right=learned_right,
                    left=a, right=b, log_q=log_q, total_left=total_left,
                    total_right=total_right, features=features)

    def forward(self, epi: Tensor, enh_idx: Tensor, prom_idx: Tensor) -> Tensor:
        features = self.terms(epi, enh_idx, prom_idx)["features"]
        k = self.heads
        with torch.autocast(device_type=epi.device.type, enabled=False):
            features = torch.cat([features[:, :k],
                                  torch.log1p(features[:, k:3 * k]),
                                  torch.log(features[:, 3 * k:].clamp(min=1e-6))], dim=1)
            return self.act(self.proj(self.norm(features)))


def build_extrusion(config: dict) -> LoopExtrusion:
    """LoopExtrusion over the configured tracks present in data.feats_order."""
    cfg = config["model"]["extrusion"]
    check_channel_count(config, "model.extrusion")
    names = track_names(config)
    wanted = cfg.get("tracks") or DEFAULT_EXTRUSION_TRACKS
    used = [name for name in wanted if name in names]
    barriers = bool(cfg.get("barriers", True))
    if barriers and not used:
        raise ValueError(f"model.extrusion reads none of its tracks {list(wanted)}; "
                         f"data.feats_order has {names}")
    swap = []
    for name in used:
        partner = strand_partner(name)
        if partner is None:
            swap.append(used.index(name))
        elif partner in used:
            swap.append(used.index(partner))
        else:
            raise ValueError(f"model.extrusion reads {name} but not {partner}; "
                             f"a stranded track needs both strands")
    return LoopExtrusion(
        channels=[1 + names.index(name) for name in used], swap=swap,
        heads=int(cfg.get("heads", 2)), hidden=int(cfg.get("hidden", 16)),
        barriers=barriers, margin_bins=int(cfg.get("margin_bins", 3)),
        anchor_pool_bins=int(cfg.get("anchor_pool_bins", 5)),
        processivity_kb=float(cfg.get("processivity_kb", 250)),
        bin_kb=config["data"].get("bin_size", 500) / 1000.0,
        out_dim=int(cfg.get("out_dim", 32)))


# ---------------------------------------------------------------------------
# Sequence pair path
# ---------------------------------------------------------------------------

class SequencePairPath(nn.Module):
    """The enhancer and promoter language-model embeddings, straight to the
    head.

    e and p are the means of the enhancer's and the promoter's embedded parts
    after one shared LayerNorm. The output is [U e * V p, P_e e, P_p p]: a
    low-rank multiplicative term (rank values) and one projection of each
    (unary values each). The DNA branch otherwise reaches the head only
    through the encoder's pooling and memory.
    """

    def __init__(self, dim: int, segments: int, rank: int = 32, unary: int = 16):
        super().__init__()
        self.segments = segments
        self.norm = nn.LayerNorm(dim)
        self.enh_low = nn.Linear(dim, rank, bias=False)
        self.prom_low = nn.Linear(dim, rank, bias=False)
        self.enh_proj = nn.Linear(dim, unary)
        self.prom_proj = nn.Linear(dim, unary)
        self.out_dim = rank + 2 * unary

    def forward(self, seq: Tensor) -> Tensor:
        """seq: (B, 2 * segments, dim), the enhancer's parts, then the promoter's."""
        e = self.norm(seq[:, :self.segments].mean(dim=1))
        p = self.norm(seq[:, self.segments:].mean(dim=1))
        return torch.cat([self.enh_low(e) * self.prom_low(p),
                          self.enh_proj(e), self.prom_proj(p)], dim=1)


def build_seq_compat(config: dict) -> SequencePairPath:
    cfg = config["model"]["seq_compat"]
    data = config["data"]
    if data.get("dna_input", "pocd") != "embedding":
        raise ValueError("model.seq_compat reads the language-model embeddings of "
                         "the enhancer and promoter, so it needs data.dna_input "
                         "embedding")
    # With data.modalities "epi" the DNA input is zeros, as for the DNA
    # branch, and the path gives a constant; it is still built, because
    # scripts/pretrain.py sets modalities "epi" on the fine-tuning config.
    return SequencePairPath(int(data["dna_embedding_dim"]), int(data["dna_segments"]),
                            rank=int(cfg.get("rank", 32)), unary=int(cfg.get("unary", 16)))


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------

class PairFeatures(nn.Module):
    """The enabled pair modules; the output is their features, concatenated
    (B, out_dim), in fp32."""

    def __init__(self, extrusion: Optional[LoopExtrusion],
                 seq_compat: Optional[SequencePairPath]):
        super().__init__()
        self.extrusion = extrusion
        self.seq_compat = seq_compat
        self.out_dim = sum(m.out_dim for m in (extrusion, seq_compat) if m is not None)

    def forward(self, seq: Tensor, epi: Tensor, enh_idx: Tensor,
                prom_idx: Tensor) -> Tensor:
        parts = []
        if self.extrusion is not None:
            parts.append(self.extrusion(epi, enh_idx, prom_idx))
        if self.seq_compat is not None:
            parts.append(self.seq_compat(seq))
        return torch.cat([p.float() for p in parts], dim=1)


def build_pair_features(config: dict) -> Optional[PairFeatures]:
    """The pair modules switched on by model.extrusion.enabled and
    model.seq_compat.enabled, or None when both are off."""
    model_cfg = config["model"]
    extrusion = seq_compat = None
    if (model_cfg.get("extrusion") or {}).get("enabled", False):
        extrusion = build_extrusion(config)
    if (model_cfg.get("seq_compat") or {}).get("enabled", False):
        seq_compat = build_seq_compat(config)
    if extrusion is None and seq_compat is None:
        return None
    return PairFeatures(extrusion, seq_compat)


def pair_head(pair_features: PairFeatures, width: int) -> nn.Linear:
    """The linear layer that adds the pair features to the head, without bias
    and initialised to zero."""
    layer = nn.Linear(pair_features.out_dim, width, bias=False)
    nn.init.zeros_(layer.weight)
    return layer

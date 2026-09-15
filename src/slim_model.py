"""
SLIM: two-branch encoder-decoder for enhancer-promoter interaction.

One model serves all three variants. The sequence branch, the chromatin
branch, the pooling and the prediction heads are shared, and the only thing
that changes between variants is the feed-forward sublayer inside the
survival-gated encoder, chosen by `model.ffn_type`:

    ffn_type: gelu   variant A
    ffn_type: kan    variant KA
    ffn_type: glu    variant GA

Because the variants are one class rather than three files, they cannot drift
apart. `tests/test_variants_matched.py` checks this by construction.

Shape flow:

    DNA sequence (64 x 6000)  -> CNN -> BiLSTM  -> 128 tokens x 180
    Chromatin (9 x 5000)      -> CNN -> BiLSTM  -> 500 tokens x 180
                              concatenate       -> 628 tokens x 180
                              survival-gated encoder
                              structured self-attention pooling
    [enhancer, promoter, attention-mean, attention-max] -> 720
                              interaction head, distance head
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.model_layers import (
    KANLinear, PositionalEncoding, SelfAttentionPooling,
)
from src.memory_encoder import MemoryConfig, MemoryEncoder

# Names of the chromatin tracks in channel order, after the position channel.
CHROMATIN_TRACKS = (
    "CTCF", "DNase", "H3K27ac", "H3K27me3",
    "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3",
)


def build_epi_channel_mask(n_channels: int, modalities: str,
                           keep_tracks: Optional[Sequence[str]] = None) -> Tensor:
    """Build a (1, n_channels, 1) mask over the chromatin branch input.

    Channel 0 carries the symmetric log-distance profile that marks the
    enhancer and promoter positions. Channels 1 onward carry the chromatin
    tracks in `CHROMATIN_TRACKS` order.

    modalities:
        "all"      every channel, the default.
        "seq"      chromatin branch input fully zeroed, so the model sees
                   DNA sequence only.
        "seq+pos"  position channel kept, all chromatin tracks zeroed.

    `keep_tracks` optionally restricts which chromatin tracks survive, for
    reintroducing marks one at a time.
    """
    mask = torch.zeros(1, n_channels, 1)
    if modalities == "seq":
        return mask
    mask[0, 0, 0] = 1.0  # the position channel
    if modalities == "seq+pos":
        return mask
    if modalities != "all":
        raise ValueError("modalities must be one of: all, seq, seq+pos")
    if keep_tracks is None:
        mask[:] = 1.0
        return mask
    unknown = set(keep_tracks) - set(CHROMATIN_TRACKS)
    if unknown:
        raise ValueError(f"unknown chromatin tracks: {sorted(unknown)}")
    for i, name in enumerate(CHROMATIN_TRACKS, start=1):
        if i < n_channels and name in keep_tracks:
            mask[0, i, 0] = 1.0
    return mask


class SLIM(nn.Module):
    """Two-branch model with the survival-gated memory encoder."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        d_model = model_cfg["hidden_dim"]
        n_seq_tokens = model_cfg.get("n_tokens", 128)
        drop = model_cfg.get("dropout", 0.1)
        depth = model_cfg.get("num_layers", 3)
        num_heads = model_cfg.get("num_heads", 6)
        kan_hidden = model_cfg.get("kan_hidden", 64)

        self.n_seq_tokens = n_seq_tokens
        self.epi_pool_factor = 10  # 5000 bins / 10 = 500 tokens
        self.n_epi_tokens = config["data"]["epigenetic_bins"] // self.epi_pool_factor
        self.d_model = d_model

        # Sequence branch: position-aware encoded DNA, 64 channels.
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_seq_tokens),
        )
        self.seq_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.seq_drop = nn.Dropout(drop)

        # Chromatin branch: one position channel plus eight tracks.
        n_epi = config["data"]["n_epigenetic_features"]
        self.epi_cnn = nn.Sequential(
            nn.Conv1d(n_epi, d_model, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.epi_pool_factor),
        )
        self.epi_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.epi_drop = nn.Dropout(drop)

        # Which input channels the chromatin branch may see. Registered as a
        # buffer so it travels with the checkpoint and applies in both
        # training and evaluation.
        self.modalities = config["data"].get("modalities", "all")
        self.register_buffer(
            "epi_channel_mask",
            build_epi_channel_mask(
                n_epi, self.modalities,
                config["data"].get("keep_chromatin_tracks"),
            ),
            persistent=True,
        )

        total_tokens = n_seq_tokens + self.n_epi_tokens
        self.proj_drop = nn.Dropout(drop)
        self.pos_enc = PositionalEncoding(d_model, max_len=total_tokens + 2)

        # Survival-gated bounded-memory encoder.
        memory_cfg = model_cfg.get("memory", {})
        self.memory_config = MemoryConfig(
            model_dim=d_model,
            num_layers=memory_cfg.get("num_layers", depth),
            num_heads=num_heads,
            local_window=memory_cfg.get("local_window", 64),
            bin_slots=memory_cfg.get("bin_slots", 16),
            bin_dim=memory_cfg.get("bin_dim", 96),
            survivors_per_layer=memory_cfg.get("survivors_per_layer", 8),
            dropout=drop,
            gate_temperature=memory_cfg.get("gate_temperature", 1.0),
            novelty_weight=memory_cfg.get("novelty_weight", 0.35),
            prediction_error_weight=memory_cfg.get("prediction_error_weight", 0.20),
            bin_decay=memory_cfg.get("bin_decay", 0.92),
            bin_update=memory_cfg.get("bin_update", "gru"),
            max_length=total_tokens + 2,
            ffn_type=model_cfg.get("ffn_type", "kan"),
            kan_hidden=kan_hidden,
            ffn_multiplier=memory_cfg.get("ffn_multiplier", 4),
            use_learned_score=memory_cfg.get("use_learned_score", True),
            use_novelty=memory_cfg.get("use_novelty", True),
            use_prediction_error=memory_cfg.get("use_prediction_error", True),
        )
        self.encoder = MemoryEncoder(
            self.memory_config,
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        )

        # Structured self-attention pooling, with the Frobenius penalty that
        # keeps its heads distinct.
        self.att_pool = SelfAttentionPooling(
            d_model, da=model_cfg.get("sa_da", 64), r=model_cfg.get("sa_r", 32))
        pool_dim = d_model * 4

        self.head_drop = nn.Dropout(model_cfg.get("head_dropout", 0.2))
        self.fc_linear = nn.Linear(pool_dim, 128)
        self.fc_kan1 = KANLinear(128, 64)
        self.fc_kan2 = KANLinear(64, 1)

        self.dist_kan1 = KANLinear(pool_dim, d_model)
        self.dist_kan2 = KANLinear(d_model, 1)

    @property
    def variant(self) -> str:
        """Short label for the feed-forward sublayer in use."""
        return {"gelu": "A", "kan": "KA", "glu": "GA"}[self.memory_config.ffn_type]

    def forward(self, seq: Tensor, epi: Tensor, enh_idx: Tensor,
                prom_idx: Tensor, return_trace: bool = False):
        """
        Args:
            seq: (B, 64, L) position-aware encoded DNA.
            epi: (B, n_epi, 5000) chromatin tracks plus the position channel.
            enh_idx: (B,) or (B, 1) enhancer bin index in the 5000-bin window.
            prom_idx: (B,) or (B, 1) promoter bin index.
            return_trace: also return the token indices written to memory.

        Returns:
            cls_out: (B, 1) interaction logits.
            reg_out: (B, 1) predicted log genomic distance.
            A: (B, r, S) pooling attention, for the Frobenius penalty.
            aux: survival-gate regularisation terms.
            trace: written token indices per layer when requested.
        """
        batch = seq.size(0)

        x_seq = self.seq_cnn(seq).permute(0, 2, 1)
        x_seq, _ = self.seq_bilstm(x_seq)
        x_seq = self.seq_drop(x_seq)

        epi = epi * self.epi_channel_mask.to(epi.dtype)
        x_epi = self.epi_cnn(epi).permute(0, 2, 1)
        x_epi, _ = self.epi_bilstm(x_epi)
        x_epi = self.epi_drop(x_epi)

        z = torch.cat([x_seq, x_epi], dim=1)
        z = self.pos_enc(self.proj_drop(z))

        z, aux, trace = self.encoder(z, return_trace=return_trace)

        pooled, A = self.att_pool(z)

        enh_token = self.n_seq_tokens + torch.div(
            enh_idx.long().view(batch), self.epi_pool_factor,
            rounding_mode="trunc")
        prom_token = self.n_seq_tokens + torch.div(
            prom_idx.long().view(batch), self.epi_pool_factor,
            rounding_mode="trunc")
        max_idx = self.n_seq_tokens + self.n_epi_tokens - 1
        enh_token = enh_token.clamp(self.n_seq_tokens, max_idx)
        prom_token = prom_token.clamp(self.n_seq_tokens, max_idx)

        rows = torch.arange(batch, device=z.device)
        enh_feat = z[rows, enh_token, :]
        prom_feat = z[rows, prom_token, :]

        z_pool = torch.cat(
            [enh_feat, prom_feat, pooled.mean(dim=1), pooled.max(dim=1)[0]],
            dim=1,
        )

        feats = self.head_drop(z_pool)
        cls_out = self.fc_kan2(self.fc_kan1(self.fc_linear(feats)))

        dist_feats = self.head_drop(z_pool)
        reg_out = self.dist_kan2(self.dist_kan1(dist_feats))

        if return_trace:
            return cls_out, reg_out, A, aux, trace
        return cls_out, reg_out, A, aux

    def attention_penalty(self, A: Tensor) -> Tensor:
        """Frobenius penalty keeping the pooling heads distinct."""
        return self.att_pool.penalization_term(A)

    def memory_auxiliary_loss(self, aux: Dict[str, Tensor],
                            entropy_weight: float = 0.01,
                            diversity_weight: float = 0.05) -> Tensor:
        """Regularisation for the survival gate and the memory.

        Gate entropy is maximised, which stops the gate collapsing onto the
        same positions every time. The slot-diversity penalty is minimised,
        which stops the memory slots becoming copies of one another.
        """
        return (-entropy_weight * aux["gate_entropy"]
                + diversity_weight * aux["slot_diversity_penalty"])


def build_model(config: dict) -> nn.Module:
    """Build the model named by `model.variant` in the config.

    "baseline" gives the global KAN-Transformer; "A", "KA" and "GA" give the
    survival-gated encoder with the matching feed-forward sublayer.
    """
    variant = str(config["model"].get("variant", "KA"))
    if variant.lower() == "baseline":
        from src.baseline_model import Kansformer
        return Kansformer(config)
    ffn_by_variant = {"a": "gelu", "ka": "kan", "ga": "glu"}
    key = variant.lower()
    if key not in ffn_by_variant:
        raise ValueError(
            f"unknown variant {variant!r}; expected baseline, A, KA or GA")
    config = dict(config)
    config["model"] = dict(config["model"])
    config["model"]["ffn_type"] = ffn_by_variant[key]
    return SLIM(config)

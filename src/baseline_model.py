import torch
import torch.nn as nn
import math
from src.model_layers import (
    KANLinear, KAN, PositionalEncoding, SelfAttentionPooling,
)
from src.slim_model import build_epi_channel_mask, sequence_input_scale


# ---------------------------------------------------------------------------
# DropPath (stochastic depth) — matches reference kanformer.py
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention — matches reference kanformer.py
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# KAN-Transformer Block — PRE-NORM with KAN replacing FFN
# Matches reference: x = x + DropPath(Attn(LN(x)))
#                    x = x + DropPath(KAN(LN(x).reshape(-1,d)).reshape(b,t,d))
# ---------------------------------------------------------------------------
class KANBlock(nn.Module):
    """Single Transformer encoder layer with KAN replacing FFN (pre-norm)."""
    def __init__(self, dim, num_heads, kan_hidden=64, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        # KAN([dim, kan_hidden, dim]) replaces the FFN
        self.kan = KAN([dim, kan_hidden, dim])

    def forward(self, x):
        b, t, d = x.shape
        # Pre-norm attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # Pre-norm KAN (must reshape to 2D for KANLinear)
        x = x + self.drop_path(
            self.kan(self.norm2(x).reshape(-1, d)).reshape(b, t, d)
        )
        return x


# ---------------------------------------------------------------------------
# KAN-Transformer Encoder — stacks KANBlocks with linearly increasing drop path
# ---------------------------------------------------------------------------
class KANTransformer(nn.Module):
    def __init__(self, embed_dim=180, depth=3, num_heads=6, kan_hidden=64,
                 qkv_bias=True, drop=0.1, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            KANBlock(
                dim=embed_dim, num_heads=num_heads, kan_hidden=kan_hidden,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path_rate=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# POCD-KansformerEPI baseline — Dual-branch with enh/prom index feature extraction
#
# Architecture notes:
#   1. Epi CNN: single layer + MaxPool1d(10) → 500 tokens
#   2. Epi BiLSTM (matches reference)
#   3. Enh/prom index extraction from transformer output (like reference)
#   4. FC head input: 4*d_model=720
#   5. Total tokens: 128 seq + 500 epi = 628
# ---------------------------------------------------------------------------
class Kansformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['model']['hidden_dim']       # 180
        n_seq_tokens = config['model'].get('n_tokens', 128)
        drop = config['model'].get('dropout', 0.1)
        depth = config['model'].get('num_layers', 3)
        num_heads = config['model'].get('num_heads', 6)
        kan_hidden = config['model'].get('kan_hidden', 64)

        self.n_seq_tokens = n_seq_tokens
        self.epi_pool_factor = 10  # MaxPool1d kernel → 5000/10 = 500 epi tokens
        self.n_epi_tokens = config['data']['epigenetic_bins'] // self.epi_pool_factor  # 500
        self.d_model = d_model

        # --- Sequence Branch (POCD-ND encoded: 64 channels x L) ---
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

        # Seq BiLSTM (2 layers, bidirectional)
        self.seq_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.seq_drop = nn.Dropout(drop)

        # --- Epigenetic Branch (n_epi channels x 5000 bins) ---
        # Single CNN layer + MaxPool1d(10) → 500 tokens (matches reference)
        n_epi = config['data']['n_epigenetic_features']  # 9 (8 epi + 1 pos_enc)
        self.epi_cnn = nn.Sequential(
            nn.Conv1d(n_epi, d_model, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.epi_pool_factor),  # 5000 → 500 tokens
        )

        # Epi BiLSTM (matches reference)
        self.epi_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.epi_drop = nn.Dropout(drop)

        # --- Fusion + Positional Encoding ---
        total_tokens = n_seq_tokens + self.n_epi_tokens  # 128 + 500 = 628
        self.proj_drop = nn.Dropout(drop)
        self.pos_enc = PositionalEncoding(d_model, max_len=total_tokens + 2)

        # --- KAN-Transformer Encoder (KAN replaces FFN) ---
        # Same stochastic-depth rate as the survival-gated encoder reads from
        # the config, so the two encoders are regularised alike.
        self.transformer = KANTransformer(
            embed_dim=d_model, depth=depth, num_heads=num_heads,
            kan_hidden=kan_hidden, qkv_bias=True, drop=drop,
            attn_drop=0.0,
            drop_path_rate=config['model'].get('drop_path_rate', 0.0),
        )

        # Input ablations, identical to SLIM's. Not persistent, so checkpoints
        # saved before these existed load unchanged.
        modalities = config['data'].get('modalities', 'all')
        self.register_buffer(
            'epi_channel_mask',
            build_epi_channel_mask(n_epi, modalities,
                                   config['data'].get('keep_chromatin_tracks')),
            persistent=False)
        self.register_buffer('seq_input_scale', sequence_input_scale(modalities),
                             persistent=False)

        # --- Self-Attention Pooling ---
        sa_da = config['model'].get('sa_da', 64)
        sa_r = config['model'].get('sa_r', 32)
        self.att_pool = SelfAttentionPooling(d_model, da=sa_da, r=sa_r)

        # Pool: [enh_feat, prom_feat, attn_mean, attn_max] = 4 * d_model
        pool_dim = d_model * 4  # 720 (was 360)

        # --- Classification Head (matches reference structure) ---
        self.head_drop = nn.Dropout(config['model'].get('head_dropout', 0.2))
        self.fc_linear = nn.Linear(pool_dim, 128)
        self.fc_kan1 = KANLinear(128, 64)
        self.fc_kan2 = KANLinear(64, 1)

        # --- Distance Regression Head ---
        self.dist_kan1 = KANLinear(pool_dim, d_model)
        self.dist_kan2 = KANLinear(d_model, 1)

    def forward(self, seq, epi, enh_idx, prom_idx):
        B = seq.size(0)

        seq = seq * self.seq_input_scale.to(seq.dtype)
        epi = epi * self.epi_channel_mask.to(epi.dtype)

        # --- Sequence branch ---
        x_seq = self.seq_cnn(seq)                # (B, d_model, n_seq_tokens)
        x_seq = x_seq.permute(0, 2, 1)          # (B, n_seq_tokens, d_model)
        x_seq, _ = self.seq_bilstm(x_seq)       # (B, n_seq_tokens, d_model)
        x_seq = self.seq_drop(x_seq)

        # --- Epigenetic branch ---
        x_epi = self.epi_cnn(epi)                # (B, d_model, n_epi_tokens)
        x_epi = x_epi.permute(0, 2, 1)          # (B, n_epi_tokens, d_model)
        x_epi, _ = self.epi_bilstm(x_epi)       # (B, n_epi_tokens, d_model)
        x_epi = self.epi_drop(x_epi)

        # --- Concat + positional encoding ---
        z = torch.cat([x_seq, x_epi], dim=1)     # (B, n_seq + n_epi, d_model)
        z = self.proj_drop(z)
        z = self.pos_enc(z)

        # --- KAN-Transformer ---
        z = self.transformer(z)                   # (B, n_seq + n_epi, d_model)

        # --- Self-Attention Pooling ---
        M, A = self.att_pool(z)                   # M: (B, r, d_model), A: (B, r, S)

        # --- Enhancer / Promoter index-based feature extraction ---
        # Map bin indices (5000-space) to token indices in the epi portion of z
        enh_token = self.n_seq_tokens + torch.div(
            enh_idx.long().view(B), self.epi_pool_factor, rounding_mode='trunc')
        prom_token = self.n_seq_tokens + torch.div(
            prom_idx.long().view(B), self.epi_pool_factor, rounding_mode='trunc')

        # Clamp to valid range
        max_idx = self.n_seq_tokens + self.n_epi_tokens - 1
        enh_token = enh_token.clamp(self.n_seq_tokens, max_idx)
        prom_token = prom_token.clamp(self.n_seq_tokens, max_idx)

        # Gather features at enh/prom positions (per-sample indexing)
        batch_idx = torch.arange(B, device=z.device)
        enh_feat = z[batch_idx, enh_token, :]     # (B, d_model)
        prom_feat = z[batch_idx, prom_token, :]   # (B, d_model)

        # Attention-weighted mean and max
        attn_mean = M.mean(dim=1)                 # (B, d_model)
        attn_max = M.max(dim=1)[0]                # (B, d_model)

        z_pool = torch.cat([enh_feat, prom_feat, attn_mean, attn_max], dim=1)  # (B, 4*d_model)

        # --- Classification head ---
        feats = self.head_drop(z_pool)
        feats = self.fc_linear(feats)             # (B, 128)
        feats = self.fc_kan1(feats)               # (B, 64)
        cls_out = self.fc_kan2(feats)             # (B, 1)

        # --- Distance head ---
        dist_feats = self.head_drop(z_pool)
        dist_feats = self.dist_kan1(dist_feats)   # (B, d_model)
        reg_out = self.dist_kan2(dist_feats)      # (B, 1)

        return cls_out, reg_out, A

    def attention_penalty(self, A):
        return self.att_pool.penalization_term(A)
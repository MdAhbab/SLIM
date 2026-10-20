"""
Masked pretraining of the chromatin encoder (Phase 2, Stage 3).

Windows come from every epigenome except the test cells, the validation cells
(training.valid_cells) and data.never_train, so the chromatin that validation
and test score is never seen. Each window is placed like a labelled one: its
enhancer and gene start are those of a pair in the training files, and its
chromatin comes from a cell drawn at random, so epigenomes without labels
contribute too. Labels are never read.

Two parts of each window are hidden and predicted from the rest:

  spans   about pretrain.mask_fraction of the chromatin tokens (5 kb each), in
          runs of 1 to pretrain.max_span tokens, on every track;
  a mark  with pretrain.hide_mark, one track drawn at random, over the whole
          window, so a mark is predicted from the others.

Hidden bins are set to -1, a value the window normalisation never produces. A
linear head reads the encoder's chromatin tokens and predicts, per token, the
mean of each track over its bins; the loss is the mean squared error over the
hidden values. For SLIM the memory's own losses (gate entropy, slot diversity
and the token predictor) are added with their training weights, so the
survival gate learns what to write before it sees a label. The baseline is
pretrained the same way, without those terms.

Only the chromatin branch and the encoder are trained here and handed on; the
DNA branch is off (data.modalities "epi"). After every epoch the encoder
weights, the head and the history are written to <output-dir>/pretrained.pt;
`train.py --init <that file>` starts fine-tuning from it.

    python scripts/pretrain.py --config configs/phase2_p1a.yaml --output-dir runs/pre_ga
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bengi import PairTable, bengi_files  # noqa: E402
from src.config import load_config  # noqa: E402
from src.epi_data_pipeline import EPIGenomicDataset, dataset_options  # noqa: E402
from src.shutdown import run_main  # noqa: E402
from src.slim_model import build_model  # noqa: E402
from train import (DEFAULT_TEST_CELLS, DEFAULT_TRAIN_CELLS, build_scheduler,  # noqa: E402
                   seed_worker, set_seed, tracked_cells, training_cells)

# Parameters trained here and handed to fine-tuning: the chromatin branch and
# the encoder (SLIM's survival-gated memory or the baseline's transformer).
PRETRAINED_PREFIXES = ("epi_cnn.", "epi_bilstm.", "encoder.", "transformer.")
MASKED = -1.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--variant", default=None, help="baseline, A, KA or GA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def corpus_cells(config, test_cells) -> list:
    """Epigenomes with every track, except the test, validation and
    never-train cells."""
    banned = (set(test_cells) | set(config["training"].get("valid_cells") or [])
              | set(config["data"].get("never_train") or []))
    return sorted(tracked_cells(config) - banned)


def window_table(table: PairTable, cells, n: int, seed: int) -> PairTable:
    """`n` windows: the enhancer and gene start of a random row of `table`,
    the chromatin of a random cell in `cells`. Labels are zero (unused)."""
    c = table.columns
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(table), size=n)
    start, end = c["enh_start"][pick], c["enh_end"][pick]
    tss = c["tss_coord"][pick]
    zeros = np.zeros(n, dtype=np.int32)
    columns = {"label": np.zeros(n, dtype=np.int8),
               "dist": np.abs((start.astype(np.int64) + end) // 2 - tss).astype(np.float32),
               "enh_start": start, "enh_end": end, "tss_coord": tss,
               "chrom": c["chrom"][pick],
               "cell": rng.integers(len(cells), size=n).astype(np.int32),
               "assay": zeros, "source": zeros, "enh": zeros, "gene": zeros}
    vocab = {"chrom": table.vocab["chrom"], "cell": list(cells), "assay": ["none"],
             "source": [""], "enh": ["-"], "gene": ["-"]}
    return PairTable(columns, vocab)


def span_mask(rng, n_tokens: int, fraction: float, max_span: int) -> np.ndarray:
    """Runs of 1 to `max_span` tokens until `fraction` of them are covered."""
    mask = np.zeros(n_tokens, dtype=bool)
    goal = max(1, int(round(fraction * n_tokens)))
    while mask.sum() < goal:
        length = int(rng.integers(1, min(max_span, n_tokens) + 1))
        start = int(rng.integers(0, n_tokens - length + 1))
        mask[start:start + length] = True
    return mask


class MaskedWindows(Dataset):
    """Windows of an EPIGenomicDataset with hidden spans and a hidden mark.

    Each item has `epi` (the input, hidden bins at -1), `target` (n_tracks,
    n_tokens: each track's mean over a token's bins, before hiding) and
    `hidden` (n_tracks, n_tokens, True where the target was hidden). With
    `fixed_seed` the hidden parts of item i are the same every time.
    """

    def __init__(self, source, n_tracks: int, pool: int, fraction: float,
                 max_span: int, hide_mark: bool, fixed_seed=None):
        self.source = source
        self.n_tracks = n_tracks
        self.pool = pool
        self.fraction = fraction
        self.max_span = max_span
        self.hide_mark = hide_mark
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        epi = self.source[i]["epi"].clone()
        rng = np.random.default_rng(None if self.fixed_seed is None
                                    else (self.fixed_seed, int(i)))
        tracks = epi[1:1 + self.n_tracks]
        n_tokens = tracks.shape[1] // self.pool
        target = tracks[:, :n_tokens * self.pool].reshape(
            self.n_tracks, n_tokens, self.pool).mean(dim=2)
        hidden = torch.zeros(self.n_tracks, n_tokens, dtype=torch.bool)
        hidden[:, torch.from_numpy(span_mask(rng, n_tokens, self.fraction,
                                             self.max_span))] = True
        if self.hide_mark:
            hidden[int(rng.integers(self.n_tracks))] = True
        bins = hidden.repeat_interleave(self.pool, dim=1)
        tracks[:, :bins.shape[1]][bins] = MASKED
        return {"epi": epi, "target": target, "hidden": hidden}


def trained_parameters(model, head):
    """(name, parameter) pairs that pretraining updates."""
    named = [(n, p) for n, p in model.named_parameters()
             if n.startswith(PRETRAINED_PREFIXES)]
    return named + [(f"head.{n}", p) for n, p in head.named_parameters()]


def build_optimizer(named, config, lr):
    """AdamW with the training rule for weight decay (training.no_decay_1d)."""
    decay = config["training"].get("weight_decay", 1e-4)
    if not config["training"].get("no_decay_1d", False):
        return optim.AdamW([p for _, p in named], lr=lr, weight_decay=decay)
    return optim.AdamW([{"params": [p for _, p in named if p.ndim >= 2], "weight_decay": decay},
                        {"params": [p for _, p in named if p.ndim < 2], "weight_decay": 0.0}],
                       lr=lr)


def reconstruction(model, head, epi, seq_shape):
    """Predicted track means per chromatin token, (B, n_tracks, n_tokens), and
    SLIM's gate terms (None for the baseline). The DNA branch is off, so it
    reads zeros, as in fine-tuning."""
    seq = torch.zeros(epi.shape[0], *seq_shape, device=epi.device, dtype=epi.dtype)
    out = model.encode(seq, epi)
    tokens, aux = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
    return head(tokens[:, model.n_seq_tokens:, :]).transpose(1, 2), aux


def masked_mse(pred, target, hidden):
    err = (pred.float() - target) ** 2
    return (err * hidden).sum() / hidden.sum().clamp(min=1)


@torch.no_grad()
def monitor(model, head, loader, device, n_tracks, seq_shape, use_amp):
    """Mean squared error and R^2 on the hidden values of fixed windows,
    for the hidden spans and the hidden whole marks separately. R^2 compares
    against predicting each track's mean over the hidden values."""
    model.eval()
    head.eval()
    sums = {k: np.zeros(n_tracks) for k in ("sse", "n", "y", "yy")}
    for batch in loader:
        epi = batch["epi"].float().to(device, non_blocking=True)
        target = batch["target"].float().to(device)
        hidden = batch["hidden"].to(device)
        with autocast(device_type="cuda", enabled=use_amp):
            pred, _ = reconstruction(model, head, epi, seq_shape)
        err = (pred.float() - target) ** 2
        h = hidden.float()
        sums["sse"] += (err * h).sum(dim=(0, 2)).cpu().numpy()
        sums["n"] += h.sum(dim=(0, 2)).cpu().numpy()
        sums["y"] += (target * h).sum(dim=(0, 2)).cpu().numpy()
        sums["yy"] += (target ** 2 * h).sum(dim=(0, 2)).cpu().numpy()
    n = np.maximum(sums["n"], 1)
    sst = sums["yy"] - sums["y"] ** 2 / n
    return {"mse": float(sums["sse"].sum() / n.sum()),
            "r2": float(1 - sums["sse"].sum() / max(sst.sum(), 1e-12)),
            "r2_per_track": (1 - sums["sse"] / np.maximum(sst, 1e-12)).round(4).tolist()}


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.variant is not None:
        config["model"]["variant"] = args.variant
    variant = config["model"].get("variant", "KA")
    pre = config.get("pretrain", {})
    if config["data"].get("modalities", "all") != "epi":
        raise SystemExit("pretraining covers the chromatin branch only: "
                         "data.modalities must be epi")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda" and config["training"].get("use_amp", True)
               and not args.no_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Windows: the geometry of the training pairs, the chromatin of the corpus.
    paths = config["paths"]
    folders = paths.get("bengi_dirs") or [paths.get("bengi_dir", "./data/BENGI")]
    cell_args = SimpleNamespace(train_cells=list(DEFAULT_TRAIN_CELLS),
                                test_cells=list(DEFAULT_TEST_CELLS))
    labelled = training_cells(config, cell_args, folders)
    cells = corpus_cells(config, cell_args.test_cells)
    if not cells:
        raise SystemExit("no epigenome is left for pretraining")
    geometry = PairTable.from_files(bengi_files(folders, labelled))
    per_epoch = int(pre.get("windows_per_epoch", 200_000))
    epochs = int(pre.get("epochs", 4))
    n_monitor = int(pre.get("monitor_windows", 2000))
    print(f"Pretraining {variant}, seed {args.seed}: {epochs} epochs of {per_epoch:,} "
          f"windows, from the geometry of {len(geometry):,} training rows "
          f"({len(labelled)} cells) and the chromatin of {len(cells)} epigenomes:")
    print("  " + ", ".join(cells))
    table = window_table(geometry, cells, per_epoch * epochs + n_monitor, args.seed)
    windows = EPIGenomicDataset(bengi_paths=None, table=table, **dataset_options(config))

    n_tracks = len(config["data"].get("feats_order") or [])
    model = build_model(config).to(device)
    k = config["data"]["kmer_size"]
    seq_shape = (4 ** k, config["data"]["sequence_length"] - k + 1)
    pool = model.epi_pool_factor
    masking = dict(n_tracks=n_tracks, pool=pool,
                   fraction=float(pre.get("mask_fraction", 0.15)),
                   max_span=int(pre.get("max_span", 4)),
                   hide_mark=bool(pre.get("hide_mark", True)))
    train_windows = MaskedWindows(windows, **masking)
    monitor_windows = MaskedWindows(windows, fixed_seed=args.seed + 1, **masking)
    monitor_idx = list(range(per_epoch * epochs, per_epoch * epochs + n_monitor))

    head = nn.Linear(model.d_model, n_tracks).to(device)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(PRETRAINED_PREFIXES))
    named = trained_parameters(model, head)
    lr = float(pre.get("lr", 2e-4))
    optimizer = build_optimizer(named, config, lr)
    batch_size = config["data"]["batch_size"]
    steps = per_epoch // batch_size
    schedule = dict(config, training=dict(config["training"], epochs=epochs,
                                          scheduler="warmup_cosine"))
    scheduler, _ = build_scheduler(optimizer, schedule, steps)
    scaler = GradScaler("cuda", enabled=use_amp)
    train_cfg = config["training"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    print(f"Masking: {masking['fraction']:.0%} of tokens in spans of 1-"
          f"{masking['max_span']} tokens, one whole track "
          f"{'hidden' if masking['hide_mark'] else 'not hidden'}; "
          f"{sum(p.numel() for _, p in named):,} parameters trained, lr {lr}")

    workers = train_cfg.get("num_workers", 4)
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    if workers > 0:
        loader_kwargs.update(worker_init_fn=seed_worker,
                             prefetch_factor=train_cfg.get("prefetch_factor", 2))
    monitor_loader = DataLoader(Subset(monitor_windows, monitor_idx), **loader_kwargs)

    history = []
    out_path = os.path.join(args.output_dir, "pretrained.pt")
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        rows = list(range((epoch - 1) * per_epoch, epoch * per_epoch))
        loader = DataLoader(Subset(train_windows, rows), drop_last=True, **loader_kwargs)
        started, total, total_aux, n = time.time(), 0.0, 0.0, 0
        for batch in loader:
            epi = batch["epi"].float().to(device, non_blocking=True)
            target = batch["target"].float().to(device, non_blocking=True)
            hidden = batch["hidden"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                pred, aux = reconstruction(model, head, epi, seq_shape)
                loss = masked_mse(pred, target, hidden)
                extra = torch.zeros((), device=device)
                if aux is not None:
                    extra = model.memory_auxiliary_loss(
                        aux, entropy_weight=train_cfg.get("entropy_loss_weight", 0.01),
                        diversity_weight=train_cfg.get("diversity_loss_weight", 0.05),
                        prediction_weight=train_cfg.get("prediction_loss_weight", 0.0))
            scaler.scale(loss + extra).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for _, p in named], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
            total_aux += extra.item()
            n += 1
        scores = monitor(model, head, monitor_loader, device, n_tracks, seq_shape,
                         use_amp)
        record = {"epoch": epoch, "seconds": round(time.time() - started, 1),
                  "train_mse": total / max(n, 1), "train_memory_loss": total_aux / max(n, 1),
                  "lr": optimizer.param_groups[0]["lr"],
                  **{f"monitor_{k}": v for k, v in scores.items()}}
        history.append(record)
        print(f"Epoch {epoch}/{epochs}: train MSE {record['train_mse']:.4f}, monitor MSE "
              f"{scores['mse']:.4f}, R2 {scores['r2']:.3f} "
              f"(per track {scores['r2_per_track']}), {record['seconds']:.0f} s", flush=True)
        state = {k: v for k, v in model.state_dict().items()
                 if k.startswith(PRETRAINED_PREFIXES)}
        torch.save({"model": state, "head": head.state_dict(), "config": config,
                    "variant": variant, "seed": args.seed, "epoch": epoch,
                    "corpus_cells": cells, "history": history}, out_path)
        with open(os.path.join(args.output_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=1)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    run_main(main)

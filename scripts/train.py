"""
Train one SLIM variant, or the KAN-Transformer baseline.

One entry point covers every model, so the training loop, the optimizer, the
data pipeline and the evaluation protocol are identical across variants. The
variant name selects the model and nothing else.

Examples:

    python scripts/train.py --config configs/slim_ka.yaml --seed 0
    python scripts/train.py --config configs/baseline.yaml --seed 1
    python scripts/train.py --config configs/slim_ga.yaml --modalities seq
    python scripts/train.py --config configs/slim_ka.yaml --dry-run

Outputs land in `results/<variant>/seed<N>/` unless `--output-dir` says
otherwise, and always include the per-example validation and test predictions,
which every downstream analysis reads instead of re-running the model.

Training resumes automatically. After every epoch the full training state is
written to `last.pt` in the output directory: weights, optimizer, scheduler,
gradient scaler, weight average, epoch number, history and early-stopping
counters. If the run is interrupted, starting the same command again continues
from the epoch after the last completed one, so a power cut costs one epoch
rather than the whole run. Pass --no-resume to ignore an existing `last.pt` and
start over.

Which rows are trained on, and how, is set in the configuration (see
configs/base.yaml): `data.train_assays` and `training.valid_assays` choose the
BENGI assays for training and validation, `data.dedup` merges repeated pairs,
and `data.encoder_fit: crossfit` keeps the supervised sequence encoder from
seeing the labels of the examples it encodes. `--protocol` applies the
training assays and inputs chosen by the pilot runs (scripts/choose_protocol.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bengi import bengi_files, read_rows, select_rows_for_config
from src.config import load_config
from src.shutdown import run_main
from src.dataset import EPIDataset
from src.encoding import CrossFitEncoder, POCD_ND_Encoder, chromosome_half
from src.epi_data_pipeline import EPIGenomicDataset
from src.slim_model import build_model
from src.metrics import compute_all_metrics, format_epoch_line, format_metrics_report

DEFAULT_TRAIN_CELLS = ["GM12878", "HeLa", "K562", "IMR90"]
DEFAULT_TEST_CELLS = ["HMEC", "NHEK"]


def parse_args():
    p = argparse.ArgumentParser(description="Train a SLIM variant.")
    p.add_argument("--config", required=True, help="YAML configuration file")
    p.add_argument("--variant", default=None,
                   choices=["baseline", "A", "KA", "GA"],
                   help="overrides model.variant in the config")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds Python, NumPy and torch; also names the output "
                        "directory")
    p.add_argument("--deterministic", action="store_true",
                   help="force deterministic kernels; slower, and some "
                        "operations have no deterministic implementation")
    p.add_argument("--modalities", default=None,
                   choices=["all", "seq", "seq+pos", "epi"],
                   help="which inputs the model may see: seq and seq+pos "
                        "remove chromatin tracks, epi removes the DNA branch")
    p.add_argument("--train-assays", nargs="+", default=None,
                   help="BENGI assays to train on, e.g. HiC; 'all' keeps every "
                        "assay. Overrides data.train_assays and --protocol")
    p.add_argument("--protocol", default=None,
                   help="JSON written by scripts/choose_protocol.py; applies "
                        "its training assays and inputs before the other "
                        "command-line overrides")
    p.add_argument("--split", default="cross-cell",
                   choices=["cross-cell", "loco"],
                   help="cross-cell holds out whole cell lines; loco holds out "
                        "one chromosome from every cell line")
    p.add_argument("--loco-chrom", default="chr1",
                   help="held-out chromosome when --split loco")
    p.add_argument("--train-cells", nargs="+", default=DEFAULT_TRAIN_CELLS)
    p.add_argument("--test-cells", nargs="+", default=DEFAULT_TEST_CELLS)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any saved training state and start from epoch 1")
    p.add_argument("--dry-run", action="store_true",
                   help="build the config and the model, then stop; needs no "
                        "data and no GPU")
    return p.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every generator the run touches.

    Data loading order, weight initialisation, dropout and augmentation all
    draw from these generators, so a seeded run is reproducible on the same
    hardware and library versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible stream."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def held_out_chroms(args, config):
    """Chromosomes held out for validation under the chosen split."""
    if args.split == "loco":
        return {args.loco_chrom}
    return set(config["training"].get("valid_chroms", ["chr11", "chr17"]))


def apply_protocol(config, path):
    """Apply the training assays and inputs chosen by the pilot runs."""
    with open(path) as handle:
        protocol = json.load(handle)
    config["data"]["train_assays"] = protocol.get("train_assays")
    config["data"]["modalities"] = protocol.get("modalities", "all")
    config["protocol"] = {"file": str(path),
                          "train_assays": protocol.get("train_assays"),
                          "modalities": protocol.get("modalities", "all"),
                          "chosen": protocol.get("chosen")}
    return config


def fit_sequence_encoder(train_genomic, train_idx, config, seed):
    """Fit the POCD-ND encoder on training sequences.

    data.encoder_fit "train" (the original rule): one encoder fitted on up to
    `encoder_fit_samples` positives and as many negatives drawn from the
    training rows, which are then trained on. Those rows are encoded with
    densities that counted their own labels.

    data.encoder_fit "crossfit": one encoder per chromosome half (odd and even
    numbers), each fitted on the training rows of its own half and used to
    encode the rows of the OTHER half, so no row is encoded with statistics
    that include its own label. Validation and test rows follow the same rule.
    """
    k = config["data"]["kmer_size"]
    seq_len = config["data"]["sequence_length"]
    max_fit = config["data"].get("encoder_fit_samples", 5000)
    mode = config["data"].get("encoder_fit", "train")
    labels_all = train_genomic.get_labels()
    rng = np.random.default_rng(seed)

    fitted = []

    def draw(pool):
        pool = list(pool)
        pos = [i for i in pool if labels_all[i] == 1]
        neg = [i for i in pool if labels_all[i] == 0]
        pos = rng.choice(pos, size=min(max_fit, len(pos)), replace=False) if pos else []
        neg = rng.choice(neg, size=min(max_fit, len(neg)), replace=False) if neg else []
        fitted.extend(int(i) for i in list(pos) + list(neg))
        join = lambda i: "".join(train_genomic.sequence_pair(int(i)))
        return [join(i) for i in pos], [join(i) for i in neg]

    if mode == "train":
        pos_seqs, neg_seqs = draw(train_idx)
        encoder = POCD_ND_Encoder(k=k)
        encoder.fit(pos_seqs, neg_seqs, seq_len)
        print(f"  Encoder fitted on {len(pos_seqs)} positive and "
              f"{len(neg_seqs)} negative training sequences (these rows are "
              f"then trained on).")
        encoder.fit_rows = sorted(fitted)
        return encoder
    if mode != "crossfit":
        raise ValueError("data.encoder_fit must be train or crossfit")
    chroms = train_genomic.get_chrom_groups()
    pos_by_half, neg_by_half = [], []
    for half in (0, 1):
        pool = [i for i in train_idx if chromosome_half(chroms[i]) == half]
        pos_seqs, neg_seqs = draw(pool)
        pos_by_half.append(pos_seqs)
        neg_by_half.append(neg_seqs)
        print(f"  Chromosome half {half}: fitted on {len(pos_seqs)} positive "
              f"and {len(neg_seqs)} negative sequences; encodes the other half.")
    encoder = CrossFitEncoder(k=k)
    encoder.fit(pos_by_half, neg_by_half, seq_len)
    encoder.fit_rows = sorted(fitted)
    return encoder


def forward_model(model, seq, epi, enh_idx, prom_idx):
    """Call the model and normalise its outputs across variants.

    The baseline returns three values; the survival-gated variants also return
    the gate regularisation terms.
    """
    out = model(seq, epi, enh_idx, prom_idx)
    if len(out) == 4:
        cls_out, reg_out, attention, aux = out
        return cls_out, reg_out, attention, aux
    cls_out, reg_out, attention = out
    return cls_out, reg_out, attention, None


def compute_loss(model, batch_out, labels, distances, config):
    """Total training loss, shared by every variant."""
    cls_out, reg_out, attention, aux = batch_out
    train_cfg = config["training"]
    loss = nn.functional.binary_cross_entropy_with_logits(cls_out, labels)
    loss = loss + train_cfg["lambda_dist"] * nn.functional.mse_loss(
        reg_out, distances)
    loss = loss + config["model"].get("att_penalty", 0.1) * \
        model.attention_penalty(attention)
    if aux is not None:
        loss = loss + model.memory_auxiliary_loss(
            aux,
            entropy_weight=train_cfg.get("entropy_loss_weight", 0.01),
            diversity_weight=train_cfg.get("diversity_loss_weight", 0.05),
            prediction_weight=train_cfg.get("prediction_loss_weight", 0.0),
        )
    return loss


@torch.no_grad()
def evaluate_loader(model, loader, device, config, use_amp=False):
    """Run the model over a loader and return predictions with metrics."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    preds, labels = [], []
    for batch in loader:
        seq = batch["seq"].float().to(device, non_blocking=True)
        epi = batch["epi"].float().to(device, non_blocking=True)
        lbl = batch["label"].float().to(device, non_blocking=True)
        dst = batch["dist"].float().to(device, non_blocking=True)
        enh_idx = batch["enh_idx"].float().to(device, non_blocking=True)
        prom_idx = batch["prom_idx"].float().to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            out = forward_model(model, seq, epi, enh_idx, prom_idx)
            loss = compute_loss(model, out, lbl, dst, config)

        total_loss += float(loss)
        n_batches += 1
        preds.append(torch.sigmoid(out[0].float()).cpu().numpy().ravel())
        labels.append(lbl.cpu().numpy().ravel())

    preds = np.concatenate(preds) if preds else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0)
    metrics = compute_all_metrics(labels, preds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["predictions"] = preds
    metrics["labels"] = labels
    return metrics


def build_optimizer(model, config):
    """AdamW, optionally sparing one-dimensional parameters from weight decay.

    training.no_decay_1d true exempts biases and normalisation scales, the
    usual practice; false (the original rule) decays every parameter.
    """
    train_cfg = config["training"]
    decay = train_cfg.get("weight_decay", 1e-4)
    if not train_cfg.get("no_decay_1d", False):
        return optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=decay)
    with_decay, without = [], []
    for param in model.parameters():
        if param.requires_grad:
            (with_decay if param.ndim >= 2 else without).append(param)
    return optim.AdamW([{"params": with_decay, "weight_decay": decay},
                        {"params": without, "weight_decay": 0.0}],
                       lr=train_cfg["lr"])


def build_scheduler(optimizer, config, steps_per_epoch):
    """Learning-rate schedule named by training.scheduler.

    "plateau" (the original default) halves the rate after five epochs without
    improvement, which never happens inside a short budget. "cosine_restarts"
    is the original alternative. "warmup_cosine" warms up linearly over
    `warmup_fraction` of all steps, then follows a cosine down to
    `min_lr_ratio` of the peak at the end of the epoch budget; it is stepped
    after every batch. Returns the scheduler and whether it steps per batch.
    """
    train_cfg = config["training"]
    name = train_cfg.get("scheduler")
    if name is None:
        name = "cosine_restarts" if train_cfg.get("use_cosine_scheduler") else "plateau"
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5), False
    if name == "cosine_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2), False
    if name != "warmup_cosine":
        raise ValueError("training.scheduler must be plateau, cosine_restarts "
                         "or warmup_cosine")
    total = max(1, steps_per_epoch * train_cfg["epochs"])
    warmup = max(1, int(train_cfg.get("warmup_fraction", 0.03) * total))
    floor = train_cfg.get("min_lr_ratio", 0.05)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, factor), True


def build_weight_average(model, config):
    """Exponential moving average of the weights, or None when switched off.

    With training.ema_decay above zero, validation, checkpoint selection and
    the final test all use the averaged weights, which change more smoothly
    than the raw ones and so depend less on where an epoch happens to end.
    Buffers (the batch-norm statistics) are averaged too.
    """
    decay = config["training"].get("ema_decay", 0.0)
    if not decay:
        return None
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay),
                         use_buffers=True)


def build_splits(train_genomic, args, config):
    """Return train and validation indices for the chosen protocol.

    cross-cell: whole cell lines are held out for test, and inside the
    training cell lines the validation chromosomes are held out.

    loco: one chromosome is held out from every cell line, which is the
    leave-one-chromosome-out protocol used in the chromosome-aware literature.
    """
    chroms = train_genomic.get_chrom_groups()
    held = held_out_chroms(args, config)
    train_idx = [i for i, c in enumerate(chroms) if c not in held]
    val_idx = [i for i, c in enumerate(chroms) if c in held]
    if not val_idx:
        raise ValueError(
            f"no samples on the held-out chromosomes {sorted(held)}; check "
            f"valid_chroms in the config or --loco-chrom")
    return train_idx, val_idx, sorted(held)


def save_predictions(save_dir, test_metrics, val_metrics, test_genomic,
                     per_cell, extra):
    """Save every per-example prediction the downstream analyses need.

    The coordinates travel with the predictions so that the leakage audit,
    the threshold selection and the calibration analysis never have to
    reconstruct the ordering of the test set.
    """
    samples = test_genomic.samples
    payload = dict(
        test_predictions=test_metrics["predictions"],
        test_labels=test_metrics["labels"],
        test_chrom=np.array([s["chrom"] for s in samples]),
        test_cell=np.array([s["cell"] for s in samples]),
        test_enh_coord=np.array([s["enh_coord"] for s in samples], dtype=np.int64),
        test_prom_coord=np.array([s["prom_coord"] for s in samples], dtype=np.int64),
        val_predictions=val_metrics["predictions"],
        val_labels=val_metrics["labels"],
    )
    for key in ("auroc", "aupr", "f1", "mcc", "balanced_accuracy",
                "accuracy", "precision", "recall"):
        payload[f"test_{key}"] = np.float64(test_metrics.get(key) or np.nan)
        payload[f"val_{key}"] = np.float64(val_metrics.get(key) or np.nan)
    for cell, m in per_cell.items():
        payload[f"{cell}_predictions"] = m["predictions"]
        payload[f"{cell}_labels"] = m["labels"]
        payload[f"{cell}_auroc"] = np.float64(m.get("auroc") or np.nan)
        payload[f"{cell}_aupr"] = np.float64(m.get("aupr") or np.nan)
    payload.update(extra)
    np.savez(os.path.join(save_dir, "eval_results.npz"), **payload)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.protocol is not None:
        config = apply_protocol(config, args.protocol)
    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.modalities is not None:
        config["data"]["modalities"] = args.modalities
    if args.train_assays is not None:
        config["data"]["train_assays"] = (
            None if args.train_assays == ["all"] else args.train_assays)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    variant = config["model"].get("variant", "KA")
    memory_cfg = config["model"].get("memory", {})
    if (memory_cfg.get("prediction_target") == "position"
            and config["training"].get("prediction_loss_weight", 0.0) <= 0):
        raise ValueError(
            "model.memory.prediction_target is 'position' but "
            "training.prediction_loss_weight is not positive, so nothing would "
            "train the predictor and its error would be noise.")
    # `output_name` lets a configuration that shares a variant with another,
    # such as the legacy KA run, keep its results in its own directory.
    save_dir = args.output_dir or os.path.join(
        "results", str(config.get("output_name", variant)).lower(),
        f"seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", True) and not args.no_amp
    if device.type != "cuda":
        use_amp = False
    if device.type == "cuda" and not args.deterministic:
        # Input shapes are fixed, so autotuning pays for itself, and TF32
        # speeds up the convolution and recurrent paths on this GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print(f"  SLIM training: variant {variant}, seed {args.seed}")
    print("=" * 70)
    print(f"  Device:      {device}")
    if device.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mixed prec:  {'on' if use_amp else 'off'}")
    print(f"  Split:       {args.split}")
    print(f"  Modalities:  {config['data'].get('modalities', 'all')}")
    print(f"  Output:      {save_dir}")

    if args.dry_run:
        model = build_model(config)
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  Model built: {total:,} parameters")
        print("  Dry run complete; no data was read.")
        return

    paths = config["paths"]
    bengi_dir = paths.get("bengi_dir", "./data/BENGI")
    feats_config = paths.get("feats_config", "")
    ref_genome = paths.get("ref_genome", "") or None

    train_files = bengi_files(bengi_dir, args.train_cells)
    test_files = bengi_files(bengi_dir, args.test_cells)
    if not train_files:
        raise FileNotFoundError(
            f"no BENGI files for training cells {args.train_cells} in {bengi_dir}")
    if not test_files:
        raise FileNotFoundError(
            f"no BENGI files for test cells {args.test_cells} in {bengi_dir}")

    ds_kwargs = dict(
        feats_config_path=feats_config,
        feats_order=config["data"].get("feats_order"),
        seq_len=config["data"].get("seq_len_bp", 2_500_000),
        bin_size=config["data"].get("bin_size", 500),
        enhancer_window=config["data"].get("enhancer_window", 3000),
        promoter_window=config["data"].get("promoter_window", 3000),
        ref_genome_path=ref_genome,
    )
    print("\n=== Loading training data ===")
    held = held_out_chroms(args, config)
    train_rows, row_report = select_rows_for_config(
        read_rows(train_files), held, config)
    print(f"  Training assays {config['data'].get('train_assays') or 'all'}, "
          f"validation assays {config['training'].get('valid_assays') or 'all'}, "
          f"dedup {config['data'].get('dedup', 'none')}: "
          f"{row_report['rows_in']:,} rows kept, "
          f"{row_report['duplicate_rows']:,} repeat a pair "
          f"({row_report['pairs_with_conflicting_labels']:,} pairs with "
          f"conflicting labels), {row_report['rows_out']:,} rows used.")
    train_genomic = EPIGenomicDataset(bengi_paths=None, rows=train_rows,
                                      **ds_kwargs)
    print("\n=== Loading test data ===")
    # The test set is the benchmark as published: every row, in file order.
    test_genomic = EPIGenomicDataset(bengi_paths=test_files, **ds_kwargs)

    train_idx, val_idx, held = build_splits(train_genomic, args, config)
    print(f"\nHeld-out chromosomes for validation: {held}")
    labels_all = train_genomic.get_labels()
    train_labels = np.array([labels_all[i] for i in train_idx])

    print(f"\nFitting the POCD-ND encoder "
          f"({config['data'].get('encoder_fit', 'train')})...")
    encoder = fit_sequence_encoder(train_genomic, train_idx, config, args.seed)
    with open(os.path.join(save_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    train_dataset = EPIDataset(config, encoder, source_dataset=train_genomic)
    test_dataset = EPIDataset(config, encoder, source_dataset=test_genomic)
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(train_dataset, val_idx)

    n_pos = int(train_labels.sum())
    print(f"\nTrain {len(train_set)} | Val {len(val_set)} | "
          f"Test {len(test_dataset)}")
    print(f"Training positives: {n_pos} "
          f"({100 * n_pos / max(len(train_labels), 1):.1f} percent)")

    num_workers = config["training"].get("num_workers", 4)
    batch_size = config["data"]["batch_size"]
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config["training"].get(
            "prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = config["training"].get(
            "persistent_workers", False)
        loader_kwargs["worker_init_fn"] = seed_worker
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    eval_kwargs = dict(loader_kwargs)
    eval_kwargs["num_workers"] = min(4, num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=generator,
                              **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, **eval_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **eval_kwargs)

    # A fixed sample of training rows, scored like validation (evaluation
    # mode, no augmentation, the weights that are validated) after every
    # epoch. The running training metrics are measured while the weights move
    # and with dropout and augmentation on, so they cannot give a clean
    # training-minus-validation gap; these can.
    n_clean = min(config["training"].get("train_eval_samples", 0), len(train_idx))
    clean_loader = None
    if n_clean > 0:
        pick = np.random.default_rng(args.seed + 1).choice(
            len(train_idx), size=n_clean, replace=False)
        clean_set = Subset(train_dataset, [train_idx[i] for i in sorted(pick)])
        clean_loader = DataLoader(clean_set, batch_size=batch_size, **eval_kwargs)

    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel {variant}: {total_params:,} parameters")

    optimizer = build_optimizer(model, config)
    scheduler, step_per_batch = build_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler("cuda", enabled=use_amp)
    average = build_weight_average(model, config)
    # The weights that are validated, selected and tested.
    scored = lambda: average.module if average is not None else model

    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 15)
    grad_clip = config["training"].get("grad_clip", 1.0)
    use_augment = config.get("augmentation", {}).get("enabled", False)

    best_metric = -float("inf")
    patience_left = patience
    history = []
    start_epoch = 1
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")
    state_path = os.path.join(save_dir, "last.pt")

    # Resume from the last completed epoch if a previous run was interrupted.
    if os.path.exists(state_path) and not args.no_resume:
        saved = torch.load(state_path, map_location=device, weights_only=False)
        saved_model_cfg = saved.get("config", {}).get("model")
        if saved.get("seed") != args.seed or saved.get("variant") != variant:
            print(f"\nFound {state_path} but it belongs to a different run "
                  f"(variant {saved.get('variant')}, seed {saved.get('seed')}). "
                  f"Starting from scratch.")
        elif saved_model_cfg != config["model"]:
            # Same variant and seed, but built with different model settings,
            # for example the memory rules before a code change. Its weights
            # would not describe the model being trained now.
            print(f"\nFound {state_path} but it was trained with different "
                  f"model settings. Starting from scratch.")
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            if average is not None:
                if saved.get("average") is None:
                    raise SystemExit(
                        f"{state_path} has no weight average but this run "
                        f"uses one; rerun with --no-resume.")
                average.load_state_dict(saved["average"])
            history = saved["history"]
            best_metric = saved["best_metric"]
            patience_left = saved["patience_left"]
            start_epoch = saved["epoch"] + 1
            print(f"\nResuming from {state_path}: epoch {saved['epoch']} is "
                  f"complete, continuing at epoch {start_epoch}.")
            if start_epoch > epochs:
                print("All epochs were already completed. Going straight to "
                      "the final evaluation.")

    print(f"\nTraining epochs {start_epoch} to {epochs}, patience {patience}, "
          f"augmentation {'on' if use_augment else 'off'}")
    print(f"Optimiser: AdamW, weight decay "
          f"{config['training'].get('weight_decay', 1e-4)}, schedule "
          f"{type(scheduler).__name__}, weight average "
          f"{config['training'].get('ema_decay', 0.0) or 'off'}")
    print("Model selection: validation AUROC plus AUPR"
          + (" of the averaged weights" if average is not None else ""))
    print("State is saved after every epoch, so an interrupted run resumes "
          "here.\n")

    stopped_early = False
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        model.train()
        train_dataset.augment = use_augment
        epoch_preds, epoch_labels = [], []
        running = 0.0

        for batch in train_loader:
            seq = batch["seq"].float().to(device, non_blocking=True)
            epi = batch["epi"].float().to(device, non_blocking=True)
            lbl = batch["label"].float().to(device, non_blocking=True)
            dst = batch["dist"].float().to(device, non_blocking=True)
            enh_idx = batch["enh_idx"].float().to(device, non_blocking=True)
            prom_idx = batch["prom_idx"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                out = forward_model(model, seq, epi, enh_idx, prom_idx)
                loss = compute_loss(model, out, lbl, dst, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if step_per_batch:
                scheduler.step()
            if average is not None:
                average.update_parameters(model)

            running += float(loss)
            with torch.no_grad():
                epoch_preds.append(
                    torch.sigmoid(out[0].float()).cpu().numpy().ravel())
                epoch_labels.append(lbl.cpu().numpy().ravel())

        train_metrics = compute_all_metrics(
            np.concatenate(epoch_labels), np.concatenate(epoch_preds))
        train_metrics["loss"] = running / max(len(train_loader), 1)

        train_dataset.augment = False
        val_metrics = evaluate_loader(scored(), val_loader, device, config, use_amp)
        clean_metrics = (evaluate_loader(scored(), clean_loader, device, config,
                                         use_amp)
                         if clean_loader is not None else None)

        elapsed = time.time() - started
        print(format_epoch_line(epoch, epochs, train_metrics, val_metrics, elapsed))

        selection = (val_metrics.get("auroc") or 0.0) + (val_metrics.get("aupr") or 0.0)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(selection)
        elif not step_per_batch:
            scheduler.step(epoch)

        row = {"epoch": epoch, "seconds": round(elapsed, 1),
               "lr": optimizer.param_groups[0]["lr"]}
        scored_sets = [("train", train_metrics), ("val", val_metrics)]
        if clean_metrics is not None:
            scored_sets.append(("train_clean", clean_metrics))
        for prefix, m in scored_sets:
            for key in ("loss", "auroc", "aupr", "accuracy",
                        "balanced_accuracy", "precision", "recall", "f1", "mcc"):
                row[f"{prefix}_{key}"] = m.get(key)
        history.append(row)
        if clean_metrics is not None:
            print(f"  clean training sample: AUROC "
                  f"{clean_metrics.get('auroc') or 0:.4f}, AUPR "
                  f"{clean_metrics.get('aupr') or 0:.4f} "
                  f"(validation {val_metrics.get('aupr') or 0:.4f})")

        if selection > best_metric:
            best_metric = selection
            patience_left = patience
            torch.save({"model": scored().state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "seed": args.seed,
                        "val_selection": selection,
                        "weights": "average" if average is not None else "raw"},
                       ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                stopped_early = True

        # Full training state, written after every epoch so an interrupted run
        # resumes from here. Written to a temporary file first, then moved, so
        # a power cut during the write cannot leave a truncated file behind.
        temporary = state_path + ".tmp"
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "average": (average.state_dict()
                                if average is not None else None),
                    "history": history,
                    "best_metric": best_metric,
                    "patience_left": patience_left,
                    "epoch": epoch,
                    "seed": args.seed,
                    "variant": variant,
                    "config": config}, temporary)
        os.replace(temporary, state_path)

        # Write the history after every epoch too, so the overfitting check can
        # run against a run that has not finished.
        with open(os.path.join(save_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)

        if stopped_early:
            break

    # Final evaluation uses the selected checkpoint, never the last epoch.
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"No checkpoint was written to {ckpt_path}, so no epoch improved "
            f"on the initial validation score. Check the training log above.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"\nLoaded the checkpoint selected at epoch {state['epoch']}.")

    val_metrics = evaluate_loader(model, val_loader, device, config, use_amp)
    test_metrics = evaluate_loader(model, test_loader, device, config, use_amp)

    print(f"\n[Validation] {len(val_metrics['labels'])} pairs, chroms {held}")
    print(format_metrics_report(val_metrics, indent=2))
    print(f"\n[Test] {len(test_metrics['labels'])} pairs from "
          f"{args.test_cells}, threshold 0.5")
    print(format_metrics_report(test_metrics, indent=2))

    cells = np.array([s["cell"] for s in test_genomic.samples])
    per_cell = {}
    for cell in sorted(set(cells.tolist())):
        sel = cells == cell
        m = compute_all_metrics(test_metrics["labels"][sel],
                                test_metrics["predictions"][sel])
        m["predictions"] = test_metrics["predictions"][sel]
        m["labels"] = test_metrics["labels"][sel]
        per_cell[cell] = m
        print(f"\n[Test {cell}] {int(sel.sum())} pairs")
        print(format_metrics_report(m, indent=2))

    save_predictions(
        save_dir, test_metrics, val_metrics, test_genomic, per_cell,
        extra=dict(
            variant=np.array(str(variant)),
            seed=np.int64(args.seed),
            split=np.array(args.split),
            modalities=np.array(config["data"].get("modalities", "all")),
            train_cells=np.array(args.train_cells),
            test_cells=np.array(args.test_cells),
            train_assays=np.array(config["data"].get("train_assays") or ["all"]),
            valid_assays=np.array(config["training"].get("valid_assays") or ["all"]),
            dedup=np.array(config["data"].get("dedup", "none")),
            encoder_fit=np.array(config["data"].get("encoder_fit", "train")),
            train_rows=np.int64(len(train_idx)),
            val_rows=np.int64(len(val_idx)),
            valid_chroms=np.array(held),
            total_params=np.int64(total_params),
            epochs_run=np.int64(len(history)),
        ),
    )
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    snapshot = dict(config)
    snapshot["_run"] = {"seed": args.seed, "split": args.split,
                        "train_cells": args.train_cells,
                        "test_cells": args.test_cells,
                        "valid_chroms": held,
                        "total_params": total_params,
                        "train_rows": len(train_idx),
                        "val_rows": len(val_idx),
                        "row_report": row_report}
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)

    # The run finished, so the resume state is no longer needed. Removing it
    # means re-running this command starts a fresh run rather than skipping
    # straight to evaluation.
    if os.path.exists(state_path):
        os.remove(state_path)

    print(f"\nSaved predictions, history and configuration to {save_dir}")


if __name__ == "__main__":
    # Exits without the Windows shutdown crash that follows GPU training;
    # see src/shutdown.py.
    run_main(main)

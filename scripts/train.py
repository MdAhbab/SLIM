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
gradient scaler, epoch number, history and early-stopping counters. If the run
is interrupted, starting the same command again continues from the epoch after
the last completed one, so a power cut costs one epoch rather than the whole
run. Pass --no-resume to ignore an existing `last.pt` and start over.
"""

from __future__ import annotations

import argparse
import glob
import json
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import EPIDataset
from src.encoding import POCD_ND_Encoder
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
                   choices=["all", "seq", "seq+pos"],
                   help="which inputs the chromatin branch may see")
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


def filter_bengi_files(bengi_dir, cell_names):
    """Collect the BENGI benchmark files belonging to the named cell lines."""
    files = sorted(glob.glob(os.path.join(bengi_dir, "*.tsv*")))
    keep = []
    for path in files:
        base = os.path.basename(path)
        cell = base.split(".")[0]
        if cell in cell_names:
            keep.append(path)
    return keep


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


def build_splits(train_genomic, args, config):
    """Return train and validation indices for the chosen protocol.

    cross-cell: whole cell lines are held out for test, and inside the
    training cell lines the validation chromosomes are held out.

    loco: one chromosome is held out from every cell line, which is the
    leave-one-chromosome-out protocol used in the chromosome-aware literature.
    """
    chroms = train_genomic.get_chrom_groups()
    if args.split == "loco":
        held = {args.loco_chrom}
    else:
        held = set(config["training"].get("valid_chroms", ["chr11", "chr17"]))
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

    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.modalities is not None:
        config["data"]["modalities"] = args.modalities
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    variant = config["model"].get("variant", "KA")
    save_dir = args.output_dir or os.path.join(
        "results", str(variant).lower(), f"seed{args.seed}")
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

    train_files = filter_bengi_files(bengi_dir, args.train_cells)
    test_files = filter_bengi_files(bengi_dir, args.test_cells)
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
    train_genomic = EPIGenomicDataset(bengi_paths=train_files, **ds_kwargs)
    print("\n=== Loading test data ===")
    test_genomic = EPIGenomicDataset(bengi_paths=test_files, **ds_kwargs)

    train_idx, val_idx, held = build_splits(train_genomic, args, config)
    print(f"\nHeld-out chromosomes for validation: {held}")

    # Fit the position-aware encoder on training sequences only.
    print("\nFitting the POCD-ND encoder on training sequences...")
    encoder = POCD_ND_Encoder(k=config["data"]["kmer_size"])
    labels_all = train_genomic.get_labels()
    train_labels = np.array([labels_all[i] for i in train_idx])
    pos_pool = [train_idx[j] for j in np.where(train_labels == 1)[0]]
    neg_pool = [train_idx[j] for j in np.where(train_labels == 0)[0]]
    max_fit = config["data"].get("encoder_fit_samples", 5000)
    rng = np.random.default_rng(args.seed)
    pos_sample = rng.choice(pos_pool, size=min(max_fit, len(pos_pool)),
                            replace=False)
    neg_sample = rng.choice(neg_pool, size=min(max_fit, len(neg_pool)),
                            replace=False)
    pos_seqs = [train_genomic[i]["enhancer_seq"] + train_genomic[i]["promoter_seq"]
                for i in pos_sample]
    neg_seqs = [train_genomic[i]["enhancer_seq"] + train_genomic[i]["promoter_seq"]
                for i in neg_sample]
    encoder.fit(pos_seqs, neg_seqs, config["data"]["sequence_length"])
    with open(os.path.join(save_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    print(f"  Encoder fitted on {len(pos_seqs)} positive and "
          f"{len(neg_seqs)} negative sequences.")

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

    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel {variant}: {total_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                            weight_decay=config["training"].get("weight_decay", 1e-4))
    if config["training"].get("use_cosine_scheduler", False):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)
    scaler = GradScaler("cuda", enabled=use_amp)

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
        if saved.get("seed") != args.seed or saved.get("variant") != variant:
            print(f"\nFound {state_path} but it belongs to a different run "
                  f"(variant {saved.get('variant')}, seed {saved.get('seed')}). "
                  f"Starting from scratch.")
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
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
    print("Model selection: validation AUROC plus AUPR")
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

            running += float(loss)
            with torch.no_grad():
                epoch_preds.append(
                    torch.sigmoid(out[0].float()).cpu().numpy().ravel())
                epoch_labels.append(lbl.cpu().numpy().ravel())

        train_metrics = compute_all_metrics(
            np.concatenate(epoch_labels), np.concatenate(epoch_preds))
        train_metrics["loss"] = running / max(len(train_loader), 1)

        train_dataset.augment = False
        val_metrics = evaluate_loader(model, val_loader, device, config, use_amp)

        elapsed = time.time() - started
        print(format_epoch_line(epoch, epochs, train_metrics, val_metrics, elapsed))

        selection = (val_metrics.get("auroc") or 0.0) + (val_metrics.get("aupr") or 0.0)
        if config["training"].get("use_cosine_scheduler", False):
            scheduler.step(epoch)
        else:
            scheduler.step(selection)

        row = {"epoch": epoch, "seconds": round(elapsed, 1),
               "lr": optimizer.param_groups[0]["lr"]}
        for prefix, m in (("train", train_metrics), ("val", val_metrics)):
            for key in ("loss", "auroc", "aupr", "accuracy",
                        "balanced_accuracy", "precision", "recall", "f1", "mcc"):
                row[f"{prefix}_{key}"] = m.get(key)
        history.append(row)

        if selection > best_metric:
            best_metric = selection
            patience_left = patience
            torch.save({"model": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "seed": args.seed,
                        "val_selection": selection}, ckpt_path)
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
                        "total_params": total_params}
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)

    # The run finished, so the resume state is no longer needed. Removing it
    # means re-running this command starts a fresh run rather than skipping
    # straight to evaluation.
    if os.path.exists(state_path):
        os.remove(state_path)

    print(f"\nSaved predictions, history and configuration to {save_dir}")


if __name__ == "__main__":
    main()

"""
Diagnose overfitting and transfer loss from a finished or in-progress run.

Two different gaps matter here and they mean different things. Confusing them
leads to the wrong fix, so this script reports them separately.

  Overfitting gap = training score minus validation score.
      Both come from the same four cell lines; validation is held-out
      chromosomes. A large gap means the model is memorising the training
      pairs. The fix is regularisation, augmentation, or fewer epochs.

  Transfer gap = validation score minus test score.
      Validation and test differ by cell line, not by chromosome. A large gap
      means the model learned something specific to the training cell types.
      Training for fewer epochs does not fix this. It is the problem this
      project is about, and a smaller transfer gap is the real claim.

A model can have a small overfitting gap and a large transfer gap, which is
the usual situation in cross-cell-line prediction.

Where the training score comes from matters. Runs made with
`training.train_eval_samples` record `train_clean_*`: a fixed training sample
scored after each epoch exactly like validation (evaluation mode, no
augmentation, the same weights). That is the training score used here when
it exists. Older runs only have the running score accumulated during the
epoch, with dropout and augmentation on and the weights still moving, which
is not a clean comparison with validation; the report says which one it used.

Reads `history.json` and `eval_results.npz` from each run directory, so it
works on a run that is still in progress.

Usage:

    python scripts/check_overfitting.py --run results/ka/seed0
    python scripts/check_overfitting.py --run results/*/seed0 --out results/overfitting.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Judged on AUPR, because the benchmark is imbalanced and AUPR is the metric
# the manuscript treats as decisive.
PRIMARY = "aupr"

# Thresholds for the printed verdict. They are conventions for reading the
# numbers, not results, and the numbers themselves are always printed.
GAP_MILD = 0.10
GAP_CLEAR = 0.20


def parse_args():
    p = argparse.ArgumentParser(
        description="Report overfitting and transfer gaps for training runs.")
    p.add_argument("--run", nargs="+", required=True,
                   help="run directories, or a glob such as results/*/seed0")
    p.add_argument("--metric", default=PRIMARY,
                   choices=["aupr", "auroc", "f1", "mcc"])
    p.add_argument("--out", default=None, help="write the report as JSON")
    return p.parse_args()


def load_history(run_dir):
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def load_final(run_dir):
    path = os.path.join(run_dir, "eval_results.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in ("test_aupr", "test_auroc", "test_f1", "test_mcc",
                "val_aupr", "val_auroc", "val_f1", "val_mcc"):
        if key in data.files:
            value = float(data[key])
            out[key] = None if np.isnan(value) else value
    return out


def describe(run_dir, metric):
    history = load_history(run_dir)
    final = load_final(run_dir)
    report = {"dir": run_dir, "metric": metric}

    print("=" * 74)
    print(f"  {run_dir}")
    print("=" * 74)

    if not history:
        print("  No history.json. Train this run first.")
        return None

    clean = all(row.get(f"train_clean_{metric}") is not None for row in history)
    source = "train_clean" if clean else "train"
    train_key, val_key = f"{source}_{metric}", f"val_{metric}"
    report["training_score"] = (
        "clean training sample, scored like validation" if clean
        else "running score during the epoch (dropout and augmentation on)")
    print(f"  Training score: {report['training_score']}")
    epochs = [row["epoch"] for row in history]
    train = [row.get(train_key) for row in history]
    val = [row.get(val_key) for row in history]
    train_loss = [row.get(f"{source}_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]

    if any(v is None for v in train) or any(v is None for v in val):
        print(f"  history.json has no {metric} columns.")
        return None

    print(f"\n  Per-epoch {metric.upper()} and loss")
    print(f"  {'epoch':>6}{'train':>9}{'val':>9}{'gap':>9}"
          f"{'tr loss':>10}{'val loss':>10}")
    for i, epoch in enumerate(epochs):
        gap = train[i] - val[i]
        mark = ""
        if val[i] == max(val):
            mark = "  <- best validation"
        print(f"  {epoch:>6}{train[i]:>9.4f}{val[i]:>9.4f}{gap:>9.4f}"
              f"{train_loss[i]:>10.4f}{val_loss[i]:>10.4f}{mark}")

    best_index = int(np.argmax(val))
    best_epoch = epochs[best_index]
    # Training keeps the epoch with the highest validation AUROC plus AUPR,
    # which need not be the epoch with the highest AUPR alone.
    selection = [(row.get("val_auroc") or 0.0) + (row.get("val_aupr") or 0.0)
                 for row in history]
    saved_epoch = epochs[int(np.argmax(selection))]
    last_index = len(epochs) - 1

    overfitting_gap = train[best_index] - val[best_index]
    final_gap = train[last_index] - val[last_index]
    decline = val[best_index] - val[last_index]

    report.update({
        "epochs_run": len(epochs),
        "best_validation_epoch": best_epoch,
        "checkpoint_epoch": saved_epoch,
        "best_validation_score": val[best_index],
        "train_at_best": train[best_index],
        "overfitting_gap_at_best": overfitting_gap,
        "overfitting_gap_at_last": final_gap,
        "validation_decline_after_peak": decline,
    })

    print(f"\n  Overfitting, on the training cell lines")
    print(f"    best validation {metric.upper()} at epoch {best_epoch} of "
          f"{len(epochs)}: {val[best_index]:.4f}")
    print(f"    training {metric.upper()} at that epoch: "
          f"{train[best_index]:.4f}")
    print(f"    overfitting gap (train minus validation): "
          f"{overfitting_gap:+.4f}")
    if len(epochs) > 1:
        print(f"    gap at the last epoch: {final_gap:+.4f}")
        print(f"    validation lost after its peak: {decline:+.4f}")

    # Loss divergence: the first epoch where validation loss rises while
    # training loss is still falling.
    divergence = None
    for i in range(1, len(epochs)):
        if val_loss[i] > val_loss[i - 1] and train_loss[i] < train_loss[i - 1]:
            divergence = epochs[i]
            break
    report["loss_divergence_epoch"] = divergence
    if divergence:
        print(f"    validation loss first rose while training loss fell at "
              f"epoch {divergence}")
    else:
        print("    validation loss never rose while training loss fell")

    # Transfer gap, which needs the final evaluation.
    if final and final.get(f"val_{metric}") is not None \
            and final.get(f"test_{metric}") is not None:
        val_final = final[f"val_{metric}"]
        test_final = final[f"test_{metric}"]
        transfer = val_final - test_final
        report.update({
            "validation_final": val_final,
            "test_final": test_final,
            "transfer_gap": transfer,
            "transfer_gap_relative": transfer / val_final if val_final else None,
        })
        print(f"\n  Transfer, from training cell lines to unseen cell lines")
        print(f"    validation {metric.upper()} (seen cell lines):   "
              f"{val_final:.4f}")
        print(f"    test {metric.upper()} (unseen cell lines):       "
              f"{test_final:.4f}")
        print(f"    transfer gap: {transfer:+.4f} "
              f"({100 * transfer / val_final:.0f} percent of validation)")
    else:
        print("\n  No final evaluation yet, so the transfer gap is unknown.")

    # Verdict, stated as a reading of the numbers above.
    print("\n  Reading")
    if best_epoch < len(epochs):
        print(f"    Validation {metric.upper()} peaked at epoch {best_epoch} and "
              f"did not improve afterwards.")
        print(f"    The saved checkpoint (best AUROC plus AUPR) is from epoch "
              f"{saved_epoch};")
        print(f"    the epochs after it do not enter the reported result.")
    else:
        print(f"    Validation was still improving at the last epoch "
              f"({len(epochs)}). The budget")
        print("    may be cutting training short; a longer run could score "
              "higher.")

    if overfitting_gap > GAP_CLEAR:
        print(f"    The overfitting gap of {overfitting_gap:.3f} is large. The "
              f"model fits the")
        print("    training cell lines much better than held-out chromosomes "
              "in them.")
    elif overfitting_gap > GAP_MILD:
        print(f"    The overfitting gap of {overfitting_gap:.3f} is moderate "
              f"and normal for")
        print("    this budget.")
    elif overfitting_gap < 0:
        print(f"    The overfitting gap is negative ({overfitting_gap:.3f}): "
              f"validation scores")
        print("    higher than training. That is expected here, because "
              "augmentation is")
        print("    applied to training batches and not to validation, and the "
              "training")
        print("    score is averaged over the epoch while the model is still "
              "changing.")
        print("    It means there is no sign of memorisation at all.")
    else:
        print(f"    The overfitting gap of {overfitting_gap:.3f} is small.")

    if report.get("transfer_gap") is not None:
        transfer = report["transfer_gap"]
        # Compare magnitudes: a negative overfitting gap means no memorisation,
        # so any positive transfer gap is the larger problem.
        if transfer > max(overfitting_gap, 0.0):
            print(f"    The transfer gap ({transfer:.3f}) is the larger of the "
                  f"two, so the")
            print("    harder problem is the change of cell type, not "
                  "memorisation.")
            print("    Shortening training would not fix it.")
        else:
            print(f"    The overfitting gap ({overfitting_gap:.3f}) exceeds the "
                  f"transfer gap")
            print(f"    ({transfer:.3f}), so regularisation or a shorter "
                  f"budget is worth trying.")

    return report


def main():
    args = parse_args()
    run_dirs = []
    for pattern in args.run:
        matches = sorted(glob.glob(pattern))
        run_dirs.extend(matches if matches else [pattern])

    reports = {}
    for run_dir in run_dirs:
        result = describe(run_dir, args.metric)
        if result:
            reports[run_dir] = result
        print()

    # Compare transfer gaps, which is the comparison the manuscript cares about.
    with_transfer = {k: v for k, v in reports.items()
                     if v.get("transfer_gap") is not None}
    if len(with_transfer) > 1:
        print("=" * 74)
        print(f"  Transfer gap by run, smallest first "
              f"({args.metric.upper()})")
        print("=" * 74)
        print(f"  {'run':<34}{'validation':>12}{'test':>10}{'gap':>10}")
        for run_dir, entry in sorted(with_transfer.items(),
                                     key=lambda kv: kv[1]["transfer_gap"]):
            print(f"  {run_dir:<34}{entry['validation_final']:>12.4f}"
                  f"{entry['test_final']:>10.4f}{entry['transfer_gap']:>10.4f}")
        print("\n  A smaller transfer gap at a similar validation score means "
              "the model")
        print("  carries more of what it learned to an unseen cell type, "
              "which is the")
        print("  property this work claims to improve.")

    if args.out and reports:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(reports, handle, indent=2, default=float)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()

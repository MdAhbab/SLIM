"""
Score saved predictions: thresholds, calibration, uncertainty and leakage.

Nothing here runs the model. Every number comes from the per-example
predictions written by `scripts/train.py`, so the whole analysis takes seconds
on a laptop and can be repeated without touching the GPU.

What it reports for each run:

  threshold-free   AUROC and AUPR, with a bootstrap confidence interval over
                   test examples
  at 0.5           the fixed threshold the manuscript reports
  at a chosen      the threshold that maximises MCC on VALIDATION, which is
  threshold        the operating point a user could actually pick
  calibration      expected calibration error, the largest bin gap, and the
                   Brier score, plus the bins a reliability diagram needs
  per cell line    the same metrics for each held-out cell line
  leakage subsets  metrics on the test pairs that share no locus with
                   training, next to the full test set

Usage:

    python scripts/evaluate.py --run results/ka
    python scripts/evaluate.py --run results/baseline results/ka/seed0 \
        --leakage results/leakage --out results/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    bootstrap_ci, compute_all_metrics, expected_calibration_error,
    metrics_at_threshold, select_threshold,
)

REPORTED = ("auroc", "aupr", "accuracy", "balanced_accuracy",
            "precision", "recall", "f1", "mcc")


def parse_args():
    p = argparse.ArgumentParser(
        description="Score saved predictions without re-running the model.")
    p.add_argument("--run", nargs="+", required=True,
                   help="run directories holding eval_results.npz")
    p.add_argument("--leakage", default=None,
                   help="directory holding leakage_index.npz from "
                        "scripts/audit_leakage.py")
    p.add_argument("--threshold-metric", default="mcc",
                   choices=["mcc", "f1", "balanced_accuracy"],
                   help="metric maximised when choosing a threshold on "
                        "validation")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples; 0 disables the intervals")
    p.add_argument("--out", default=None, help="write the report as JSON")
    return p.parse_args()


def load_run(run_dir):
    """Read predictions from a run directory.

    Accepts both the current layout, where validation predictions live in
    eval_results.npz, and the earlier runs that kept them in results.npz.
    """
    eval_path = os.path.join(run_dir, "eval_results.npz")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"{eval_path} not found")
    data = np.load(eval_path, allow_pickle=True)
    run = {
        "dir": run_dir,
        "test_predictions": data["test_predictions"],
        "test_labels": data["test_labels"],
        "cells": {},
    }
    for key in ("variant", "seed", "split", "modalities", "total_params"):
        if key in data.files:
            run[key] = data[key].item()

    if "val_predictions" in data.files:
        run["val_predictions"] = data["val_predictions"]
        run["val_labels"] = data["val_labels"]
    else:
        legacy = os.path.join(run_dir, "results.npz")
        if os.path.exists(legacy):
            old = np.load(legacy, allow_pickle=True)
            if "val_preds" in old.files:
                run["val_predictions"] = old["val_preds"]
                run["val_labels"] = old["val_labels"]

    for key in data.files:
        if key.endswith("_predictions") and key not in (
                "test_predictions", "val_predictions"):
            cell = key[: -len("_predictions")]
            if f"{cell}_labels" in data.files:
                run["cells"][cell] = (data[key], data[f"{cell}_labels"])
    return run


def summarise(labels, probs, threshold, bootstrap=0, seed=0):
    """Metrics at one threshold, with optional bootstrap intervals."""
    out = {k: v for k, v in metrics_at_threshold(labels, probs, threshold).items()
           if k in REPORTED or k == "threshold"}
    out["n"] = int(len(labels))
    out["positives"] = int(np.asarray(labels).sum())
    if bootstrap:
        for metric in ("aupr", "mcc"):
            ci = bootstrap_ci(labels, probs, metric=metric,
                              n_resamples=bootstrap, seed=seed,
                              threshold=threshold)
            out[f"{metric}_ci_lower"] = ci["lower"]
            out[f"{metric}_ci_upper"] = ci["upper"]
    return out


def print_block(title, metrics):
    print(f"\n  {title}")
    print(f"    n={metrics['n']:,}  positives={metrics['positives']:,} "
          f"({100 * metrics['positives'] / max(metrics['n'], 1):.1f} percent)"
          f"  threshold={metrics['threshold']:.3f}")
    line = "    "
    for key in REPORTED:
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        line += f"{key.upper().replace('BALANCED_ACCURACY','BAL_ACC')}={value:.4f}  "
    print(line.rstrip())
    for metric in ("aupr", "mcc"):
        lo, hi = metrics.get(f"{metric}_ci_lower"), metrics.get(f"{metric}_ci_upper")
        if lo is not None and not np.isnan(lo):
            print(f"    {metric.upper()} 95 percent interval "
                  f"[{lo:.4f}, {hi:.4f}]")


def main():
    args = parse_args()

    leakage = None
    if args.leakage:
        path = os.path.join(args.leakage, "leakage_index.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run scripts/audit_leakage.py first")
        leakage = np.load(path, allow_pickle=True)

    report = {}
    for run_dir in args.run:
        run = load_run(run_dir)
        labels = run["test_labels"]
        probs = run["test_predictions"]

        print("=" * 70)
        label = run.get("variant", os.path.basename(os.path.normpath(run_dir)))
        print(f"  {run_dir}   variant {label}"
              + (f", seed {run['seed']}" if "seed" in run else ""))
        print("=" * 70)

        entry = {"dir": run_dir}
        for key in ("variant", "seed", "split", "modalities", "total_params"):
            if key in run:
                entry[key] = run[key]

        entry["fixed_threshold"] = summarise(
            labels, probs, 0.5, bootstrap=args.bootstrap)
        print_block("Test set, fixed threshold 0.5", entry["fixed_threshold"])

        if "val_predictions" in run:
            chosen, value = select_threshold(
                run["val_labels"], run["val_predictions"],
                metric=args.threshold_metric)
            entry["selected_threshold"] = {
                "threshold": chosen,
                "chosen_on": "validation",
                "metric": args.threshold_metric,
                "validation_value": value,
            }
            entry["at_selected_threshold"] = summarise(
                labels, probs, chosen, bootstrap=args.bootstrap)
            print_block(
                f"Test set, threshold {chosen:.3f} chosen on validation by "
                f"{args.threshold_metric.upper()}",
                entry["at_selected_threshold"])
        else:
            print("\n  No validation predictions saved for this run, so a "
                  "threshold\n  cannot be chosen honestly. Only the fixed "
                  "0.5 threshold is reported.")

        calibration = expected_calibration_error(labels, probs)
        entry["calibration"] = calibration
        print(f"\n  Calibration: ECE={calibration['ece']:.4f}  "
              f"largest bin gap={calibration['max_calibration_error']:.4f}  "
              f"Brier={calibration['brier']:.4f}")

        entry["per_cell"] = {}
        for cell, (cell_probs, cell_labels) in sorted(run["cells"].items()):
            cell_metrics = summarise(cell_labels, cell_probs, 0.5, bootstrap=0)
            entry["per_cell"][cell] = cell_metrics
            print_block(f"Cell line {cell}, threshold 0.5", cell_metrics)

        if leakage is not None:
            n_leak = len(leakage["disjoint"])
            if n_leak != len(labels):
                print(f"\n  Leakage index holds {n_leak:,} rows but this run "
                      f"has {len(labels):,} test pairs.\n  They do not "
                      f"correspond, so the subset analysis is skipped. Check "
                      f"that\n  the audit used the same cell lines.")
            else:
                entry["leakage"] = {}
                for name, mask in (("disjoint", leakage["disjoint"]),
                                   ("reused", leakage["either_reused"])):
                    if mask.sum() == 0:
                        continue
                    subset = summarise(labels[mask], probs[mask], 0.5,
                                       bootstrap=args.bootstrap)
                    entry["leakage"][name] = subset
                    title = ("Test pairs sharing no locus with training"
                             if name == "disjoint"
                             else "Test pairs reusing a training locus")
                    print_block(title, subset)

                if "disjoint" in entry["leakage"] and "reused" in entry["leakage"]:
                    d = entry["leakage"]["disjoint"]
                    r = entry["leakage"]["reused"]
                    print(f"\n  AUPR on reused loci minus disjoint loci: "
                          f"{r['aupr'] - d['aupr']:+.4f}")
                    print("  A gap near zero means the result does not depend "
                          "on reused territory.")

        report[run_dir] = entry

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2, default=float)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()

"""
Combine repeated runs into per-seed values, means and paired differences.

Two different uncertainties matter and they are not interchangeable:

  between runs   how much the result moves when only the random seed changes.
                 Repeated training runs measure this, and it is what
                 "single-seed results" criticism is about.
  within a run   how much the result would move on a different sample of test
                 pairs. The bootstrap in `scripts/evaluate.py` measures this.

This script reports the first. With three seeds a paired t-test has almost no
power, so it prints every seed's value, the mean, the standard deviation and
the paired differences, and it does not assert significance. If you later run
enough seeds for a test to mean something, pass --paired-test to add one.

Usage:

    python scripts/aggregate_seeds.py --results results
    python scripts/aggregate_seeds.py --reference baseline --paired-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_all_metrics

METRICS = ("auroc", "aupr", "balanced_accuracy", "f1", "mcc")


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate repeated runs of each variant.")
    p.add_argument("--results", default="results",
                   help="root directory holding <variant>/seed<N>/")
    p.add_argument("--reference", default="baseline",
                   help="variant that paired differences are measured against")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="threshold for F1, MCC and balanced accuracy")
    p.add_argument("--paired-test", action="store_true",
                   help="also report a paired t-test; only meaningful with a "
                        "reasonable number of seeds")
    p.add_argument("--out", default="results/seed_summary.json")
    return p.parse_args()


def discover_runs(root):
    """Find every seeded run, grouped by variant directory name."""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*", "seed*",
                                              "eval_results.npz"))):
        seed_dir = os.path.basename(os.path.dirname(path))
        variant = os.path.basename(os.path.dirname(os.path.dirname(path)))
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue
        runs[variant][seed] = path
    return runs


def score(path, threshold):
    data = np.load(path, allow_pickle=True)
    labels = data["test_labels"]
    probs = data["test_predictions"]
    preds = (probs >= threshold).astype(int)
    metrics = compute_all_metrics(labels, probs, preds=preds)
    return {k: metrics.get(k) for k in METRICS}


def main():
    args = parse_args()
    runs = discover_runs(args.results)
    if not runs:
        print(f"No seeded runs found under {args.results}/<variant>/seed<N>/.")
        print("Train at least one, for example:")
        print("  python scripts/train.py --config configs/slim_ka.yaml --seed 0")
        return

    print("=" * 78)
    print(f"  Repeated runs, threshold {args.threshold}")
    print("=" * 78)

    per_variant = {}
    for variant in sorted(runs):
        seeds = sorted(runs[variant])
        rows = {seed: score(runs[variant][seed], args.threshold)
                for seed in seeds}
        per_variant[variant] = rows

        print(f"\n  {variant}  ({len(seeds)} "
              f"{'seed' if len(seeds) == 1 else 'seeds'}: "
              f"{', '.join(str(s) for s in seeds)})")
        header = "    seed  " + "".join(f"{m.upper():>12}" for m in METRICS)
        print(header)
        for seed in seeds:
            line = f"    {seed:<6}"
            for metric in METRICS:
                value = rows[seed].get(metric)
                line += f"{value:>12.4f}" if value is not None else f"{'n/a':>12}"
            print(line)
        if len(seeds) > 1:
            for label, fn in (("mean", np.mean), ("sd", np.std)):
                line = f"    {label:<6}"
                for metric in METRICS:
                    values = [rows[s][metric] for s in seeds
                              if rows[s][metric] is not None]
                    value = fn(values, **({"ddof": 1} if label == "sd" and
                                          len(values) > 1 else {}))
                    line += f"{value:>12.4f}"
                print(line)
        else:
            print("    Only one seed, so no spread can be reported.")

    summary = {"threshold": args.threshold,
               "variants": {v: {str(s): r for s, r in rows.items()}
                            for v, rows in per_variant.items()}}

    reference = args.reference
    if reference in per_variant:
        ref_rows = per_variant[reference]
        print(f"\n{'=' * 78}")
        print(f"  Paired differences against {reference}, "
              f"on seeds both variants ran")
        print("=" * 78)
        summary["paired_differences"] = {}
        for variant in sorted(per_variant):
            if variant == reference:
                continue
            shared = sorted(set(per_variant[variant]) & set(ref_rows))
            if not shared:
                continue
            print(f"\n  {variant} minus {reference}  "
                  f"(seeds {', '.join(str(s) for s in shared)})")
            entry = {}
            for metric in METRICS:
                diffs = [per_variant[variant][s][metric] - ref_rows[s][metric]
                         for s in shared
                         if per_variant[variant][s][metric] is not None
                         and ref_rows[s][metric] is not None]
                if not diffs:
                    continue
                diffs = np.asarray(diffs, dtype=float)
                record = {"n_seeds": len(diffs),
                          "mean_difference": float(diffs.mean()),
                          "per_seed": [float(d) for d in diffs]}
                text = (f"    {metric.upper():<20} "
                        f"mean {diffs.mean():+.4f}")
                if len(diffs) > 1:
                    record["sd_difference"] = float(diffs.std(ddof=1))
                    text += f"   sd {diffs.std(ddof=1):.4f}"
                    text += ("   every seed agrees in sign"
                             if np.all(np.sign(diffs) == np.sign(diffs[0]))
                             else "   seeds disagree in sign")
                if args.paired_test and len(diffs) > 1:
                    from scipy import stats
                    t_stat, p_value = stats.ttest_rel(
                        [per_variant[variant][s][metric] for s in shared],
                        [ref_rows[s][metric] for s in shared])
                    record["t_statistic"] = float(t_stat)
                    record["p_value"] = float(p_value)
                    text += f"   t={t_stat:.3f}, p={p_value:.3f}"
                print(text)
                entry[metric] = record
            summary["paired_differences"][variant] = entry

        n_seeds = max(len(rows) for rows in per_variant.values())
        if n_seeds < 5:
            print(f"\n  With {n_seeds} seeds these differences describe what "
                  f"was observed.")
            print("  They do not support a claim of statistical significance. "
                  "Report the")
            print("  per-seed values and the spread, and pair them with the "
                  "bootstrap")
            print("  intervals from scripts/evaluate.py.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()

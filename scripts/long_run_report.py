"""
Report of the long-training suite (python run.py --suite long).

Reads the runs the suite writes and answers three questions from the saved
histories and predictions:

1. How does each model learn over 30 epochs? Clean training and validation
   AUPR per epoch, the epoch of the best validation score, and the epoch at
   which early stopping ended the run.
2. Does longer training change test accuracy? The long runs against the
   standard eight-epoch runs of the main suite, paired by seed.
3. Does more training data help? Test AUPR of SLIM-GA on 25, 50 and 100
   percent of the training rows, all at seed 0.

The checkpoint that is tested is always the one with the best validation
score, so no number here is chosen on the test set. Missing runs are skipped,
so the report can be built while the suite is still running.

Usage:

    python scripts/long_run_report.py --out results/long
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

import paper_figures as pf
import matplotlib.pyplot as plt

PAIRS = [("ga_long", "ga", "SLIM-GA", pf.GREEN, "D"),
         ("baseline_long", "baseline", "Baseline", pf.GREY, "s")]
FRACTIONS = [(0.25, "ga_long_frac25"), (0.5, "ga_long_frac50"), (1.0, "ga_long")]


def parse_args():
    p = argparse.ArgumentParser(description="Report the long-training suite.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/long")
    p.add_argument("--figdir", default="figures")
    return p.parse_args()


def load_run(path):
    """History and test scores of one run directory, or None if unfinished."""
    npz = os.path.join(path, "eval_results.npz")
    hist = os.path.join(path, "history.json")
    if not (os.path.exists(npz) and os.path.exists(hist)):
        return None
    with open(hist, encoding="utf-8") as handle:
        history = json.load(handle)
    data = np.load(npz)
    selection = [(r.get("val_auroc") or 0) + (r.get("val_aupr") or 0) for r in history]
    best = int(np.argmax(selection))
    record = {
        "epochs_run": len(history),
        "best_epoch": history[best]["epoch"],
        "val_aupr_best": history[best].get("val_aupr"),
        "val_auroc_best": history[best].get("val_auroc"),
        "clean_train_aupr_best": history[best].get("train_clean_aupr"),
        "test_auroc": float(data["test_auroc"]),
        "test_aupr": float(data["test_aupr"]),
        "train_rows": int(data["train_rows"]) if "train_rows" in data else None,
        "curve": {
            "epoch": [r["epoch"] for r in history],
            "val_aupr": [r.get("val_aupr") for r in history],
            "train_clean_aupr": [r.get("train_clean_aupr") for r in history],
        },
    }
    for cell in ("HMEC", "NHEK"):
        if f"{cell}_aupr" in data:
            record[f"aupr_{cell}"] = float(data[f"{cell}_aupr"])
    return record


def runs_of(results, name):
    found = {}
    for path in sorted(glob.glob(os.path.join(results, name, "seed*"))):
        record = load_run(path)
        if record is not None:
            found[int(os.path.basename(path)[4:])] = record
    return found


def paired(long_runs, short_runs, metric):
    """Long minus standard training, over the seeds both have finished."""
    seeds = sorted(set(long_runs) & set(short_runs))
    diffs = [long_runs[s][metric] - short_runs[s][metric] for s in seeds]
    out = {"seeds": seeds, "differences": diffs}
    if len(diffs) >= 2:
        out["mean"] = float(np.mean(diffs))
        out["sd"] = float(np.std(diffs, ddof=1))
        t = stats.ttest_rel([long_runs[s][metric] for s in seeds],
                            [short_runs[s][metric] for s in seeds])
        out["t"], out["p"] = float(t.statistic), float(t.pvalue)
    return out


def summarise(runs, metric):
    vals = [r[metric] for r in runs.values()]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "n": len(vals)}


def figure(report, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(pf.DOUBLE, 2.6), gridspec_kw={"wspace": 0.38})
    for ax, (long_name, _, label, colour, marker), letter in zip(axes[:2], PAIRS, "ab"):
        for seed, record in report["runs"][long_name].items():
            c = record["curve"]
            ax.plot(c["epoch"], c["train_clean_aupr"], color=colour, linestyle="--",
                    linewidth=1.0, alpha=0.8)
            ax.plot(c["epoch"], c["val_aupr"], color=colour, marker=marker,
                    markersize=3, linewidth=1.2)
            ax.axvline(record["best_epoch"], color=colour, linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}, long recipe")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUPR")
        pf.style(ax)
        pf.panel(ax, letter)
    ax = axes[2]
    points = [(f, report["scaling"][str(f)]) for f, _ in FRACTIONS
              if report["scaling"].get(str(f))]
    if points:
        xs = [f * 100 for f, _ in points]
        ax.plot(xs, [p["test_aupr"] for _, p in points], color=pf.GREEN, marker="D",
                label="Test AUPR")
        ax.plot(xs, [p["val_aupr_best"] for _, p in points], color=pf.GREEN,
                marker="D", linestyle="--", alpha=0.6, label="Validation AUPR")
        for x, (_, p) in zip(xs, points):
            ax.text(x, p["test_aupr"] + 0.01, f"{p['test_aupr']:.3f}", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xlabel("Training rows used (percent)")
    ax.set_ylabel("AUPR")
    ax.set_xticks([25, 50, 100])
    ax.margins(y=0.2)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    pf.style(ax)
    pf.panel(ax, "c")
    handles = [plt.Line2D([], [], color=pf.INK, linestyle="--"),
               plt.Line2D([], [], color=pf.INK, marker="o", markersize=3),
               plt.Line2D([], [], color=pf.INK, linestyle=":")]
    fig.legend(handles, ["Clean training sample", "Validation (Hi-C)", "Best validation epoch"],
               frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.14))
    os.makedirs(figdir, exist_ok=True)
    pf.save(fig, figdir, "fig_long_training")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"runs": {}, "standard": {}, "long_vs_standard": {}, "summary": {},
              "scaling": {}}
    for long_name, short_name, label, *_ in PAIRS:
        long_runs = runs_of(args.results, long_name)
        short_runs = runs_of(args.results, short_name)
        report["runs"][long_name] = long_runs
        report["summary"][long_name] = {
            m: summarise(long_runs, m) for m in ("test_aupr", "test_auroc", "best_epoch",
                                                 "epochs_run", "val_aupr_best")}
        report["summary"][short_name] = {
            m: summarise(short_runs, m) for m in ("test_aupr", "test_auroc", "best_epoch",
                                                  "epochs_run", "val_aupr_best")}
        report["long_vs_standard"][long_name] = {
            m: paired(long_runs, short_runs, m) for m in ("test_aupr", "test_auroc")}
        print(f"{label}: {len(long_runs)} long run(s), {len(short_runs)} standard run(s)")
        for seed, r in sorted(long_runs.items()):
            print(f"  seed {seed}: best epoch {r['best_epoch']} of {r['epochs_run']}, "
                  f"test AUPR {r['test_aupr']:.4f}, AUROC {r['test_auroc']:.4f}")
    ga_long = report["runs"]["ga_long"]
    for fraction, name in FRACTIONS:
        record = runs_of(args.results, name).get(0) if name != "ga_long" else ga_long.get(0)
        if record:
            report["scaling"][str(fraction)] = {
                k: record[k] for k in ("test_aupr", "test_auroc", "val_aupr_best",
                                       "best_epoch", "train_rows")}
            print(f"  {int(fraction * 100)} percent of rows: test AUPR {record['test_aupr']:.4f}")
    with open(os.path.join(args.out, "long_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {os.path.join(args.out, 'long_report.json')}")
    if any(report["runs"].values()):
        figure(report, args.figdir)


if __name__ == "__main__":
    main()

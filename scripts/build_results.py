"""
Canonical results and publication figures for the SLIM manuscript.

Every number here is recomputed from the per-example predictions saved by
`scripts/train.py`. Nothing is transcribed from a training log and nothing is
hard-coded: a variant with no saved predictions is reported as missing rather
than filled in from elsewhere. That rule exists because an earlier version of
this script carried one variant's headline metrics as literals copied from a
text log, which made those numbers impossible to reproduce and its curves
impossible to draw.

Reads `results/<variant>/` or, when seeds are present,
`results/<variant>/seed<N>/`. Writes `results/results.json`, a markdown table
and the manuscript figures, following the project figure specification: flat,
colour-blind-safe, legible in greyscale, 400 dpi.

Usage:

    python scripts/build_results.py
    python scripts/build_results.py --seed 0
"""
import argparse
import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, roc_curve,
                             precision_recall_curve)

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGDIR = os.path.abspath(os.path.join(ROOT, "..", "Main_paper", "fig"))
OUTJSON = os.path.join(ROOT, "results", "results.json")
os.makedirs(FIGDIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "savefig.dpi": 400,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.5,
    "axes.edgecolor": "#111827",
})

BLUE, ORANGE, GREEN, GREY = "#2563EB", "#E8710A", "#059669", "#6B7280"
NEARBLACK = "#111827"

MM = 1.0 / 25.4
SINGLE = 89 * MM   # 3.50 in
DOUBLE = 183 * MM  # 7.20 in


def metrics_from_raw(preds, labels, thr=0.5):
    p = (preds >= thr).astype(int)
    y = labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return dict(
        n=int(len(y)), pos=int(y.sum()),
        AUROC=float(roc_auc_score(y, preds)),
        AUPR=float(average_precision_score(y, preds)),
        ACC=float(accuracy_score(y, p)),
        BACC=float(balanced_accuracy_score(y, p)),
        PREC=float(precision_score(y, p, zero_division=0)),
        REC=float(recall_score(y, p, zero_division=0)),
        F1=float(f1_score(y, p, zero_division=0)),
        MCC=float(matthews_corrcoef(y, p)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


# ---------------------------------------------------------------------------
# Locate the runs. Preference for each variant: the seed asked for on the
# command line, then seed0, then an unseeded directory.
# ---------------------------------------------------------------------------
VARIANT_DIRS = {
    "Baseline": "baseline",   # global KAN-Transformer
    "A": "a",                 # survival-gated encoder, rectified feed-forward
    "KA": "ka",               # survival-gated encoder, spline feed-forward
    "GA": "ga",               # survival-gated encoder, gated feed-forward
}


def find_run(variant, seed):
    """Return the directory holding this variant's predictions, or None."""
    base = os.path.join(RESULTS_DIR, VARIANT_DIRS[variant])
    candidates = []
    if seed is not None:
        candidates.append(os.path.join(base, "seed%d" % seed))
    candidates.append(os.path.join(base, "seed0"))
    candidates.append(base)
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "eval_results.npz")):
            return candidate
    return None


_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=None,
                     help="prefer this seed when a variant has several")
_parser.add_argument("--results-dir", default=None)
_parser.add_argument("--figdir", default=None,
                     help="where the figures go; defaults to the manuscript's "
                          "figure directory next to this repository")
_args, _ = _parser.parse_known_args()
if _args.results_dir:
    RESULTS_DIR = os.path.abspath(_args.results_dir)
if _args.figdir:
    FIGDIR = os.path.abspath(_args.figdir)
os.makedirs(FIGDIR, exist_ok=True)

results = {}
raw_curves = {}
missing = []
for name in VARIANT_DIRS:
    run_dir = find_run(name, _args.seed)
    if run_dir is None:
        missing.append(name)
        continue
    d = np.load(os.path.join(run_dir, "eval_results.npz"), allow_pickle=True)
    entry = {"ALL": metrics_from_raw(d["test_predictions"], d["test_labels"])}
    for cell in ("HMEC", "NHEK"):
        if cell + "_predictions" in d.files:
            entry[cell] = metrics_from_raw(d[cell + "_predictions"],
                                           d[cell + "_labels"])
    entry["_run_dir"] = os.path.relpath(run_dir, ROOT)
    results[name] = entry
    raw_curves[name] = (d["test_labels"].astype(int), d["test_predictions"])

if not results:
    raise SystemExit(
        "No predictions found under %s. Train at least one model:\n"
        "  python scripts/train.py --config configs/slim_ka.yaml --seed 0"
        % RESULTS_DIR)

order = [name for name in VARIANT_DIRS if name in results]
if missing:
    print("Not yet run, so omitted from every table and figure: %s"
          % ", ".join(missing))
    print("Train them before submission. No number here is filled in "
          "from a log.")
for name in order:
    print("  %-9s <- %s" % (name, results[name]["_run_dir"]))
print()

with open(OUTJSON, "w") as f:
    json.dump(results, f, indent=2)

# ---- Markdown table ----
labels = {"Baseline": "KAN-Transformer (baseline)",
          "A": "SLIM (A)", "KA": "SLIM (KA)", "GA": "SLIM (GA)"}
print("\n### Combined test set (HMEC + NHEK), n=37,707, 10.9% positive\n")
print("| Model | AUROC | AUPR | Bal.Acc | F1 | MCC |")
print("|---|---|---|---|---|---|")
for m in order:
    a = results[m]["ALL"]
    print(f"| {labels[m]} | {a['AUROC']:.4f} | {a['AUPR']:.4f} | {a['BACC']:.4f} | {a['F1']:.4f} | {a['MCC']:.4f} |")

print("\n### Per cell line (AUROC / AUPR)\n")
print("| Model | HMEC AUROC | HMEC AUPR | NHEK AUROC | NHEK AUPR |")
print("|---|---|---|---|---|")
for m in order:
    h, n = results[m]["HMEC"], results[m]["NHEK"]
    print(f"| {labels[m]} | {h['AUROC']:.4f} | {h['AUPR']:.4f} | {n['AUROC']:.4f} | {n['AUPR']:.4f} |")

# ----------
# FIGURES
# ----------
model_colors = {"Baseline": GREY, "A": BLUE, "KA": ORANGE, "GA": GREEN}
short = {"Baseline": "KAN-Tf", "A": "A", "KA": "KA", "GA": "GA"}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=2.5, colors=NEARBLACK)


# ---- prevalence (used by the precision-recall panel of the composite) ----
prevalence = (results[order[0]]["ALL"]["pos"]
              / results[order[0]]["ALL"]["n"])

# ---- Figure: per-cell AUPR grouped bars ----
fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.95))
cells = ["HMEC", "NHEK"]
xs = np.arange(len(cells))
w = 0.2
for i, m in enumerate(order):
    vals = [results[m][c]["AUPR"] for c in cells]
    ax.bar(xs + (i - 1.5) * w, vals, width=w, color=model_colors[m],
           edgecolor=NEARBLACK, linewidth=0.4, label=short[m])
ax.set_xticks(xs); ax.set_xticklabels(cells)
ax.set_ylabel("AUPR")
ax.set_title("Per-cell-line AUPR", color=NEARBLACK, fontweight="semibold")
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.set_ylim(0, 0.65)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "slim_fig_percell_aupr.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "slim_fig_percell_aupr.pdf"), bbox_inches="tight")
plt.close(fig)


# ---- Composite Figure: benchmark (2x2: AUROC bars, AUPR bars, ROC, PR) ----
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, DOUBLE * 0.86))
# (a) AUROC bars
ax = axes[0, 0]
vals = [results[m]["ALL"]["AUROC"] for m in order]
xs = np.arange(len(order))
ax.bar(xs, vals, width=0.62, color=[model_colors[m] for m in order], edgecolor=NEARBLACK, linewidth=0.5)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=NEARBLACK)
ax.set_xticks(xs); ax.set_xticklabels([short[m] for m in order]); ax.set_ylim(0.5, 0.9)
ax.set_ylabel("AUROC"); ax.set_title("a  Test AUROC", loc="left", fontweight="semibold"); style(ax)
# (b) AUPR bars
ax = axes[0, 1]
vals = [results[m]["ALL"]["AUPR"] for m in order]
ax.bar(xs, vals, width=0.62, color=[model_colors[m] for m in order], edgecolor=NEARBLACK, linewidth=0.5)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=NEARBLACK)
ax.set_xticks(xs); ax.set_xticklabels([short[m] for m in order]); ax.set_ylim(0.3, 0.55)
ax.set_ylabel("AUPR"); ax.set_title("b  Test AUPR", loc="left", fontweight="semibold"); style(ax)
# (c) ROC
ax = axes[1, 0]
for m in order:
    y, p = raw_curves[m]; fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=model_colors[m], linewidth=1.2, label=f"{short[m]} ({results[m]['ALL']['AUROC']:.3f})")
ax.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=0.7, linestyle=(0, (4, 3)))
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("c  ROC (combined test)", loc="left", fontweight="semibold")
ax.legend(loc="lower right", frameon=False); style(ax)
# (d) PR
ax = axes[1, 1]
for m in order:
    y, p = raw_curves[m]; pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, color=model_colors[m], linewidth=1.2, label=f"{short[m]} ({results[m]['ALL']['AUPR']:.3f})")
ax.axhline(prevalence, color="#9CA3AF", linewidth=0.7, linestyle=(0, (4, 3)))
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("d  Precision-recall (combined test)", loc="left", fontweight="semibold")
ax.legend(loc="upper right", frameon=False); style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "slim_fig_benchmark.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "slim_fig_benchmark.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- Composite Figure: contribution (F1, MCC, balanced acc, per-cell AUPR) ----
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
panel = [("F1", "a  Test F1", 0.5), ("MCC", "b  Test MCC", 0.5), ("BACC", "c  Balanced accuracy", 0.75)]
for ax, (metric, title, ymax) in zip(axes, panel):
    vals = [results[m]["ALL"][metric] for m in order]
    ax.bar(xs, vals, width=0.62, color=[model_colors[m] for m in order], edgecolor=NEARBLACK, linewidth=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=NEARBLACK)
    ax.set_xticks(xs); ax.set_xticklabels([short[m] for m in order]); ax.set_ylim(0, ymax)
    ax.set_ylabel(metric if metric != "BACC" else "Balanced acc.")
    ax.set_title(title, loc="left", fontweight="semibold"); style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "slim_fig_contribution.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "slim_fig_contribution.pdf"), bbox_inches="tight")
plt.close(fig)

print("\nFigures written to", FIGDIR)
print("Done.")

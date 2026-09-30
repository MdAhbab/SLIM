"""
Publication data figures for the SLIM manuscript, supplement and thesis.

Every panel is drawn from the files the pipeline saved (per-example predictions,
histories and JSON reports), through `results/paper_numbers.json` where a
summary is needed. Run `scripts/paper_numbers.py` first. Nothing is typed in by
hand, and the figures are never edited afterwards: change this script instead.

Style follows the project figure specification: flat 2D, white background,
sans-serif at 8.5 to 9.5 pt, one colour per model kept across every figure,
and a marker or hatch per model so the figures read in greyscale.

    Fig. 2  fig_audit        what the pipeline audit found
    Fig. 3  fig_benchmark    unseen cell lines, cost
    Fig. 4  fig_memory       ablations and what the memory writes
    Supp.   figS_curves, figS_calibration, figS_pr

Usage:

    python scripts/paper_figures.py                 # writes figures/
    python scripts/paper_figures.py --out ../Main_paper/fig
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 10.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#111827",
    "axes.labelcolor": "#111827",
    "xtick.color": "#111827",
    "ytick.color": "#111827",
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE, ORANGE, GREEN, GREY, INK = "#2563EB", "#E8710A", "#059669", "#6B7280", "#111827"
MM = 1 / 25.4
DOUBLE = 183 * MM

# One colour, marker and hatch per model, used in every figure.
MODELS = [("baseline", "Baseline", GREY, "s", ""),
          ("a", "SLIM-A", BLUE, "o", "//"),
          ("ka", "SLIM-KA", ORANGE, "^", ".."),
          ("ga", "SLIM-GA", GREEN, "D", "\\\\")]
EFF_KEY = {"baseline": "baseline", "a": "A", "ka": "KA", "ga": "GA"}


def parse_args():
    p = argparse.ArgumentParser(description="Draw the manuscript data figures.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    return p.parse_args()


def style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="bottom", ha="right")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def seed_panel(ax, N, metric, ylabel):
    """Every seed as a point, with the mean and one standard deviation."""
    for i, (key, label, colour, marker, _) in enumerate(MODELS):
        block = N["models"][key][metric]
        vals = np.array(block["values"])
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(i + jitter, vals, s=16, color=colour, marker=marker, alpha=0.55,
                   edgecolor="none", zorder=3)
        ax.errorbar(i + 0.3, block["mean"], yerr=block["sd"], fmt=marker, color=colour,
                    ecolor=INK, elinewidth=1.0, capsize=3, markersize=6, zorder=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(MODELS) - 0.4)
    style(ax)


def fig_audit(N, R, out):
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 4.9), gridspec_kw={"wspace": 0.42, "hspace": 0.62})
    axes = axs.ravel()
    width = 0.36

    # a: label separation carried by the sequence encoding
    ax = axes[0]
    rules = N["encoder_leakage"]["rules"]
    groups = [("fitted", "Rows used\nto fit"), ("unfitted", "Other\ntraining rows"),
              ("validation", "Validation\nrows")]
    for j, (rule, label, colour, hatch) in enumerate((("train", "Fitted on training rows", GREY, ""),
                                                      ("crossfit", "Cross-fitted by chromosome", BLUE, "//"))):
        vals = [rules[rule][g].get("auroc") for g, _ in groups]
        xs = np.arange(len(groups)) + (j - 0.5) * width
        for x, v in zip(xs, vals):
            if v is None:
                ax.text(x, 0.505, "none", ha="center", va="bottom", fontsize=8, color=GREY, rotation=90)
                continue
            ax.bar(x, v, width, color=colour, hatch=hatch, edgecolor="white", linewidth=0.5,
                   label=label if x == xs[0] else None)
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color=INK, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylim(0.45, 1.12)
    ax.set_ylabel("AUROC of encoding score")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=8)
    style(ax)
    panel(ax, "a")

    # b: distance between the two anchors, by assay and class
    ax = axes[1]
    dist = N["dataset_audit"]["hic_distance_quantiles_kb"]
    assays = [("HiC", "Hi-C"), ("CTCF-ChIAPET", "CTCF\nChIA-PET"), ("RNAPII-ChIAPET", "RNAPII\nChIA-PET")]
    for i, (key, label) in enumerate(assays):
        colour = BLUE if key == "HiC" else GREY
        for cls, off, mk, alpha in (("positive", -0.14, "o", 0.9), ("negative", 0.14, "s", 0.35)):
            q = dist[key][cls]["quantiles_kb"]
            ax.plot([i + off] * 2, [q[1], q[3]], color=colour, linewidth=5, solid_capstyle="butt",
                    alpha=alpha)
            ax.plot([i + off] * 2, [q[0], q[4]], color=colour, linewidth=1)
            ax.plot(i + off, q[2], marker=mk, color="white", markeredgecolor=INK, markersize=5.5)
    ax.set_yscale("log")
    ax.set_ylim(5, 2500)
    ax.set_xticks(range(len(assays)))
    ax.set_xticklabels([a[1] for a in assays])
    ax.set_ylabel("Anchor distance (kb, log scale)")
    ax.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.plot([], [], marker="o", color="white", markeredgecolor=INK, linestyle="", label="Positive pairs")
    ax.plot([], [], marker="s", color="white", markeredgecolor=INK, linestyle="", label="Negative pairs")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.2)
    style(ax)
    panel(ax, "b")

    # c: reference predictors without a network, by the rows they were fitted on
    ax = axes[2]
    names = [("distance_rank", "Distance\nonly"), ("gene_prior_only", "Gene\npositive rate"),
             ("distance_plus_locus_priors", "Distance and\nlocus rates")]
    for j, (fit, label, colour, hatch) in enumerate((("trained_on_all_assays", "Fitted on all assays", GREY, ""),
                                                     ("trained_on_hic", "Fitted on Hi-C rows", BLUE, "//"))):
        vals = [N["baselines"][fit]["test"][k]["aupr"] for k, _ in names]
        xs = np.arange(len(names)) + (j - 0.5) * width
        ax.bar(xs, vals, width, color=colour, hatch=hatch, edgecolor="white", linewidth=0.5, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    net = N["models"]["baseline"]["aupr"]["mean"]
    ax.axhline(net, color=INK, linestyle="--", linewidth=0.8)
    ax.text(-0.45, net - 0.012, "Baseline network (mean, 5 seeds)", ha="left", va="top", fontsize=8)
    ax.axhline(N["runs"]["ka/seed0"]["test_prevalence"], color=INK, linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names])
    ax.set_ylabel("Test AUPR")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=8, ncol=1)
    style(ax)
    panel(ax, "c")

    # d: the four protocol pilots, scored on the same Hi-C validation rows
    ax = axes[3]
    pilots = list(N["pilots"].values())
    labels = [f"{p['assays']}\n{'with DNA' if 'DNA' in p['inputs'] else 'no DNA'}" for p in pilots]
    for i, p in enumerate(pilots):
        c, h = (BLUE, "//") if p["chosen"] else (GREY, "")
        ax.barh(i, p["val_aupr"], 0.6, color=c, hatch=h, edgecolor="white", linewidth=0.5)
        ax.text(p["val_aupr"] + 0.004, i, f"{p['val_aupr']:.3f}" + ("  chosen" if p["chosen"] else ""),
                va="center", fontsize=8)
    ax.set_yticks(range(len(pilots)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Validation AUPR (Hi-C rows, chr11 and chr17)")
    style(ax, grid_axis="x")
    panel(ax, "d")
    save(fig, out, "fig_audit")


def fig_benchmark(N, R, out):
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    seed_panel(ax, N, "auroc", "Test AUROC")
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    seed_panel(ax, N, "aupr", "Test AUPR")
    panel(ax, "b")

    # c: per unseen cell line
    ax = fig.add_subplot(gs[0, 2])
    width = 0.19
    for j, (key, label, colour, marker, hatch) in enumerate(MODELS):
        m = N["models"][key]
        means = [m["aupr_HMEC"]["mean"], m["aupr_NHEK"]["mean"]]
        sds = [m["aupr_HMEC"]["sd"], m["aupr_NHEK"]["sd"]]
        xs = np.arange(2) + (j - 1.5) * width
        ax.bar(xs, means, width, yerr=sds, color=colour, hatch=hatch, edgecolor="white",
               linewidth=0.5, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK},
               label=label)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HMEC", "NHEK"])
    ax.set_ylabel("Test AUPR")
    ax.set_ylim(0.3, 0.78)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left", columnspacing=0.8,
              handlelength=1.2)
    style(ax)
    panel(ax, "c")

    # d: pairs sharing a locus with training versus pairs that share none
    ax = fig.add_subplot(gs[1, 0])
    for j, (key, label, colour, marker, hatch) in enumerate(MODELS):
        m = N["models"][key]
        means = [m["aupr_reused"]["mean"], m["aupr_disjoint"]["mean"]]
        sds = [m["aupr_reused"]["sd"], m["aupr_disjoint"]["sd"]]
        xs = np.arange(2) + (j - 1.5) * width
        ax.bar(xs, means, width, yerr=sds, color=colour, hatch=hatch, edgecolor="white",
               linewidth=0.5, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK})
    ov = N["overlap"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Shares a locus\n(n = {ov['either_reused']:,})",
                        f"Shares none\n(n = {ov['disjoint']:,})"], fontsize=8)
    ax.set_ylabel("Test AUPR")
    ax.set_ylim(0.3, 0.65)
    style(ax)
    panel(ax, "d")

    # e, f: measured cost on one GPU, batch of 32 pairs
    eff = N["efficiency"]["models"]
    for letter, pos, field, scale, ylabel in (
            ("e", gs[1, 1], "inference_ms_median", 1.0, "Inference latency (ms)"),
            ("f", gs[1, 2], "peak_memory_mib_training", 1 / 1024, "Peak training memory (GiB)")):
        ax = fig.add_subplot(pos)
        for i, (key, label, colour, marker, hatch) in enumerate(MODELS):
            v = eff[EFF_KEY[key]][field] * scale
            ax.bar(i, v, 0.62, color=colour, hatch=hatch, edgecolor="white", linewidth=0.5)
            ax.text(i, v * 1.02, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m[1] for m in MODELS], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(eff[EFF_KEY[k]][field] * scale for k, *_ in MODELS) * 1.18)
        style(ax)
        panel(ax, letter)
    save(fig, out, "fig_benchmark")


def fig_memory(N, R, out):
    fig = plt.figure(figsize=(DOUBLE, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.25], wspace=0.55)

    # a: ablations at seed 0 against the spread of SLIM-KA over five seeds
    ax = fig.add_subplot(gs[0, 0])
    abl = list(N["ablations"].values())
    ka = N["models"]["ka"]["aupr"]
    ax.axvspan(ka["min"], ka["max"], color=ORANGE, alpha=0.12, linewidth=0)
    ax.axvline(ka["mean"], color=ORANGE, linewidth=1.0)
    ax.fill_between([], [], color=ORANGE, alpha=0.12, label="SLIM-KA, 5 seeds")
    for i, rec in enumerate(abl):
        chosen = i == 0
        colour = ORANGE if chosen else GREY
        ax.barh(i, rec["aupr"], 0.62, color=colour, hatch=".." if chosen else "",
                edgecolor="white", linewidth=0.5)
        ax.text(rec["aupr"] + 0.004, i, f"{rec['aupr']:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels([r["label"].replace("SLIM-KA, chosen protocol", "SLIM-KA (seed 0)")
                        for r in abl], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.6)
    ax.set_xlabel("Test AUPR")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), handlelength=1.2)
    style(ax, grid_axis="x")
    panel(ax, "a")

    # b: chromatin signal in written versus unwritten bins
    ax = fig.add_subplot(gs[0, 1])
    tracks = ["H3K27ac", "H3K4me3", "DNase", "CTCF", "H3K27me3"]
    runs = [("ka/seed0", "SLIM-KA", ORANGE, "^"), ("ga/seed0", "SLIM-GA", GREEN, "D"),
            ("ka_legacy/seed0", "Original rules", GREY, "s")]
    for j, (rel, label, colour, marker) in enumerate(runs):
        w = N["writelog"].get(rel)
        if not w:
            continue
        vals = [w["tracks"][t]["ratio"] for t in tracks]
        ys = np.arange(len(tracks)) + (j - 1) * 0.22
        ax.scatter(vals, ys, color=colour, marker=marker, s=22, label=label, zorder=3)
    ax.axvline(1.0, color=INK, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 1.5, 2, 3])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels(tracks)
    ax.invert_yaxis()
    ax.set_xlabel("Written / unwritten (log scale)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", handletextpad=0.2)
    style(ax, grid_axis="x")
    panel(ax, "b")

    # c: where SLIM-KA writes, by token position
    ax = fig.add_subplot(gs[0, 2])
    w = N["writelog"]["ka/seed0"]
    with open(os.path.join(R, "ka", "seed0", "writelog.json"), encoding="utf-8") as handle:
        hist = np.array(json.load(handle)["token_histogram"], dtype=float)
    frac = hist / w["n_windows"]
    ax.bar(np.arange(128), frac[:128], 1.0, color=BLUE, linewidth=0)
    ax.bar(np.arange(128, len(frac)), frac[128:], 1.0, color=ORANGE, linewidth=0)
    ax.axvline(127.5, color=INK, linestyle=":", linewidth=0.8)
    ax.text(64, 1.1, "DNA", ha="center", va="top", fontsize=8, color=BLUE)
    ax.text(378, 1.1, "Chromatin (2.5 Mb)", ha="center", va="top", fontsize=8, color=ORANGE)
    ax.set_xlim(0, len(frac))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Windows with a write")
    style(ax)
    panel(ax, "c")
    save(fig, out, "fig_memory")


def figS_curves(N, R, out):
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.6), sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (key, label, colour, marker, _) in zip(axes, MODELS):
        ax.set_xlim(0.6, 8.4)
        for seed_dir in sorted(glob.glob(os.path.join(R, key, "seed*"))):
            with open(os.path.join(seed_dir, "history.json"), encoding="utf-8") as handle:
                hist = json.load(handle)
            ep = [r["epoch"] for r in hist]
            ax.plot(ep, [r["train_clean_aupr"] for r in hist], color=colour, linestyle="--",
                    linewidth=1.0, alpha=0.8)
            ax.plot(ep, [r["val_aupr"] for r in hist], color=colour, marker=marker,
                    markersize=3.5, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
        style(ax)
    axes[0].set_ylabel("AUPR")
    axes[0].plot([], [], color=INK, linestyle="--", linewidth=1.0, label="Clean training sample")
    axes[0].plot([], [], color=INK, marker="o", markersize=3.5, label="Validation (Hi-C)")
    fig.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    save(fig, out, "figS_curves")


def figS_calibration(N, R, out):
    with open(os.path.join(R, "evaluation.json"), encoding="utf-8") as handle:
        ev = json.load(handle)
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (key, label, colour, marker, _) in zip(axes, MODELS):
        for seed_dir in sorted(glob.glob(os.path.join(R, key, "seed*"))):
            k = os.path.join("results", key, os.path.basename(seed_dir))
            bins = [b for b in ev[k]["calibration"]["bins"] if b["count"] >= 50]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=colour,
                    marker=marker, markersize=3, linewidth=1.0, alpha=0.8)
        ax.plot([0, 1], [0, 1], color=INK, linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}, ECE {N['models'][key]['ece']['mean']:.3f}")
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Observed positive rate")
    save(fig, out, "figS_calibration")


def figS_pr(N, R, out):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 2.8), gridspec_kw={"wspace": 0.3})
    for key, label, colour, marker, _ in MODELS:
        probs, labels = [], None
        for s in (0, 1, 2):
            data = np.load(os.path.join(R, key, f"seed{s}", "eval_results.npz"))
            probs.append(data["test_predictions"])
            labels = data["test_labels"]
        p = np.mean(probs, axis=0)
        fpr, tpr, _ = roc_curve(labels, p)
        prec, rec, _ = precision_recall_curve(labels, p)
        e = N["ensembles"]["seeds_0_2"][key]
        axes[0].plot(fpr, tpr, color=colour, linewidth=1.2, label=f"{label} ({e['auroc']:.3f})")
        axes[1].plot(rec, prec, color=colour, linewidth=1.2, label=f"{label} ({e['aupr']:.3f})")
    axes[0].plot([0, 1], [0, 1], color=INK, linestyle=":", linewidth=0.8)
    axes[1].axhline(N["runs"]["ka/seed0"]["test_prevalence"], color=INK, linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax, letter in zip(axes, "ab"):
        ax.legend(frameon=False, fontsize=7.5, loc="lower right" if letter == "a" else "upper right")
        style(ax, grid_axis="both")
        panel(ax, letter)
    save(fig, out, "figS_pr")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.results, "paper_numbers.json"), encoding="utf-8") as handle:
        N = json.load(handle)
    fig_audit(N, args.results, args.out)
    fig_benchmark(N, args.results, args.out)
    fig_memory(N, args.results, args.out)
    figS_curves(N, args.results, args.out)
    figS_calibration(N, args.results, args.out)
    figS_pr(N, args.results, args.out)


if __name__ == "__main__":
    main()

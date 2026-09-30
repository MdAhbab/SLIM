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
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
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


# Pictograms: small vector line drawings that name the subject of a panel. They
# are drawn in their own inset axes above the panel and carry no data.
LW = 0.9
TINT = {BLUE: "#DBEAFE", ORANGE: "#FFEDD5", GREEN: "#D1FAE5", GREY: "#F3F4F6"}


def icon_axes(ax, rect, xlim=(0, 1), ylim=(0, 1)):
    """An inset in axes coordinates with equal aspect, no frame and no ticks."""
    ia = ax.inset_axes(rect)
    ia.set_xlim(*xlim)
    ia.set_ylim(*ylim)
    ia.set_aspect("equal", adjustable="box", anchor="C")
    ia.axis("off")
    return ia


def ico_cell(ia, x, y, r=0.4, dashed=False, colour=GREY, label=None):
    ia.add_patch(Circle((x, y), r, fill=False, ec=colour, lw=LW,
                        linestyle=(0, (2.2, 1.6)) if dashed else "-"))
    ia.add_patch(Circle((x, y), r * 0.36, fc=colour, ec="none"))
    if label:
        ia.text(x, y - r - 0.12, label, ha="center", va="top", fontsize=8, color=INK)


def ico_die(ia, pips, colour=GREY):
    ia.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.14",
                                fc=TINT.get(colour, "white"), ec=colour, lw=LW))
    spots = {"c": (0.5, 0.5), "tl": (0.28, 0.72), "tr": (0.72, 0.72), "bl": (0.28, 0.28),
             "br": (0.72, 0.28), "ml": (0.28, 0.5), "mr": (0.72, 0.5)}
    layout = {1: ["c"], 2: ["tl", "br"], 3: ["tl", "c", "br"], 4: ["tl", "tr", "bl", "br"],
              5: ["tl", "tr", "c", "bl", "br"], 6: ["tl", "tr", "ml", "mr", "bl", "br"]}
    for s in layout[pips]:
        ia.add_patch(Circle(spots[s], 0.075, fc=INK, ec="none"))


def ico_stopwatch(ia):
    ia.add_patch(Circle((0.5, 0.44), 0.36, fill=False, ec=INK, lw=LW))
    ia.add_patch(Rectangle((0.42, 0.84), 0.16, 0.09, fc=INK, ec="none"))
    ia.plot([0.5, 0.5], [0.8, 0.84], color=INK, lw=LW)
    ia.plot([0.5, 0.64], [0.44, 0.62], color=INK, lw=LW * 1.3, solid_capstyle="round")


def ico_chip(ia):
    ia.add_patch(FancyBboxPatch((0.24, 0.24), 0.52, 0.52, boxstyle="round,pad=0,rounding_size=0.06",
                                fill=False, ec=INK, lw=LW))
    ia.add_patch(Rectangle((0.38, 0.38), 0.24, 0.24, fill=False, ec=INK, lw=LW))
    for t in (0.34, 0.5, 0.66):
        for a, b in (((t, 0.76), (t, 0.9)), ((t, 0.24), (t, 0.1)), ((0.24, t), (0.1, t)), ((0.76, t), (0.9, t))):
            ia.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=LW)


def ico_leak(ia):
    for y in (0.62, 0.06):
        ia.add_patch(FancyBboxPatch((0.18, y), 0.64, 0.28, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=TINT[GREY], ec=GREY, lw=LW))
    ia.add_patch(Polygon([(0.43, 0.45), (0.57, 0.45), (0.5, 0.58)], fc=BLUE, ec="none"))
    ia.add_patch(Circle((0.5, 0.43), 0.075, fc=BLUE, ec="none"))


def ico_loop(ia):
    verts = [(0.02, 0.18), (0.34, 0.18), (0.34, 0.18), (0.05, 0.98), (0.95, 0.98), (0.64, 0.18),
             (0.64, 0.18), (0.98, 0.18)]
    codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.MOVETO, Path.LINETO]
    ia.add_patch(PathPatch(Path(verts, codes), fill=False, ec=GREY, lw=LW * 1.2))
    ia.add_patch(Polygon([(0.34, 0.1), (0.42, 0.18), (0.34, 0.26), (0.26, 0.18)], fc=BLUE, ec="none"))
    ia.plot([0.66, 0.66, 0.8], [0.18, 0.34, 0.34], color=GREEN, lw=LW * 1.3)
    ia.add_patch(Polygon([(0.8, 0.29), (0.88, 0.34), (0.8, 0.39)], fc=GREEN, ec="none"))


def ico_audit(ia):
    ia.add_patch(Rectangle((0.04, 0.22), 0.58, 0.56, fill=False, ec=GREY, lw=LW))
    for y in (0.41, 0.6):
        ia.plot([0.04, 0.62], [y, y], color=GREY, lw=LW * 0.7)
    for x in (0.23, 0.43):
        ia.plot([x, x], [0.22, 0.78], color=GREY, lw=LW * 0.7)
    ia.add_patch(Rectangle((0.04, 0.6), 0.58, 0.18, fc=TINT[BLUE], ec=GREY, lw=LW * 0.7))
    ia.add_patch(Circle((0.66, 0.44), 0.19, fc="white", ec=INK, lw=LW * 1.2))
    ia.plot([0.79, 0.95], [0.31, 0.12], color=INK, lw=LW * 2.2, solid_capstyle="round")


def ico_flasks(ia, chosen=0):
    for i in range(4):
        x0 = i * 1.1
        colour = BLUE if i == chosen else GREY
        ia.add_patch(Polygon([(x0 + 0.36, 0.95), (x0 + 0.36, 0.6), (x0 + 0.05, 0.08), (x0 + 0.95, 0.08),
                              (x0 + 0.64, 0.6), (x0 + 0.64, 0.95)], fill=False, ec=colour, lw=LW))
        ia.add_patch(Polygon([(x0 + 0.22, 0.36), (x0 + 0.05, 0.08), (x0 + 0.95, 0.08), (x0 + 0.78, 0.36)],
                             fc=TINT[colour], ec="none"))


def ico_overlap(ia, shared):
    xs = (0.36, 0.64) if shared else (0.24, 0.76)
    for x in xs:
        ia.add_patch(Circle((x, 0.5), 0.22, fill=False, ec=INK, lw=LW))


def ico_cut_strip(ia):
    for i in range(6):
        cut = i == 3
        ia.add_patch(Rectangle((0.04 + i * 0.26, 0.3), 0.2, 0.4, fc="white" if cut else TINT[GREEN],
                               ec=GREEN, lw=LW, linestyle=(0, (2, 1.4)) if cut else "-"))
    x0 = 0.04 + 3 * 0.26
    ia.plot([x0 + 0.02, x0 + 0.18], [0.34, 0.66], color=ORANGE, lw=LW * 1.3)
    ia.plot([x0 + 0.02, x0 + 0.18], [0.66, 0.34], color=ORANGE, lw=LW * 1.3)


def ico_gate(ia):
    for x in (0.12, 0.34):
        ia.add_patch(Rectangle((x, 0.1), 0.12, 0.8, fc=GREEN, ec="none"))
    ia.annotate("", xy=(0.82, 0.5), xytext=(0.5, 0.5),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": LW * 1.2, "mutation_scale": 10})
    ia.add_patch(Rectangle((0.84, 0.34), 0.32, 0.32, fc=GREEN, ec="none"))


def ico_tokens(ia):
    x = 0.0
    for colour, n, written in ((BLUE, 2, ()), (ORANGE, 5, (1, 3))):
        for i in range(n):
            ia.add_patch(Rectangle((x, 0.3), 0.22, 0.4, fc=GREEN if i in written else "white",
                                   ec=colour, lw=LW))
            x += 0.27
        x += 0.12


def unseen_cells(ax):
    """The two unseen test cell lines, dashed as in every figure of the project."""
    ia = icon_axes(ax, [0.5, 1.0, 0.5, 0.24], xlim=(0, 3.4), ylim=(-0.62, 0.9))
    ico_cell(ia, 0.75, 0.42, r=0.38, dashed=True, label="HMEC")
    ico_cell(ia, 2.65, 0.42, r=0.38, dashed=True, label="NHEK")


def titled_die(ax, title, seeds, colour):
    """Panel title preceded by a die whose pips give the number of seeds."""
    ico_die(icon_axes(ax, [0.0, 1.02, 0.14, 0.14]), seeds, colour)
    ax.text(0.17, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5)


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
    ico_leak(icon_axes(ax, [0.86, 1.01, 0.14, 0.2]))

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
    ico_loop(icon_axes(ax, [0.84, 1.01, 0.16, 0.2]))

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
    ico_audit(icon_axes(ax, [0.85, 1.01, 0.15, 0.2]))

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
    ico_flasks(icon_axes(ax, [0.7, 1.01, 0.3, 0.16], xlim=(0, 4.4)),
               chosen=[p["chosen"] for p in pilots].index(True))
    save(fig, out, "fig_audit")


def fig_benchmark(N, R, out):
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    seed_panel(ax, N, "auroc", "Test AUROC")
    panel(ax, "a")
    unseen_cells(ax)
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
    unseen_cells(ax)

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
    ax.set_xlim(-0.5, 1.5)
    style(ax)
    panel(ax, "d")
    ico_overlap(icon_axes(ax, [0.15, 1.01, 0.2, 0.16]), shared=True)
    ico_overlap(icon_axes(ax, [0.65, 1.01, 0.2, 0.16]), shared=False)

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
        (ico_stopwatch if letter == "e" else ico_chip)(icon_axes(ax, [0.83, 1.01, 0.17, 0.2]))
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
    ico_cut_strip(icon_axes(ax, [0.6, 1.02, 0.4, 0.12], xlim=(0, 1.6)))

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
    ico_gate(icon_axes(ax, [0.7, 1.02, 0.3, 0.14], xlim=(0, 1.2)))

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
    ico_tokens(icon_axes(ax, [0.55, 1.02, 0.45, 0.12], xlim=(0, 2.15)))
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
        titled_die(ax, label, N["models"][key]["seeds"], colour)
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
        titled_die(ax, label, N["models"][key]["seeds"], colour)
        ax.text(0.04, 0.96, f"ECE {N['models'][key]['ece']['mean']:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5)
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
    ico_die(icon_axes(axes[0], [0.86, 1.02, 0.14, 0.14]), 3)
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

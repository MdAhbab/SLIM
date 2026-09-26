"""
Write the manuscript's tables as LaTeX fragments, from the saved results.

Every number is recomputed from the per-example predictions in
`eval_results.npz` or read from the audit reports, so a table cannot drift
from the runs behind it. Each fragment is a complete booktabs `tabular` that
the manuscript pulls in with \\input{...}.

    main.tex             every model, mean and spread over its seeds
    ablations.tex        variant KA at seed 0 with one part changed
    baselines.tex        distance-only and locus-prior baselines
    encoder_leakage.tex  label separation through the sequence encoding

Usage:

    python scripts/make_tables.py
    python scripts/make_tables.py --results results --out results/tables

The default output directory is inside results/, never the manuscript folder,
so nothing is overwritten by accident; copy the fragments across on purpose.
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

from src.metrics import compute_all_metrics

MODELS = [("baseline", "Baseline (global attention)"),
          ("ka", "SLIM-KA (spline)"),
          ("ga", "SLIM-GA (gated linear)"),
          ("a", "SLIM-A (rectified)")]

# Variant KA at seed 0 with one thing changed, and what changed.
ABLATIONS = [
    ("ka/seed0", "SLIM-KA, chosen protocol"),
    ("pilot_hic_dna", "Hi-C training rows, DNA branch on"),
    ("pilot_hic_nodna", "Hi-C training rows, DNA branch off"),
    ("pilot_all_dna", "All assays, DNA branch on"),
    ("pilot_all_nodna", "All assays, DNA branch off"),
    ("ka_dna_geometry/seed0", "No chromatin tracks (DNA and pair geometry)"),
    ("ka_old_recipe/seed0", "Original training recipe"),
    ("ka_legacy/seed0", "Original memory rules"),
    ("ka_random_select/seed0", "Random survivors"),
    ("ka_no_readback/seed0", "No memory read-back"),
    ("ka_learned_only/seed0", "Learned score only"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Write LaTeX tables from results.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/tables")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def test_metrics(npz_path, threshold):
    data = np.load(npz_path, allow_pickle=True)
    labels, probs = data["test_labels"], data["test_predictions"]
    m = compute_all_metrics(labels, probs, preds=(probs >= threshold).astype(int))
    out = {k: m.get(k) for k in ("auroc", "aupr", "mcc", "f1")}
    cells = data["test_cell"] if "test_cell" in data.files else None
    if cells is not None:
        for cell in sorted(set(cells.tolist())):
            sel = cells == cell
            out[f"aupr_{cell}"] = compute_all_metrics(labels[sel], probs[sel])["aupr"]
    out["val_aupr"] = float(data["val_aupr"]) if "val_aupr" in data.files else None
    return out


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def mean_sd(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} $\\pm$ {np.std(values, ddof=1):.{digits}f}"


def write(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main_table(results, threshold):
    rows, cells = [], set()
    for key, label in MODELS:
        runs = sorted(glob.glob(os.path.join(results, key, "seed*", "eval_results.npz")))
        if not runs:
            continue
        scores = [test_metrics(r, threshold) for r in runs]
        cells |= {k[5:] for s in scores for k in s if k.startswith("aupr_")}
        rows.append((label, len(runs), scores))
    cells = sorted(cells)
    lines = ["\\begin{tabular}{lc" + "c" * (4 + len(cells)) + "}", "\\toprule",
             "Model & Seeds & AUROC & AUPR & MCC & F1"
             + "".join(f" & AUPR {c}" for c in cells) + " \\\\", "\\midrule"]
    for label, n, scores in rows:
        cols = [mean_sd([s[k] for s in scores]) for k in ("auroc", "aupr", "mcc", "f1")]
        cols += [mean_sd([s.get(f"aupr_{c}") for s in scores]) for c in cells]
        lines.append(f"{label} & {n} & " + " & ".join(cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def ablation_table(results, threshold):
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Configuration & Validation AUPR & Test AUROC & Test AUPR & Test MCC \\\\",
             "\\midrule"]
    found = 0
    for rel, label in ABLATIONS:
        path = os.path.join(results, rel, "eval_results.npz")
        if not os.path.exists(path):
            continue
        s = test_metrics(path, threshold)
        lines.append(f"{label} & {fmt(s['val_aupr'])} & {fmt(s['auroc'])} & "
                     f"{fmt(s['aupr'])} & {fmt(s['mcc'])} \\\\")
        found += 1
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines if found else None


def baseline_table(results):
    path = os.path.join(results, "audit", "dataset_audit.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        audit = json.load(handle)["baselines"]
    names = [("distance_rank", "Distance (closer ranks higher)"),
             ("distance_binned", "Distance, positive rate per distance bin"),
             ("gene_prior_only", "Gene's positive rate in training cells"),
             ("distance_plus_locus_priors", "Distance and gene and enhancer rates")]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Fitted on all assays} & "
             "\\multicolumn{2}{c}{Fitted on Hi-C} \\\\",
             "Baseline & Test AUROC & Test AUPR & Test AUROC & Test AUPR \\\\",
             "\\midrule"]
    for key, label in names:
        cols = []
        for trained_on in ("trained_on_all_assays", "trained_on_hic"):
            m = audit[trained_on]["test"][key]
            cols += [fmt(m["auroc"]), fmt(m["aupr"])]
        lines.append(f"{label} & " + " & ".join(cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def leakage_table(results):
    path = os.path.join(results, "audit", "encoder_leakage.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rules = json.load(handle)["rules"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Encoder fit & Rows & Fitted & Not fitted & Validation \\\\",
             "\\midrule"]
    for rule, label in (("train", "On training rows (original)"),
                        ("crossfit", "Cross-fitted by chromosome half")):
        block = rules.get(rule, {})
        cols = [fmt(block.get(g, {}).get("auroc")) for g in
                ("fitted", "unfitted", "validation")]
        n = block.get("fitted", {}).get("n")
        lines.append(f"{label} & {n if n is not None else '--'} & "
                     + " & ".join(cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    tables = {
        "main.tex": main_table(args.results, args.threshold),
        "ablations.tex": ablation_table(args.results, args.threshold),
        "baselines.tex": baseline_table(args.results),
        "encoder_leakage.tex": leakage_table(args.results),
    }
    for name, lines in tables.items():
        if lines:
            write(os.path.join(args.out, name), lines)
        else:
            print(f"  skipped {name}: its inputs do not exist yet")


if __name__ == "__main__":
    main()

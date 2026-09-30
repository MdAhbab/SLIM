"""
Every number the manuscript, the supplement and the thesis quote, in one file.

The script reads only what the pipeline saved: the per-example predictions in
each run's `eval_results.npz`, each run's `history.json`, and the JSON reports
written by the audit, evaluation, efficiency and write-log scripts. It writes

    results/paper_numbers.json   every quoted statistic, keyed by name
    results/tables/*.tex         booktabs fragments for the manuscript

and prints a readable summary. Nothing is typed in by hand, so a number in the
text can be traced to the file and the key it came from.

Usage:

    python scripts/paper_numbers.py
    python scripts/paper_numbers.py --results results --out results/tables
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_all_metrics  # noqa: E402

MODELS = [("baseline", "Baseline"), ("a", "SLIM-A"), ("ka", "SLIM-KA"), ("ga", "SLIM-GA")]

ABLATIONS = [
    ("ka/seed0", "SLIM-KA, chosen protocol"),
    ("ka_old_recipe/seed0", "Original training recipe"),
    ("ka_legacy/seed0", "Original memory rules"),
    ("ka_random_select/seed0", "Random survivors"),
    ("ka_no_readback/seed0", "No memory read-back"),
    ("ka_learned_only/seed0", "Learned score only"),
    ("ka_dna_geometry/seed0", "DNA and pair geometry, no chromatin"),
]

PILOTS = [
    ("pilot_hic_nodna", "Hi-C", "Chromatin"),
    ("pilot_hic_dna", "Hi-C", "Chromatin and DNA"),
    ("pilot_all_nodna", "All assays", "Chromatin"),
    ("pilot_all_dna", "All assays", "Chromatin and DNA"),
]

METRICS = ("auroc", "aupr", "mcc", "f1", "balanced_accuracy", "precision", "recall")


def parse_args():
    p = argparse.ArgumentParser(description="Recompute every quoted number.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/tables")
    return p.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_key(results, rel):
    """The key evaluate.py and check_overfitting.py use for a run directory."""
    return os.path.join("results", *rel.split("/"))


def test_scores(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels, probs = data["test_labels"], data["test_predictions"]
    m = compute_all_metrics(labels, probs, preds=(probs >= 0.5).astype(int))
    out = {k: m.get(k) for k in METRICS}
    for cell in sorted(set(data["test_cell"].tolist())):
        sel = data["test_cell"] == cell
        cm = compute_all_metrics(labels[sel], probs[sel], preds=(probs[sel] >= 0.5).astype(int))
        out[f"auroc_{cell}"] = cm["auroc"]
        out[f"aupr_{cell}"] = cm["aupr"]
        out[f"mcc_{cell}"] = cm["mcc"]
    vl, vp = data["val_labels"], data["val_predictions"]
    out["val_auroc"] = compute_all_metrics(vl, vp)["auroc"]
    out["val_aupr"] = compute_all_metrics(vl, vp)["aupr"]
    out["val_prevalence"] = float(vl.mean())
    out["test_prevalence"] = float(labels.mean())
    out["val_lift"] = out["val_aupr"] / out["val_prevalence"]
    out["test_lift"] = out["aupr"] / out["test_prevalence"]
    out["params"] = int(data["total_params"])
    out["epochs_run"] = int(data["epochs_run"])
    return out


def history_scores(run_dir, checkpoint_epoch):
    hist = load_json(os.path.join(run_dir, "history.json"))
    row = next(r for r in hist if r["epoch"] == checkpoint_epoch)
    return {
        "epoch_seconds": float(np.mean([r["seconds"] for r in hist])),
        "clean_train_aupr": row.get("train_clean_aupr"),
        "clean_train_auroc": row.get("train_clean_auroc"),
        "clean_gap_aupr": row.get("train_clean_aupr") - row["val_aupr"],
        "clean_gap_auroc": row.get("train_clean_auroc") - row["val_auroc"],
    }


def run_record(results, rel, evaluation, overfitting):
    run_dir = os.path.join(results, *rel.split("/"))
    rec = test_scores(os.path.join(run_dir, "eval_results.npz"))
    key = run_key(results, rel)
    ev = evaluation.get(key)
    if ev:
        sel = ev["at_selected_threshold"]
        rec.update({
            "threshold_selected": ev["selected_threshold"]["threshold"],
            "mcc_selected": sel["mcc"], "f1_selected": sel["f1"],
            "balanced_accuracy_selected": sel["balanced_accuracy"],
            "aupr_ci": [ev["fixed_threshold"]["aupr_ci_lower"], ev["fixed_threshold"]["aupr_ci_upper"]],
            "ece": ev["calibration"]["ece"], "brier": ev["calibration"]["brier"],
            "aupr_disjoint": ev["leakage"]["disjoint"]["aupr"],
            "auroc_disjoint": ev["leakage"]["disjoint"]["auroc"],
            "aupr_reused": ev["leakage"]["reused"]["aupr"],
            "auroc_reused": ev["leakage"]["reused"]["auroc"],
        })
    of = overfitting.get(key)
    if of:
        rec["best_validation_epoch"] = of["best_validation_epoch"]
        rec["checkpoint_epoch"] = of["checkpoint_epoch"]
        rec.update(history_scores(run_dir, of["checkpoint_epoch"]))
    return rec


def summary(records, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        if not vals:
            continue
        out[k] = {"mean": float(np.mean(vals)),
                  "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                  "min": float(np.min(vals)), "max": float(np.max(vals)),
                  "values": [float(v) for v in vals]}
    return out


def paired(a_records, b_records, metric):
    """Paired comparison of two models over the seeds both ran (a minus b)."""
    n = min(len(a_records), len(b_records))
    a = np.array([r[metric] for r in a_records[:n]])
    b = np.array([r[metric] for r in b_records[:n]])
    d = a - b
    out = {"n": int(n), "differences": d.tolist(), "mean": float(d.mean()),
           "sd": float(d.std(ddof=1)) if n > 1 else None,
           "positive": int((d > 0).sum())}
    if n > 1:
        t = stats.ttest_rel(a, b)
        out.update({"t": float(t.statistic), "df": int(n - 1), "p_t": float(t.pvalue)})
        w = stats.wilcoxon(a, b, alternative="two-sided", method="exact")
        out.update({"wilcoxon_w": float(w.statistic), "p_wilcoxon": float(w.pvalue)})
    return out


def ensembles(results, resamples=2000):
    from sklearn.metrics import average_precision_score, roc_auc_score

    def stack(key, seeds):
        probs, labels = [], None
        for s in seeds:
            data = np.load(os.path.join(results, key, f"seed{s}", "eval_results.npz"))
            probs.append(data["test_predictions"])
            labels = data["test_labels"]
        return np.mean(probs, axis=0), labels

    out = {}
    rng = np.random.default_rng(0)
    for name, seeds, keys in (("seeds_0_2", (0, 1, 2), [k for k, _ in MODELS]),
                              ("seeds_0_4", (0, 1, 2, 3, 4), ["baseline", "ka"])):
        preds = {k: stack(k, seeds) for k in keys}
        labels = preds["baseline"][1]
        idx = [rng.integers(0, len(labels), len(labels)) for _ in range(resamples)]
        block = {}
        for k in keys:
            p = preds[k][0]
            block[k] = {"auroc": float(roc_auc_score(labels, p)),
                        "aupr": float(average_precision_score(labels, p))}
            if k != "baseline":
                base = preds["baseline"][0]
                diffs = [average_precision_score(labels[i], p[i])
                         - average_precision_score(labels[i], base[i]) for i in idx]
                block[k]["aupr_minus_baseline"] = block[k]["aupr"] - float(
                    average_precision_score(labels, base))
                block[k]["aupr_minus_baseline_ci"] = [float(np.percentile(diffs, 2.5)),
                                                      float(np.percentile(diffs, 97.5))]
        out[name] = block
    return out


def fmt(v, d=3):
    return "--" if v is None else f"{v:.{d}f}"


def ms(block, d=3):
    if block is None:
        return "--"
    if block.get("sd") is None:
        return f"{block['mean']:.{d}f}"
    return f"{block['mean']:.{d}f} $\\pm$ {block['sd']:.{d}f}"


def write(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    R = args.results
    os.makedirs(args.out, exist_ok=True)
    evaluation = load_json(os.path.join(R, "evaluation.json"))
    overfitting = load_json(os.path.join(R, "overfitting.json"))
    N = {"runs": {}, "models": {}, "paired_vs_baseline": {}, "ablations": {}, "pilots": {}}

    # Main models, every seed.
    per_model = {}
    for key, label in MODELS:
        recs = []
        for seed_dir in sorted(glob.glob(os.path.join(R, key, "seed*"))):
            rel = f"{key}/{os.path.basename(seed_dir)}"
            rec = run_record(R, rel, evaluation, overfitting)
            N["runs"][rel] = rec
            recs.append(rec)
        per_model[key] = recs
        keys = list(METRICS) + ["auroc_HMEC", "aupr_HMEC", "auroc_NHEK", "aupr_NHEK",
                                "mcc_selected", "f1_selected", "balanced_accuracy_selected",
                                "threshold_selected", "ece", "brier", "aupr_disjoint",
                                "aupr_reused", "auroc_disjoint", "val_aupr", "val_auroc",
                                "val_lift", "test_lift", "best_validation_epoch",
                                "epochs_run", "clean_gap_aupr", "clean_gap_auroc",
                                "clean_train_aupr", "epoch_seconds"]
        N["models"][key] = {"label": label, "seeds": len(recs), "params": recs[0]["params"],
                            **summary(recs, keys)}
    # Does validation on the training cell lines rank runs the way the test does?
    all_runs = [r for recs in per_model.values() for r in recs]
    rho = stats.spearmanr([r["val_aupr"] for r in all_runs], [r["aupr"] for r in all_runs])
    N["validation_vs_test"] = {"runs": len(all_runs), "spearman_rho": float(rho.statistic),
                               "p": float(rho.pvalue)}
    for key in ("a", "ka", "ga"):
        N["paired_vs_baseline"][key] = {m: paired(per_model[key], per_model["baseline"], m)
                                        for m in ("auroc", "aupr", "mcc", "mcc_selected",
                                                  "ece", "aupr_disjoint")}
    N["paired_ka_vs_a"] = {m: paired(per_model["ka"], per_model["a"], m) for m in ("auroc", "aupr")}
    N["paired_ga_vs_a"] = {m: paired(per_model["ga"], per_model["a"], m) for m in ("auroc", "aupr")}
    ka_sd = N["models"]["ka"]["aupr"]["sd"]

    # Seed ensembles: the mean prediction over the seeds every model ran (0-2),
    # and over all five seeds where two models have them. The same rule is
    # applied to every model; intervals are a paired bootstrap over test pairs.
    N["ensembles"] = ensembles(R)

    # Ablations: seed-0 runs, compared with KA seed 0 and with the KA seed spread.
    ref = N["runs"]["ka/seed0"]
    for rel, label in ABLATIONS:
        if not os.path.exists(os.path.join(R, *rel.split("/"), "eval_results.npz")):
            continue
        rec = N["runs"].get(rel) or run_record(R, rel, evaluation, overfitting)
        rec["label"] = label
        rec["delta_aupr_vs_ka0"] = rec["aupr"] - ref["aupr"]
        rec["delta_in_ka_seed_sd"] = rec["delta_aupr_vs_ka0"] / ka_sd
        rec["within_ka_seed_range"] = (N["models"]["ka"]["aupr"]["min"] <= rec["aupr"]
                                       <= N["models"]["ka"]["aupr"]["max"])
        N["ablations"][rel] = rec

    # Protocol pilots (chosen on validation AUPR; test shown for the record only).
    protocol = load_json(os.path.join(R, "protocol.json"))
    for rel, assays, inputs in PILOTS:
        rec = run_record(R, rel, evaluation, overfitting)
        rec.update({"assays": assays, "inputs": inputs, "chosen": rel == protocol["chosen"]})
        N["pilots"][rel] = rec
    N["protocol"] = {k: protocol[k] for k in ("train_assays", "modalities", "chosen", "metric")}

    # The runs before the pipeline fixes, for the before-and-after comparison.
    old = {}
    arch = os.path.join(R, "archive", "before_pipeline_fixes")
    for key, label in MODELS:
        dirs = sorted(glob.glob(os.path.join(arch, key, "seed*")))
        recs = [test_scores(os.path.join(d, "eval_results.npz")) for d in dirs]
        if recs:
            old[key] = {"label": label, "seeds": len(recs), **summary(recs, ["auroc", "aupr", "mcc"])}
    N["before_pipeline_fixes"] = old

    # Audits, overlap, efficiency and memory write logs, copied by key.
    audit = load_json(os.path.join(R, "audit", "dataset_audit.json"))
    N["dataset_audit"] = {k: audit[k] for k in ("training_repeats", "test_repeats",
                                                "after_union_dedup", "anchors",
                                                "test_overlap_with_training", "files")}
    N["dataset_audit"]["hic_distance_quantiles_kb"] = {
        a: audit["positives_by_assay"][a]["distance"]
        for a in audit["positives_by_assay"] if a in ("HiC", "RNAPII-ChIAPET", "CTCF-ChIAPET")}
    N["baselines"] = {fit: {split: audit["baselines"][fit][split]
                            for split in ("validation_hic", "test", "test_HMEC", "test_NHEK")}
                      for fit in ("trained_on_all_assays", "trained_on_hic")}
    N["encoder_leakage"] = load_json(os.path.join(R, "audit", "encoder_leakage.json"))
    N["overlap"] = {k: v for k, v in load_json(os.path.join(R, "leakage", "leakage_report.json")).items()
                    if k != "by_chromosome"}
    N["efficiency"] = load_json(os.path.join(R, "efficiency.json"))
    N["writelog"] = {}
    for rel in ("ka/seed0", "ga/seed0", "ka_legacy/seed0"):
        path = os.path.join(R, *rel.split("/"), "writelog.json")
        if os.path.exists(path):
            w = load_json(path)
            N["writelog"][rel] = {k: w[k] for k in w if k not in ("token_histogram",)}

    with open(os.path.join(R, "paper_numbers.json"), "w", encoding="utf-8") as handle:
        json.dump(N, handle, indent=1)

    # ---- LaTeX fragments ----
    M = N["models"]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Model & Seeds & AUROC & AUPR & MCC & MCC (val.\\ thr.) & ECE \\\\", "\\midrule"]
    for key, _ in MODELS:
        m = M[key]
        lines.append(f"{m['label']} & {m['seeds']} & {ms(m['auroc'])} & {ms(m['aupr'])} & "
                     f"{ms(m['mcc'])} & {ms(m['mcc_selected'])} & {ms(m['ece'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "main.tex"), lines)

    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{HMEC} & \\multicolumn{2}{c}{NHEK} & "
             "\\multicolumn{2}{c}{No shared locus} \\\\",
             "Model & AUROC & AUPR & AUROC & AUPR & AUROC & AUPR \\\\", "\\midrule"]
    for key, _ in MODELS:
        m = M[key]
        lines.append(f"{m['label']} & {ms(m['auroc_HMEC'])} & {ms(m['aupr_HMEC'])} & "
                     f"{ms(m['auroc_NHEK'])} & {ms(m['aupr_NHEK'])} & "
                     f"{ms(m['auroc_disjoint'])} & {ms(m['aupr_disjoint'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "percell.tex"), lines)

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Configuration (seed 0) & Val.\\ AUPR & Test AUROC & Test AUPR & $\\Delta$ AUPR \\\\",
             "\\midrule"]
    for rel, rec in N["ablations"].items():
        delta = "--" if rel == "ka/seed0" else f"{rec['delta_aupr_vs_ka0']:+.3f}"
        lines.append(f"{rec['label']} & {fmt(rec['val_aupr'])} & {fmt(rec['auroc'])} & "
                     f"{fmt(rec['aupr'])} & {delta} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "ablations.tex"), lines)

    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Training rows & Inputs & Val.\\ AUROC & Val.\\ AUPR & Clean gap & Test AUPR \\\\",
             "\\midrule"]
    for rel, rec in N["pilots"].items():
        mark = "$^{\\ast}$" if rec["chosen"] else ""
        lines.append(f"{rec['assays']} & {rec['inputs']}{mark} & {fmt(rec['val_auroc'])} & "
                     f"{fmt(rec['val_aupr'])} & {fmt(rec['clean_gap_aupr'])} & {fmt(rec['aupr'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "pilots.tex"), lines)

    names = [("distance_rank", "Distance only"),
             ("gene_prior_only", "Gene positive rate"),
             ("distance_plus_locus_priors", "Distance and locus rates")]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Fitted on all assays} & \\multicolumn{2}{c}{Fitted on Hi-C} \\\\",
             "Reference predictor & Test AUROC & Test AUPR & Test AUROC & Test AUPR \\\\", "\\midrule"]
    for key, label in names:
        cols = []
        for fit in ("trained_on_all_assays", "trained_on_hic"):
            m = N["baselines"][fit]["test"][key]
            cols += [fmt(m["auroc"]), fmt(m["aupr"])]
        lines.append(f"{label} & " + " & ".join(cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "baselines.tex"), lines)

    rules = N["encoder_leakage"]["rules"]
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Encoder fitted on & Fitted rows & Other training rows & Validation rows \\\\",
             "\\midrule"]
    for rule, label in (("train", "Training rows (original)"),
                        ("crossfit", "Other chromosome half")):
        b = rules[rule]
        lines.append(f"{label} & {fmt(b['fitted']['auroc'])} & {fmt(b['unfitted']['auroc'])} & "
                     f"{fmt(b['validation']['auroc'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "encoder_leakage.tex"), lines)

    eff = N["efficiency"]["models"]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Model & Parameters & Latency (ms) & Pairs/s & Inference MiB & Training MiB \\\\",
             "\\midrule"]
    for key, label in (("baseline", "Baseline"), ("A", "SLIM-A"), ("KA", "SLIM-KA"), ("GA", "SLIM-GA")):
        e = eff[key]
        lines.append(f"{label} & {e['parameters']:,} & {e['inference_ms_median']:.1f} & "
                     f"{e['inference_pairs_per_second']:.0f} & {e['peak_memory_mib_inference']:.0f} & "
                     f"{e['peak_memory_mib_training']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "efficiency.tex"), lines)

    # Every run, one row per seed.
    lines = ["\\begin{tabular}{llccccccc}", "\\toprule",
             "Model & Seed & Epochs & Val.\\ AUPR & AUROC & AUPR & AUPR 95\\% CI & MCC & ECE \\\\",
             "\\midrule"]
    for key, label in MODELS:
        for rel in sorted(r for r in N["runs"] if r.startswith(key + "/")):
            r = N["runs"][rel]
            lo, hi = r["aupr_ci"]
            lines.append(f"{label} & {rel.split('seed')[-1]} & {r['checkpoint_epoch']}/{r['epochs_run']} & "
                         f"{fmt(r['val_aupr'])} & {fmt(r['auroc'])} & {fmt(r['aupr'])} & "
                         f"{lo:.3f}--{hi:.3f} & {fmt(r['mcc'])} & {fmt(r['ece'])} \\\\")
        lines.append("\\midrule" if key != MODELS[-1][0] else "")
    lines = [l for l in lines if l] + ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "perseed.tex"), lines)

    # Operating points and calibration, mean and SD over seeds.
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Model & F1 (0.5) & Bal.\\ acc.\\ (0.5) & Val.\\ threshold & MCC (val.\\ thr.) "
             "& ECE & Brier \\\\", "\\midrule"]
    for key, _ in MODELS:
        m = M[key]
        lines.append(f"{m['label']} & {ms(m['f1'])} & {ms(m['balanced_accuracy'])} & "
                     f"{ms(m['threshold_selected'], 2)} & {ms(m['mcc_selected'])} & "
                     f"{ms(m['ece'])} & {ms(m['brier'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "operating.tex"), lines)

    # Overfitting on the training cell lines, at the saved checkpoint.
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Model & Best epoch & Clean train AUPR & Val.\\ AUPR & Gap (AUPR) & Gap (AUROC) \\\\",
             "\\midrule"]
    for key, _ in MODELS:
        m = M[key]
        lines.append(f"{m['label']} & {ms(m['best_validation_epoch'], 1)} & {ms(m['clean_train_aupr'])} & "
                     f"{ms(m['val_aupr'])} & {ms(m['clean_gap_aupr'])} & {ms(m['clean_gap_auroc'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "overfitting.tex"), lines)

    # Seed ensembles.
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Seeds & Model & AUROC & AUPR & AUPR minus baseline (95\\% CI) \\\\", "\\midrule"]
    names = dict((k, l) for k, l in MODELS)
    for block, seeds in (("seeds_0_2", "0--2"), ("seeds_0_4", "0--4")):
        for k, e in N["ensembles"][block].items():
            diff = "--"
            if "aupr_minus_baseline" in e:
                lo, hi = e["aupr_minus_baseline_ci"]
                diff = f"{e['aupr_minus_baseline']:+.3f} ({lo:+.3f} to {hi:+.3f})"
            lines.append(f"{seeds} & {names[k]} & {e['auroc']:.3f} & {e['aupr']:.3f} & {diff} \\\\")
        if block == "seeds_0_2":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "ensembles.tex"), lines)

    # Before and after the pipeline fixes.
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{Original pipeline} & \\multicolumn{3}{c}{Corrected pipeline} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Model & Seeds & AUROC & AUPR & Seeds & AUROC & AUPR \\\\", "\\midrule"]
    for key, label in MODELS:
        o, m = old.get(key), M[key]
        if not o:
            continue
        lines.append(f"{label} & {o['seeds']} & {ms(o['auroc'])} & {ms(o['aupr'])} & "
                     f"{m['seeds']} & {ms(m['auroc'])} & {ms(m['aupr'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "before_after.tex"), lines)

    # Memory write log.
    tracks = ["H3K27ac", "H3K4me3", "DNase", "CTCF", "H3K27me3"]
    lines = ["\\begin{tabular}{l" + "c" * len(tracks) + "ccc}", "\\toprule",
             "Model (seed 0) & " + " & ".join(tracks) + " & Positions written & Jaccard ratio & Top position \\\\",
             "\\midrule"]
    for rel, label in (("ka/seed0", "SLIM-KA"), ("ga/seed0", "SLIM-GA"), ("ka_legacy/seed0", "SLIM-KA, original rules")):
        w = N["writelog"].get(rel)
        if not w:
            continue
        c = w["collapse"]
        lines.append(f"{label} & " + " & ".join(f"{w['tracks'][t]['ratio']:.2f}" for t in tracks)
                     + f" & {c['positions_ever_written']} & {c['jaccard_ratio']:.1f} & "
                     f"{100 * c['max_position_frequency']:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "writelog.tex"), lines)

    # Genomic overlap between test pairs and Hi-C training pairs.
    ov = N["overlap"]
    n = ov["n_test_pairs"]
    rows = [("Enhancer locus in training", ov["enhancer_reused"]),
            ("Promoter locus in training", ov["promoter_reused"]),
            ("Either locus in training", ov["either_reused"]),
            ("Both loci in training", ov["both_reused"]),
            ("Identical pair in a training cell line", ov["pair_reused"]),
            ("No shared locus", ov["disjoint"])]
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Test pairs & Count & Percent \\\\", "\\midrule"]
    lines += [f"{label} & {count:,} & {100 * count / n:.1f} \\\\" for label, count in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "overlap.tex"), lines)

    # BENGI files in the training cell lines.
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "File & Rows & Positives & Positive rate \\\\", "\\midrule"]
    for f in N["dataset_audit"]["files"]:
        name = f["file"].replace(".tsv.gz", "").replace("-Benchmark.v3", "").replace("_", "\\_")
        lines.append(f"{name} & {f['rows']:,} & {f['positives']:,} & {f['positive_rate']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(args.out, "files.tex"), lines)

    # ---- readable summary ----
    print("Model summary (mean, SD over seeds)")
    for key, _ in MODELS:
        m = M[key]
        print(f"  {m['label']:9s} n={m['seeds']}  AUROC {ms(m['auroc'],4)}  AUPR {ms(m['aupr'],4)}  "
              f"MCC {ms(m['mcc'],4)}  MCCsel {ms(m['mcc_selected'],4)}  ECE {ms(m['ece'],4)}  "
              f"disjoint AUPR {ms(m['aupr_disjoint'],4)}")
    print("Paired against the baseline (model minus baseline)")
    for key in ("a", "ka", "ga"):
        for met, p in N["paired_vs_baseline"][key].items():
            print(f"  {key:3s} {met:13s} n={p['n']} mean {p['mean']:+.4f} sd {p['sd']:.4f} "
                  f"t={p.get('t', float('nan')):.2f} p_t={p.get('p_t', float('nan')):.3f} "
                  f"p_w={p.get('p_wilcoxon', float('nan')):.3f} positive {p['positive']}/{p['n']}")
    print("Ablations (seed 0), delta AUPR against KA seed 0, KA seed SD", f"{ka_sd:.4f}")
    for rel, rec in N["ablations"].items():
        print(f"  {rec['label']:40s} val {rec['val_aupr']:.4f} test AUROC {rec['auroc']:.4f} "
              f"AUPR {rec['aupr']:.4f} d {rec['delta_aupr_vs_ka0']:+.4f} "
              f"({rec['delta_in_ka_seed_sd']:+.2f} SD) in range {rec['within_ka_seed_range']}")
    print("Seed ensembles (mean prediction), with the paired-bootstrap AUPR difference to the baseline")
    for name, block in N["ensembles"].items():
        for k, e in block.items():
            extra = ""
            if "aupr_minus_baseline" in e:
                lo, hi = e["aupr_minus_baseline_ci"]
                extra = f"  diff {e['aupr_minus_baseline']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(f"  {name} {k:9s} AUROC {e['auroc']:.4f} AUPR {e['aupr']:.4f}{extra}")
    print("Before the pipeline fixes (test, mean over seeds)")
    for key, m in old.items():
        print(f"  {m['label']:9s} n={m['seeds']} AUROC {ms(m['auroc'],4)} AUPR {ms(m['aupr'],4)}")
    print(f"Written to {os.path.join(R, 'paper_numbers.json')} and {args.out}")


if __name__ == "__main__":
    main()

"""
Score the two extra test sets from saved predictions (no model is run).

Run after scripts/data/build_extra_tests.py (which writes the test files and
training_pairs.pkl) and scripts/evaluate_external.py (which scores every run
on them and saves <external>/<test>/<tag>.npz). Predictions are matched to the
test rows by chromosome, enhancer midpoint and transcript start site, since
the scoring pipeline may merge repeated rows.

  eqtl      Per study: the candidate (enhancer, gene) pairs of enhancers that
            hold a fine-mapped variant, a gene scored by its best transcript.
            AUROC and AUPR over all pairs, and top-1 accuracy, the share of
            enhancers whose highest-scored gene is an eGene. The baseline is
            distance: the nearest gene ranks first. Also on the pairs never
            seen in training (top-1 over enhancers with no seen pair).
  javierre  Per cell: AUROC and AUPR on the unseen capture Hi-C pairs, with
            the distance baseline on the same rows.

Runs are named by evaluate_external.py as <variant folder>_seed<N>; seeds of
one variant are averaged (mean and standard deviation).

Usage:
    python scripts/eval_extra.py --external results/final/external \
        --data datasets/slim_data --out results/final/extra_eval.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bengi import open_text, parse_line  # noqa: E402


def ranking(labels, scores) -> Dict[str, float]:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return {"auroc": None, "aupr": None}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores))}


def read_test(path: Path) -> Dict[tuple, List[Dict]]:
    """Test rows keyed as the scoring pipeline keys them."""
    rows = defaultdict(list)
    with open_text(str(path)) as handle:
        for line in handle:
            row = parse_line(line)
            if row is not None:
                rows[(row["chrom"], row["enh_coord"], row["tss_coord"])].append(row)
    return rows


def predictions(npz_path: Path) -> Dict[tuple, float]:
    saved = np.load(npz_path, allow_pickle=True)
    keys = zip(saved["chrom"].tolist(), saved["enh_coord"].tolist(), saved["prom_coord"].tolist())
    return {(c, int(e), int(p)): float(s) for (c, e, p), s in zip(keys, saved["predictions"])}


def gene_table(rows, scores, seen) -> Dict[tuple, Dict]:
    """(enhancer, gene) -> best score, label, nearest distance, seen in training."""
    table = {}
    for key, group in rows.items():
        score = scores.get(key)
        if score is None:
            raise KeyError(f"no prediction for test row {key}")
        for r in group:
            pair = (r["enh_id"], r["gene"])
            entry = table.setdefault(pair, {"score": -np.inf, "label": 0, "dist": np.inf,
                                            "seen": pair in seen})
            entry["score"] = max(entry["score"], score)
            entry["label"] = max(entry["label"], r["label"])
            entry["dist"] = min(entry["dist"], abs(r["dist"]))
    return table


def top1(pairs: Dict[tuple, Dict], enhancers) -> Dict[str, float]:
    """Share of enhancers whose best-scored (model) or nearest (distance) gene
    is an eGene."""
    by_enh = defaultdict(list)
    for (enh, _gene), entry in pairs.items():
        if enh in enhancers:
            by_enh[enh].append(entry)
    model = [max(v, key=lambda e: e["score"])["label"] for v in by_enh.values()]
    nearest = [min(v, key=lambda e: e["dist"])["label"] for v in by_enh.values()]
    n = len(by_enh)
    return {"enhancers": n, "model": float(np.mean(model)) if n else None,
            "nearest_gene": float(np.mean(nearest)) if n else None}


def eqtl_metrics(pairs: Dict[tuple, Dict]) -> Dict:
    entries = list(pairs.values())
    labels = [e["label"] for e in entries]
    out = {"pairs": len(entries), "positives": int(sum(labels)),
           "model": ranking(labels, [e["score"] for e in entries]),
           "distance": ranking(labels, [-e["dist"] for e in entries]),
           "top1": top1(pairs, {enh for enh, _ in pairs})}
    unseen = {k: e for k, e in pairs.items() if not e["seen"]}
    u_labels = [e["label"] for e in unseen.values()]
    clean = {enh for enh, _ in pairs} - {enh for (enh, _), e in pairs.items() if e["seen"]}
    out["unseen"] = {"pairs": len(unseen), "positives": int(sum(u_labels)),
                     "model": ranking(u_labels, [e["score"] for e in unseen.values()]) if unseen else {},
                     "distance": ranking(u_labels, [-e["dist"] for e in unseen.values()]) if unseen else {},
                     "top1": top1(pairs, clean)}
    return out


def javierre_metrics(rows, scores) -> Dict:
    labels, model, dist = [], [], []
    for key, group in rows.items():
        score = scores.get(key)
        if score is None:
            raise KeyError(f"no prediction for test row {key}")
        for r in group:
            labels.append(r["label"])
            model.append(score)
            dist.append(-abs(r["dist"]))
    return {"rows": len(labels), "positives": int(sum(labels)),
            "model": ranking(labels, model), "distance": ranking(labels, dist)}


def flatten(d, prefix=""):
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flatten(v, name + ".")
        elif isinstance(v, (int, float)) and v is not None:
            yield name, v


def summarise(per_run: Dict[str, Dict]) -> Dict[str, Dict]:
    """Mean and standard deviation over the seeds of each variant."""
    groups = defaultdict(list)
    for tag, result in per_run.items():
        groups[tag.rsplit("_seed", 1)[0]].append(dict(flatten(result)))
    out = {}
    for variant, results in groups.items():
        keys = sorted(set().union(*results))
        out[variant] = {"seeds": len(results)}
        for k in keys:
            vals = [r[k] for r in results if k in r and r[k] is not None]
            if vals:
                out[variant][k] = [float(np.mean(vals)), float(np.std(vals))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--external", required=True, help="evaluate_external.py --out folder")
    parser.add_argument("--data", default="datasets/slim_data")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data, external = Path(args.data), Path(args.external)
    with open(data / "eval" / "training_pairs.pkl", "rb") as handle:
        seen = pickle.load(handle)["pairs"]
    built = json.loads((data / "eval" / "extra_tests_report.json").read_text())
    result = {"eqtl": {}, "javierre": {}}

    for study, info in sorted(built.get("eqtl", {}).get("studies", {}).items()):
        if not info.get("file"):
            continue
        test = data / "eval" / "eqtl_pairs" / info["file"]
        name = info["file"].replace("-Benchmark.v3.tsv.gz", "")
        runs = sorted((external / name).glob("*.npz"))
        if not runs:
            print(f"  {name}: not scored yet, skipped")
            continue
        rows = read_test(test)
        per_run = {p.stem: eqtl_metrics(gene_table(rows, predictions(p), seen)) for p in runs}
        result["eqtl"][study] = {k: info[k] for k in ("description", "cell", "match")}
        result["eqtl"][study].update(runs=per_run, mean=summarise(per_run))

    for cell, info in sorted(built.get("javierre", {}).items()):
        test = data / "eval" / "javierre_unseen" / info["file"]
        name = info["file"].replace("-Benchmark.v3.tsv.gz", "")
        runs = sorted((external / name).glob("*.npz"))
        if not runs:
            print(f"  {name}: not scored yet, skipped")
            continue
        rows = read_test(test)
        per_run = {p.stem: javierre_metrics(rows, predictions(p)) for p in runs}
        result["javierre"][cell] = {"runs": per_run, "mean": summarise(per_run)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))

    def fmt(m, key):
        v = m.get(key)
        return f"{v[0]:.3f}" if v else "  -  "

    print(f"\n{'eQTL study':30s} {'cell':14s} {'variant':10s} AUROC  dist   AUPR   dist   "
          f"top-1  nearest  (unseen AUPR / dist)")
    for study, r in result["eqtl"].items():
        for variant, m in r["mean"].items():
            print(f"{r['description'][:30]:30s} {r['cell']:14s} {variant:10s} "
                  f"{fmt(m, 'model.auroc')}  {fmt(m, 'distance.auroc')}  {fmt(m, 'model.aupr')}  "
                  f"{fmt(m, 'distance.aupr')}  {fmt(m, 'top1.model')}  {fmt(m, 'top1.nearest_gene')}"
                  f"    ({fmt(m, 'unseen.model.aupr')} / {fmt(m, 'unseen.distance.aupr')})")
    print(f"\n{'Javierre (unseen pairs)':30s} {'variant':10s} AUROC  dist   AUPR   dist")
    for cell, r in result["javierre"].items():
        for variant, m in r["mean"].items():
            print(f"{cell:30s} {variant:10s} {fmt(m, 'model.auroc')}  {fmt(m, 'distance.auroc')}  "
                  f"{fmt(m, 'model.aupr')}  {fmt(m, 'distance.aupr')}")
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()

"""
Audit the BENGI pairs themselves, before any model sees them.

Needs only the BENGI files, no chromatin tracks, no genome and no GPU. It
answers four questions that decide how every model result should be read.

  1. Repeats. GM12878 and HeLa each come in three assay files. How many rows
     describe a pair already present in another file, and how often do the
     copies disagree on the label?
  2. Anchor reuse. How many pairs does a typical promoter or enhancer take
     part in? A locus that recurs can be memorised.
  3. Distance. How far apart are interacting and non-interacting pairs, and
     how well does genomic distance alone rank the test pairs?
  4. Locus priors. How well does a model that knows only which genes and
     enhancers interacted in the training cell lines do on the test cell
     lines? This is the most that memorising loci could contribute.

The baselines are fitted twice, once on every training assay and once on the
Hi-C rows alone, because the test cell lines have Hi-C labels only and the
ChIA-PET assays define interaction at a much shorter range. Both are scored
on the same rows: validation Hi-C pairs (held-out chromosomes of the training
cell lines) and the test cell lines.

Usage:

    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --bengi-dir path/to/BENGI --out results/audit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bengi import bengi_files, deduplicate, read_rows
from src.config import load_config

DEFAULT_TRAIN_CELLS = ["GM12878", "HeLa", "K562", "IMR90"]
DEFAULT_TEST_CELLS = ["HMEC", "NHEK"]


def parse_args():
    p = argparse.ArgumentParser(description="Audit the BENGI pairs.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--bengi-dir", default=None,
                   help="overrides paths.bengi_dir from the config")
    p.add_argument("--train-cells", nargs="+", default=DEFAULT_TRAIN_CELLS)
    p.add_argument("--test-cells", nargs="+", default=DEFAULT_TEST_CELLS)
    p.add_argument("--out", default="results/audit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def ranking(labels, scores):
    """AUROC and AUPR of a score, plus the positive rate for reading AUPR."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        return {"n": int(len(labels)), "positive_rate": float(labels.mean()),
                "auroc": None, "aupr": None}
    return {"n": int(len(labels)),
            "positive_rate": round(float(labels.mean()), 4),
            "auroc": round(float(roc_auc_score(labels, scores)), 4),
            "aupr": round(float(average_precision_score(labels, scores)), 4)}


def log_distance(rows):
    return np.log10(np.maximum(np.array([r["dist"] for r in rows]), 1.0))


class DistancePrior:
    """Positive rate per distance bin, learned from the training rows.

    Quantile bins on log distance, with a small pseudo-count toward the
    overall rate. This is the best any model can do from distance alone,
    without assuming the relation is monotone.
    """

    def __init__(self, n_bins=30, strength=5.0):
        self.n_bins = n_bins
        self.strength = strength

    def fit(self, rows):
        x = log_distance(rows)
        y = np.array([r["label"] for r in rows])
        self.edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        bins = np.clip(np.searchsorted(self.edges, x, side="right") - 1,
                       0, len(self.edges) - 2)
        base = y.mean()
        self.rate = np.array([
            (y[bins == b].sum() + self.strength * base)
            / ((bins == b).sum() + self.strength)
            for b in range(len(self.edges) - 1)])
        return self

    def score(self, rows):
        x = log_distance(rows)
        bins = np.clip(np.searchsorted(self.edges, x, side="right") - 1,
                       0, len(self.edges) - 2)
        return self.rate[bins]


class LocusPrior:
    """Smoothed positive rate of a gene or an enhancer in the training rows.

    For a training row the rate is taken from the OTHER training cell lines
    only, so it measures what carries over between cell lines rather than
    reproducing the row's own label.
    """

    def __init__(self, key, strength=5.0):
        self.key = key
        self.strength = strength

    def fit(self, rows):
        self.total = defaultdict(lambda: [0, 0])
        self.by_cell = defaultdict(lambda: [0, 0])
        for r in rows:
            k = r[self.key]
            self.total[k][0] += r["label"]
            self.total[k][1] += 1
            self.by_cell[(r["cell"], k)][0] += r["label"]
            self.by_cell[(r["cell"], k)][1] += 1
        self.base = float(np.mean([r["label"] for r in rows]))
        return self

    def _rate(self, pos, n):
        return (pos + self.strength * self.base) / (n + self.strength)

    def score(self, rows, exclude_own_cell=False):
        out = np.empty(len(rows))
        for i, r in enumerate(rows):
            pos, n = self.total.get(r[self.key], (0, 0))
            if exclude_own_cell:
                own_pos, own_n = self.by_cell.get((r["cell"], r[self.key]), (0, 0))
                pos, n = pos - own_pos, n - own_n
            out[i] = self._rate(pos, n)
        return out

    def seen(self, rows):
        return np.array([r[self.key] in self.total for r in rows])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feature_matrix(rows, gene_prior, enh_prior, exclude_own_cell):
    d = log_distance(rows)
    return np.column_stack([
        d, d ** 2,
        logit(gene_prior.score(rows, exclude_own_cell)),
        logit(enh_prior.score(rows, exclude_own_cell)),
    ])


# ---------------------------------------------------------------------------
# Sections of the report
# ---------------------------------------------------------------------------


def file_table(paths):
    table = []
    for path in paths:
        rows = read_rows([path])
        pos = sum(r["label"] for r in rows)
        table.append({"file": os.path.basename(path), "rows": len(rows),
                      "positives": pos,
                      "positive_rate": round(pos / max(len(rows), 1), 4)})
    return table


def repeat_report(rows):
    """Exact repeats across assay files, and looser repeats of one pair."""
    _, exact = deduplicate(rows, "none")
    # The same enhancer and gene in one cell line, through different transcripts.
    by_gene = defaultdict(list)
    for r in rows:
        by_gene[(r["cell"], r["enh_id"], r["gene"])].append(r["label"])
    # The same pair (enhancer and transcript start) in more than one cell line.
    by_locus = defaultdict(dict)
    for r in rows:
        by_locus[(r["chrom"], r["enh_coord"], r["tss_coord"])][r["cell"]] = r["label"]
    multi_cell = [cells for cells in by_locus.values() if len(cells) > 1]
    per_cell_assays = Counter()
    for r in rows:
        per_cell_assays[(r["cell"], r["assay"])] += 1
    return {
        "exact_repeats_across_assays": exact,
        "enhancer_gene_groups": {
            "groups": len(by_gene),
            "rows": len(rows),
            "rows_per_group_mean": round(len(rows) / max(len(by_gene), 1), 3),
            "groups_with_conflicting_labels": sum(
                1 for v in by_gene.values() if len(set(v)) > 1),
        },
        "pairs_in_several_cell_lines": {
            "distinct_pairs": len(by_locus),
            "pairs_in_two_or_more_cells": len(multi_cell),
            "of_those_label_differs_between_cells": sum(
                1 for c in multi_cell if len(set(c.values())) > 1),
        },
        "rows_by_cell_and_assay": {f"{c}/{a}": n
                                   for (c, a), n in sorted(per_cell_assays.items())},
    }


def anchor_report(rows):
    """How many pairs each promoter and each enhancer takes part in."""
    def degree(key):
        counts = Counter(key(r) for r in rows)
        per_row = np.array([counts[key(r)] for r in rows])
        values = np.array(list(counts.values()))
        return {"distinct": len(counts),
                "pairs_per_anchor_median": float(np.median(values)),
                "pairs_per_anchor_mean": round(float(values.mean()), 2),
                "pairs_per_anchor_max": int(values.max()),
                "rows_whose_anchor_has_10_or_more_pairs": round(
                    float((per_row >= 10).mean()), 4)}
    return {"promoter": degree(lambda r: (r["cell"], r["chrom"], r["tss_coord"])),
            "enhancer": degree(lambda r: (r["cell"], r["chrom"], r["enh_coord"])),
            "gene_any_cell": degree(lambda r: r["gene"])}


def chromosome_table(rows, valid_chroms):
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[r["chrom"]][0] += 1
        counts[r["chrom"]][1] += r["label"]

    def order(c):
        tail = c.replace("chr", "")
        return (0, int(tail)) if tail.isdigit() else (1, tail)
    return [{"chrom": c, "rows": n, "positives": p,
             "positive_rate": round(p / max(n, 1), 4),
             "validation": c in valid_chroms}
            for c, (n, p) in sorted(counts.items(), key=lambda kv: order(kv[0]))]


def assay_table(rows):
    """Rows, positive rate and positive distances for every assay combination."""
    by_assay = defaultdict(list)
    for r in rows:
        by_assay[r["assay"]].append(r)
    table = {}
    for assay, group in sorted(by_assay.items()):
        y = np.array([r["label"] for r in group])
        table[assay] = {"rows": len(group),
                        "positive_rate": round(float(y.mean()), 4),
                        "distance": distance_summary(group)}
    return table


def distance_summary(rows):
    d = np.array([r["dist"] for r in rows])
    y = np.array([r["label"] for r in rows])
    out = {}
    for name, mask in (("positive", y == 1), ("negative", y == 0)):
        if mask.any():
            q = np.quantile(d[mask], [0.1, 0.25, 0.5, 0.75, 0.9])
            out[name] = {"n": int(mask.sum()),
                         "quantiles_kb": [round(float(v) / 1000, 1) for v in q]}
    return out


def baselines(train_rows, eval_sets):
    """Distance-only and locus-prior baselines, fitted on the training split."""
    from sklearn.linear_model import LogisticRegression

    dist_prior = DistancePrior().fit(train_rows)
    gene_prior = LocusPrior("gene").fit(train_rows)
    enh_prior = LocusPrior("enh_id").fit(train_rows)

    y_train = np.array([r["label"] for r in train_rows])
    d_train = log_distance(train_rows)
    dist_lr = LogisticRegression(max_iter=1000).fit(
        np.column_stack([d_train, d_train ** 2]), y_train)
    # Locus priors for training rows come from the other cell lines, so this
    # fit learns how much carries over between cell lines.
    full_lr = LogisticRegression(max_iter=1000).fit(
        feature_matrix(train_rows, gene_prior, enh_prior, True), y_train)

    report = {}
    for name, rows in eval_sets.items():
        if not rows:
            continue
        y = np.array([r["label"] for r in rows])
        d = log_distance(rows)
        seen_gene = gene_prior.seen(rows)
        report[name] = {
            "distance_rank": ranking(y, -d),
            "distance_logistic": ranking(
                y, dist_lr.predict_proba(np.column_stack([d, d ** 2]))[:, 1]),
            "distance_binned": ranking(y, dist_prior.score(rows)),
            "gene_prior_only": ranking(y, gene_prior.score(rows)),
            "enhancer_prior_only": ranking(y, enh_prior.score(rows)),
            "distance_plus_locus_priors": ranking(y, full_lr.predict_proba(
                feature_matrix(rows, gene_prior, enh_prior, False))[:, 1]),
            "fraction_with_gene_seen_in_training": round(float(seen_gene.mean()), 4),
            "distance": distance_summary(rows),
        }
    return report


def overlap_with_training(train_rows, test_rows):
    """Test pairs whose enhancer and gene were paired in a training cell line."""
    seen = defaultdict(set)
    for r in train_rows:
        seen[(r["enh_id"], r["gene"])].add(r["label"])
    hits = [r for r in test_rows if (r["enh_id"], r["gene"]) in seen]
    agree = sum(1 for r in hits if seen[(r["enh_id"], r["gene"])] == {r["label"]})
    return {"test_rows": len(test_rows),
            "enhancer_gene_pair_seen_in_training": len(hits),
            "fraction": round(len(hits) / max(len(test_rows), 1), 4),
            "of_those_same_label_in_every_training_occurrence": agree}


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args.config)
    bengi_dir = args.bengi_dir or config["paths"]["bengi_dir"]
    valid_chroms = set(config["training"].get("valid_chroms", ["chr11", "chr17"]))

    train_paths = bengi_files(bengi_dir, args.train_cells)
    test_paths = bengi_files(bengi_dir, args.test_cells)
    if not train_paths or not test_paths:
        raise SystemExit(f"no BENGI files for the requested cells in {bengi_dir}")

    raw_train = read_rows(train_paths)
    raw_test = read_rows(test_paths)
    train_pool, dedup_report = deduplicate(raw_train, "union")
    # The test set stays as published; the report only counts its repeats.
    test_rows, test_dedup = deduplicate(raw_test, "none")

    train_rows = [r for r in train_pool if r["chrom"] not in valid_chroms]
    hic_rows = [r for r in train_rows if r["assay"] == "HiC"]
    val_rows = [r for r in train_pool if r["chrom"] in valid_chroms]
    val_hic = [r for r in val_rows if r["assay"] == "HiC"]

    eval_sets = {"validation_hic": val_hic, "validation_all_assays": val_rows,
                 "test": test_rows}
    for cell in args.test_cells:
        eval_sets[f"test_{cell}"] = [r for r in test_rows if r["cell"] == cell]

    report = {
        "bengi_dir": str(bengi_dir),
        "train_cells": args.train_cells,
        "test_cells": args.test_cells,
        "valid_chroms": sorted(valid_chroms),
        "files": file_table(train_paths + test_paths),
        "training_repeats": repeat_report(raw_train),
        "test_repeats": test_dedup,
        "after_union_dedup": {
            "training_pool_rows": len(train_pool),
            "training_pool_positive_rate": round(
                float(np.mean([r["label"] for r in train_pool])), 4),
            "train_split_rows": len(train_rows),
            "train_split_hic_rows": len(hic_rows),
            "validation_rows": len(val_rows),
            "validation_hic_rows": len(val_hic),
            "rows_by_cell": dict(Counter(r["cell"] for r in train_pool)),
            "rows_by_cell_before": dict(Counter(r["cell"] for r in raw_train)),
        },
        "anchors": anchor_report(train_pool),
        "chromosomes": chromosome_table(train_pool, valid_chroms),
        "positives_by_assay": assay_table(train_pool),
        "baselines": {"trained_on_all_assays": baselines(train_rows, eval_sets),
                      "trained_on_hic": baselines(hic_rows, eval_sets)},
        "test_overlap_with_training": {
            "all_assays": overlap_with_training(train_rows, test_rows),
            "hic": overlap_with_training(hic_rows, test_rows)},
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dataset_audit.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    rep = report["training_repeats"]["exact_repeats_across_assays"]
    print(f"\nTraining rows {rep['rows_in']:,}, distinct pairs {rep['unique_pairs']:,}: "
          f"{rep['duplicate_rows']:,} rows repeat a pair from another assay file, "
          f"{rep['pairs_with_conflicting_labels']:,} pairs have conflicting labels.")
    after = report["after_union_dedup"]
    print(f"After union dedup: {after['training_pool_rows']:,} rows "
          f"(train {after['train_split_rows']:,}, of which Hi-C "
          f"{after['train_split_hic_rows']:,}; validation "
          f"{after['validation_rows']:,}, of which Hi-C "
          f"{after['validation_hic_rows']:,}).")
    for trained_on, blocks in report["baselines"].items():
        print(f"\nBaselines {trained_on.replace('_', ' ')} (AUROC / AUPR):")
        for split, block in blocks.items():
            print(f"  {split} (n={block['distance_rank']['n']:,}, "
                  f"positive rate {block['distance_rank']['positive_rate']}):")
            for name in ("distance_rank", "distance_binned",
                         "gene_prior_only", "distance_plus_locus_priors"):
                m = block[name]
                print(f"    {name:<28} {m['auroc']}  /  {m['aupr']}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

"""
Does the supervised sequence encoding carry training labels into its inputs?

POCD-ND is fitted on labelled sequences: at every position it records how
often each k-mer occurs in interacting and in non-interacting pairs. When the
encoder is fitted on training rows that are then trained on (the original
rule, `data.encoder_fit: train`), each fitted row is encoded with densities
that counted its own label. A network can learn that signature on the
training rows, and it does not exist on validation or test rows.

This script measures it without training anything. For each fitting rule it
fits the encoder exactly as scripts/train.py does, then scores every row with
the naive-Bayes log ratio sum_i log(p_i / n_i) of the k-mers the row holds,
and reports how well that one number separates the labels in three groups:

    fitted      training rows the encoder was fitted on
    unfitted    training rows it was not fitted on
    validation  rows on the held-out chromosomes

If the encoding leaks, `fitted` separates far better than `unfitted`, which
matches `validation`. Under `crossfit` every row is encoded by densities
fitted on the other half of the chromosomes, so the three should agree.

Needs the reference genome and the BENGI files, not the chromatin tracks and
not a GPU.

Usage:

    python scripts/audit_encoder_leakage.py
    python scripts/audit_encoder_leakage.py --protocol results/protocol.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.bengi import bengi_files, read_rows, select_rows_for_config
from src.config import load_config
from src.encoding import CrossFitEncoder
from src.epi_data_pipeline import extract_sequence
from train import fit_sequence_encoder

DEFAULT_TRAIN_CELLS = ["GM12878", "HeLa", "K562", "IMR90"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure label leakage through the POCD-ND encoding.")
    p.add_argument("--config", default="configs/slim_ka.yaml")
    p.add_argument("--protocol", default=None,
                   help="protocol JSON; its training assays choose the rows")
    p.add_argument("--train-cells", nargs="+", default=DEFAULT_TRAIN_CELLS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-group", type=int, default=3000,
                   help="rows scored in each group (fewer if a group is smaller)")
    p.add_argument("--out", default="results/audit/encoder_leakage.json")
    return p.parse_args()


class SequenceRows:
    """The three things `fit_sequence_encoder` needs, without chromatin."""

    def __init__(self, rows, genome_path, enhancer_window, promoter_window):
        self.rows = rows
        self.enhancer_window = enhancer_window
        self.promoter_window = promoter_window
        self.genome = None
        if genome_path and os.path.exists(genome_path):
            import pyfaidx
            self.genome = pyfaidx.Fasta(genome_path)
        else:
            print(f"  WARNING: reference genome {genome_path!r} not found; "
                  f"every sequence is N and the scores are meaningless")

    def get_labels(self):
        return np.array([r["label"] for r in self.rows])

    def get_chrom_groups(self):
        return np.array([r["chrom"] for r in self.rows])

    def sequence_pair(self, idx):
        r = self.rows[idx]
        return (extract_sequence(self.genome, r["chrom"], r["enh_coord"],
                                 self.enhancer_window),
                extract_sequence(self.genome, r["chrom"], r["tss_coord"],
                                 self.promoter_window))


def score_rows(encoder, source, indices):
    scores = []
    for i in indices:
        sequence = "".join(source.sequence_pair(i))
        if isinstance(encoder, CrossFitEncoder):
            scores.append(encoder.log_ratio_score(sequence, source.rows[i]["chrom"]))
        else:
            scores.append(encoder.log_ratio_score(sequence))
    return np.array(scores)


def separation(labels, scores):
    """AUROC of the score, and the gap between class means in pooled SDs."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return {"n": int(len(labels)), "auroc": None, "standardised_gap": None}
    pos, neg = scores[labels == 1], scores[labels == 0]
    pooled = np.sqrt((pos.var() + neg.var()) / 2) or 1.0
    return {"n": int(len(labels)),
            "positives": int(labels.sum()),
            "auroc": round(float(roc_auc_score(labels, scores)), 4),
            "standardised_gap": round(float((pos.mean() - neg.mean()) / pooled), 3)}


def balanced_sample(indices, labels, size, rng):
    """Up to `size` indices, half positive and half negative where possible."""
    indices = np.asarray(indices)
    pos = indices[labels[indices] == 1]
    neg = indices[labels[indices] == 0]
    take_pos = min(len(pos), size // 2)
    take_neg = min(len(neg), size - take_pos)
    return np.concatenate([rng.choice(pos, take_pos, replace=False),
                           rng.choice(neg, take_neg, replace=False)])


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.protocol:
        with open(args.protocol) as handle:
            config["data"]["train_assays"] = json.load(handle).get("train_assays")
    held = set(config["training"].get("valid_chroms", ["chr11", "chr17"]))
    files = bengi_files(config["paths"]["bengi_dir"], args.train_cells)
    rows, _ = select_rows_for_config(read_rows(files), held, config)
    source = SequenceRows(rows, config["paths"].get("ref_genome"),
                          config["data"].get("enhancer_window", 3000),
                          config["data"].get("promoter_window", 3000))
    labels = source.get_labels()
    chroms = source.get_chrom_groups()
    train_idx = [i for i, c in enumerate(chroms) if c not in held]
    val_idx = [i for i, c in enumerate(chroms) if c in held]
    print(f"Rows: {len(train_idx):,} training, {len(val_idx):,} validation "
          f"(assays {config['data'].get('train_assays') or 'all'})")

    report = {"train_assays": config["data"].get("train_assays"),
              "train_rows": len(train_idx), "validation_rows": len(val_idx),
              "rules": {}}
    for rule in ("train", "crossfit"):
        config["data"]["encoder_fit"] = rule
        print(f"\n=== encoder_fit: {rule} ===")
        encoder = fit_sequence_encoder(source, train_idx, config, args.seed)
        fitted = set(encoder.fit_rows)
        rng = np.random.default_rng(args.seed)
        groups = {
            "fitted": balanced_sample(sorted(fitted), labels, args.per_group, rng),
            "unfitted": balanced_sample(
                [i for i in train_idx if i not in fitted], labels,
                args.per_group, rng),
            "validation": balanced_sample(val_idx, labels, args.per_group, rng),
        }
        block = {}
        for name, idx in groups.items():
            if len(idx) == 0:
                continue
            block[name] = separation(labels[idx], score_rows(encoder, source, idx))
            print(f"  {name:<11} n={block[name]['n']:>5}  AUROC "
                  f"{block[name]['auroc']}  gap {block[name]['standardised_gap']} SD")
        report["rules"][rule] = block

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

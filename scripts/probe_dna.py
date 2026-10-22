"""
A linear probe of the frozen language-model DNA embeddings: do the pooled
Nucleotide Transformer embeddings of an enhancer and its promoter say more
about whether the two loop than their distance does? It runs on the CPU in a
few minutes, an early signal for keeping the language-model branch.

The rows are those train.py reads for the configuration (default the P1a
pilot, configs/phase2_p1a.yaml; the language-model arms train on the same
rows), selected by train.training_cells and train.extended_rows: the training
cells and assays, without the never-trained and validation cells, and without
rows that repeat a validation or test pair. The probe trains on --train-rows
of them, drawn cell-balanced as an epoch is (CellBalancedSampler, seed 0), and
is scored on train.py's fixed validation sample (HepG2 and A549,
training.valid_samples rows, class-stratified, seed 0).

Three feature sets, each standardised on the training rows and fitted by
logistic regression:
  distance   log10 of the enhancer-promoter distance
  means      distance, and the mean embedding (the 8 parts averaged) of the
             enhancer and of the promoter
  product    means, and their elementwise product: a multiplicative
             enhancer-promoter term
A gain of `means` over `distance` can also come from the elements alone (a
promoter that loops in every cell), since validation cells share enhancers
and genes with the training cells; the gain of `product` over `means` is the
pairing.

Usage (from Project/, data in datasets/slim_data):
    python scripts/probe_dna.py --data datasets/slim_data
Writes results/probe_dna.json: validation AUROC and AUPR of each set, overall
and per validation cell, and the differences between the sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.bengi import bengi_files, fraction_of_rows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dna_embeddings import CHROMS, ElementEmbeddings, element_key  # noqa: E402

SETS = ("distance", "means", "product")


def probe_rows(config, data: Path, feats: Path):
    """The configuration's rows as train.py selects them: the pair table,
    the training positions, the fixed validation sample and the training
    cells."""
    import train as train_script
    folders = [str(data / "pairs" / "BENGI"), str(data / "pairs" / "BENGI_extended")]
    config["paths"].update(
        bengi_dirs=folders, feats_config=str(feats),
        exclusion_files=[str(data / "pairs" / "BENGI_extra_test" / "*_rao2014-Benchmark.v3.tsv*")])
    args = SimpleNamespace(train_cells=None, test_cells=train_script.DEFAULT_TEST_CELLS)
    cells = train_script.training_cells(config, args, folders)
    valid = config["training"].get("valid_cells") or []
    files = bengi_files(folders, list(cells) + list(valid))
    table, train_idx, val_idx, _report = train_script.extended_rows(config, files)
    n_valid = config["training"].get("valid_samples", 0)
    if n_valid and len(val_idx) > n_valid:
        # train.py's sample: the same for every run and seed.
        val_idx = fraction_of_rows(val_idx, table.columns["label"], n_valid / len(val_idx),
                                   seed=0)
    return table, np.asarray(train_idx), np.asarray(val_idx), cells


def cell_balanced(table, train_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n training rows shared fairly between the cells, as one epoch draws
    them (src/dataset.py CellBalancedSampler)."""
    from src.dataset import CellBalancedSampler
    sampler = CellBalancedSampler(table.strings("cell")[train_idx], n, seed=seed)
    return np.sort(train_idx[np.fromiter(iter(sampler), dtype=np.int64)])


def element_keys(table, rows: np.ndarray):
    """Embedding keys of the enhancer midpoint and the gene start of each
    row, and whether its chromosome can have one."""
    c = table.columns
    chroms = np.asarray(table.vocab["chrom"], dtype=object)[c["chrom"][rows]]
    known = np.isin(chroms, CHROMS)
    chroms = np.where(known, chroms, CHROMS[0])
    enh = (c["enh_start"][rows].astype(np.int64) + c["enh_end"][rows]) // 2
    prom = c["tss_coord"][rows].astype(np.int64)
    return element_key(chroms, enh), element_key(chroms, prom), known


def pooled(embeddings: ElementEmbeddings, keys: np.ndarray, chunk: int = 20_000):
    """The mean over the parts of each key's embedding (float32), and
    whether the key is in the table (zeros if not)."""
    unique, inverse = np.unique(keys, return_inverse=True)
    pos = np.minimum(np.searchsorted(embeddings.keys, unique), len(embeddings.keys) - 1)
    found = embeddings.keys[pos] == unique
    means = np.zeros((len(unique), embeddings.dim), dtype=np.float32)
    rows = np.flatnonzero(found)
    rows = rows[np.argsort(pos[rows])]           # read the memory map in order
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        means[part] = np.asarray(embeddings.array[pos[part]], dtype=np.float32).mean(axis=1)
    return means[inverse], found[inverse]


def features(kind: str, log_dist: np.ndarray, enh: np.ndarray, prom: np.ndarray) -> np.ndarray:
    """The design matrix of one feature set."""
    parts = [log_dist[:, None]]
    if kind in ("means", "product"):
        parts += [enh, prom]
    if kind == "product":
        parts.append(enh * prom)
    return np.hstack(parts).astype(np.float32)


def fit_and_score(x_train, y_train, x_valid, y_valid, cells=None, C: float = 1.0) -> Dict:
    """Standardise on the training rows, fit a logistic regression and score
    the validation rows (AUROC, AUPR; per cell too when `cells` is given)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(scaler.transform(x_train), y_train)
    score = model.decision_function(scaler.transform(x_valid))
    result = {"auroc": float(roc_auc_score(y_valid, score)),
              "aupr": float(average_precision_score(y_valid, score)),
              "iterations": int(np.max(model.n_iter_)), "features": int(x_train.shape[1])}
    if cells is not None:
        result["per_cell"] = {
            str(c): {"auroc": float(roc_auc_score(y_valid[cells == c], score[cells == c])),
                     "aupr": float(average_precision_score(y_valid[cells == c],
                                                           score[cells == c])),
                     "rows": int((cells == c).sum())}
            for c in np.unique(cells)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="datasets/slim_data")
    p.add_argument("--config", default="configs/phase2_p1a.yaml")
    p.add_argument("--feats", default=None,
                   help="feature list that decides which cells have complete tracks "
                        "(default <data>/features.json)")
    p.add_argument("--embeddings", default=None,
                   help="embedding folder (default <data>/dna_features/embeddings)")
    p.add_argument("--train-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 penalty")
    p.add_argument("--out", default="results/probe_dna.json")
    args = p.parse_args()

    start = time.time()
    data = Path(args.data)
    config = load_config(args.config)
    table, train_idx, val_idx, cells = probe_rows(
        config, data, Path(args.feats or data / "features.json"))
    train_rows = cell_balanced(table, train_idx, args.train_rows, args.seed)
    print(f"{len(cells)} training cells, {len(train_idx):,} training rows "
          f"({len(train_rows):,} drawn), {len(val_idx):,} validation rows "
          f"[{time.time() - start:.0f} s]")

    embeddings = ElementEmbeddings(str(args.embeddings or data / "dna_features" / "embeddings"))
    split = {}
    missing = {}
    for name, rows in (("train", train_rows), ("valid", val_idx)):
        enh_keys, prom_keys, known = element_keys(table, rows)
        enh, enh_found = pooled(embeddings, enh_keys)
        prom, prom_found = pooled(embeddings, prom_keys)
        keep = known & enh_found & prom_found
        missing[name] = int((~keep).sum())
        rows = rows[keep]
        dist = table.columns["dist"][rows].astype(np.float64)
        split[name] = {"log_dist": np.log10(np.maximum(dist, 1.0)), "enh": enh[keep],
                       "prom": prom[keep], "y": table.columns["label"][rows].astype(np.int64),
                       "cells": table.strings("cell")[rows]}
    print(f"rows without an embedding (dropped): {missing} [{time.time() - start:.0f} s]")

    tr, va = split["train"], split["valid"]
    results = {}
    for kind in SETS:
        results[kind] = fit_and_score(
            features(kind, tr["log_dist"], tr["enh"], tr["prom"]), tr["y"],
            features(kind, va["log_dist"], va["enh"], va["prom"]), va["y"],
            cells=va["cells"], C=args.C)
        print(f"  {kind:<9} AUROC {results[kind]['auroc']:.4f}  AUPR {results[kind]['aupr']:.4f}"
              f"  ({results[kind]['features']} features) [{time.time() - start:.0f} s]")
    difference = {f"{a} - {b}": {m: results[a][m] - results[b][m] for m in ("auroc", "aupr")}
                  for a, b in (("means", "distance"), ("product", "distance"),
                               ("product", "means"))}
    for name, d in difference.items():
        print(f"  {name:<20} AUROC {d['auroc']:+.4f}  AUPR {d['aupr']:+.4f}")

    report = {
        "config": args.config, "embeddings": embeddings.meta.get("model"),
        "train_cells": list(cells), "valid_cells": config["training"].get("valid_cells"),
        "train_rows": int(len(tr["y"])), "train_positive_rate": float(tr["y"].mean()),
        "valid_rows": int(len(va["y"])), "valid_positive_rate": float(va["y"].mean()),
        "rows_without_embedding": missing, "seed": args.seed, "C": args.C,
        "sets": results, "difference": difference,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"written {out} [{time.time() - start:.0f} s]")


if __name__ == "__main__":
    main()

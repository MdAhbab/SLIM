"""
Build the two extra test sets of the final evaluation, in the benchmark format.

  eqtl      Target genes of fine-mapped eQTLs (eQTL Catalogue credible sets,
            posterior inclusion probability above 0.5; convert_eval.py eqtl).
            For each study, every candidate enhancer that contains one of its
            variants is paired with every gene the benchmark files already
            pair it with. The rows of a gene that the variant regulates in that
            study (its eGene) are positive, the others negative. An enhancer
            whose eGenes are all outside its candidate genes is dropped, as no
            row could be right. The rows carry the cell type matched to the
            study (MATCHED_CELLS), so the model reads that cell's tracks.
  javierre  Javierre et al. 2016 promoter capture Hi-C (B cells, monocytes and
            T cells, from build_pairs.py), restricted to pairs the final
            training never sees: no training row, in any cell, has the same
            enhancer and gene. These cells are trained on with Hi-C labels, so
            only the unseen pairs test transfer to a new assay.

Both use only enhancers and genes from the benchmark files, so every element
has a DNA embedding. With --embeddings, rows whose elements are missing from
the table are dropped and counted (the model would read zeros for them).

The training pairs are those of the final configuration: the cells and assays
that train.py keeps, with the validation and test pairs removed as in
training. They are saved (training_pairs.pkl) for scripts/eval_extra.py,
which marks the eQTL rows seen in training.

Usage (from Project/, data in datasets/slim_data):
    python scripts/data/build_extra_tests.py --data datasets/slim_data \
        --config configs/phase2_dna_both.yaml
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.bengi import bengi_files, open_text, parse_line  # noqa: E402
from src.config import load_config  # noqa: E402

# Study -> (description, cell type whose tracks are read, how close the match is).
MATCHED_CELLS = {
    "QTD000110": ("GEUVADIS LCL", "GM12878", "same cell type"),
    "QTD000221": ("GTEx LCL", "GM12878", "same cell type"),
    "QTD000216": ("GTEx skin fibroblast", "NHDF", "same cell type (dermal fibroblast)"),
    "QTD000100": ("GENCORD cord fibroblast", "NHDF", "proxy (dermal fibroblast)"),
    "QTD000021": ("BLUEPRINT monocyte", "Monocyte", "same cell type"),
    "QTD000409": ("Quach monocyte", "Monocyte", "same cell type"),
    "QTD000474": ("Schmiedel naive B cell", "Bcell", "same cell type"),
    "QTD000479": ("Schmiedel naive CD4 T cell", "Tcell", "same cell type"),
    "QTD000509": ("Schmiedel NK cell", "NK", "same cell type"),
    "QTD000281": ("GTEx skeletal muscle", "Psoas", "same tissue"),
    "QTD000291": ("GTEx ovary", "Ovary", "same tissue"),
    "QTD000296": ("GTEx pancreas", "Pancreas", "same tissue"),
    "QTD000321": ("GTEx small intestine", "SmallIntestine", "same tissue"),
    "QTD000331": ("GTEx stomach", "Gastric", "same tissue"),
    "QTD000361": ("HipSci iPSC", "H1", "proxy (embryonic stem cell)"),
    "QTD000316": ("GTEx skin, sun-exposed", "NHEK", "proxy (keratinocyte; a test cell)"),
}
JAVIERRE_CELLS = ["Bcell", "Monocyte", "Tcell"]


# -- training pairs ------------------------------------------------------------
def training_pairs(config_path: str, data: Path, feats: Path) -> Tuple[Set, List[str], Dict]:
    """(enhancer ID, gene) of every training row of the configuration, read
    exactly as train.py reads it (train.training_cells, train.extended_rows)."""
    import train as train_script
    config = load_config(config_path)
    folders = [str(data / "pairs" / "BENGI"), str(data / "pairs" / "BENGI_extended")]
    config["paths"].update(
        bengi_dirs=folders, feats_config=str(feats),
        exclusion_files=[str(data / "pairs" / "BENGI_extra_test" / "*_rao2014-Benchmark.v3.tsv*")])
    args = SimpleNamespace(train_cells=None, test_cells=train_script.DEFAULT_TEST_CELLS)
    cells = train_script.training_cells(config, args, folders)
    valid = config["training"].get("valid_cells") or []
    files = bengi_files(folders, list(cells) + list(valid))
    table, train_idx, _val_idx, report = train_script.extended_rows(config, files)
    return table.pairs(train_idx), cells, report


# -- embedding coverage --------------------------------------------------------
class Coverage:
    """Which elements have a DNA embedding (src/dna_embeddings.py keys)."""

    def __init__(self, folder):
        from src.dna_embeddings import element_key
        self.element_key = element_key
        self.keys = set(np.load(Path(folder) / "keys.npy").tolist()) if folder else None

    def has(self, row: Dict) -> bool:
        if self.keys is None:
            return True
        keys = self.element_key([row["chrom"], row["chrom"]],
                                [row["enh_coord"], row["tss_coord"]])
        return bool(keys[0] in self.keys and keys[1] in self.keys)


def with_cell(name: str, cell: str) -> str:
    """An enhancer or promoter name ("chrom:a-b|CELL|...") with another cell."""
    parts = name.split("|")
    parts[1] = cell
    return "|".join(parts)


def write_rows(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.writelines(lines)


# -- eQTL ----------------------------------------------------------------------
def read_variants(path: Path) -> Dict[str, Tuple[List[int], List[str]]]:
    """Per chromosome: sorted 1-based positions and their genes."""
    found = defaultdict(list)
    with open_text(str(path)) as handle:
        header = next(handle).rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in handle:
            f = line.rstrip("\n").split("\t")
            found[f[col["chrom"]]].append((int(f[col["pos"]]), f[col["gene_id"]].split(".")[0]))
    return {c: ([p for p, _ in sorted(v)], [g for _, g in sorted(v)]) for c, v in found.items()}


def genes_in(variants, chrom: str, start: int, end: int, flank: int) -> Set[str]:
    """Genes of the variants inside a 0-based half-open enhancer [start, end)
    widened by `flank` on each side (variant positions are 1-based)."""
    if chrom not in variants:
        return set()
    positions, genes = variants[chrom]
    lo = bisect.bisect_right(positions, start - flank)
    hi = bisect.bisect_right(positions, end + flank)
    return set(genes[lo:hi])


def build_eqtl(data: Path, out: Path, flank: int, coverage: Coverage, seen: Set) -> Dict:
    studies = {}
    for path in sorted((data / "eval" / "eqtl").glob("QTD*.tsv*")):
        study = path.name.split(".")[0]
        if study in MATCHED_CELLS:
            studies[study] = read_variants(path)
    # One pass over every pair file: the candidate genes of each enhancer that
    # holds a variant of any study.
    files = sorted(str(p) for folder in ("BENGI", "BENGI_extended", "BENGI_extra_test")
                   for p in (data / "pairs" / folder).glob("*.tsv*"))
    hit: Dict[str, Dict[str, Set[str]]] = {}      # enhancer -> study -> eGenes
    checked: Set[str] = set()
    rows: Dict[str, Dict[Tuple, Tuple[Dict, List[str]]]] = defaultdict(dict)
    for name in files:
        with open_text(name) as handle:
            for line in handle:
                row = parse_line(line)
                if row is None:
                    continue
                enh = row["enh_id"]
                if enh not in checked:
                    checked.add(enh)
                    found = {s: genes_in(v, row["chrom"], row["enh_start"], row["enh_end"], flank)
                             for s, v in studies.items()}
                    found = {s: g for s, g in found.items() if g}
                    if found:
                        hit[enh] = found
                if enh in hit:
                    key = (row["chrom"], row["tss_coord"], row["transcript"])
                    rows[enh].setdefault(key, (row, [x for x in line.rstrip("\n").split("\t") if x]))
    report = {"enhancers_checked": len(checked), "studies": {}}
    for study in sorted(studies):
        label, cell, match = MATCHED_CELLS[study]
        lines, stats = [], defaultdict(int)
        stats["variants"] = sum(len(v[0]) for v in studies[study].values())
        for enh, by_study in hit.items():
            egenes = by_study.get(study)
            if not egenes:
                continue
            stats["enhancers_with_variant"] += 1
            candidates = [(r, f) for r, f in rows[enh].values() if coverage.has(r)]
            stats["rows_without_embedding"] += len(rows[enh]) - len(candidates)
            if not any(r["gene"] in egenes for r, _ in candidates):
                stats["enhancers_without_egene_candidate"] += 1
                continue
            stats["enhancers"] += 1
            for r, f in candidates:
                positive = int(r["gene"] in egenes)
                f = list(f)
                f[0] = str(positive)
                f[5], f[9] = with_cell(f[5], cell), with_cell(f[9], cell)
                lines.append("\t".join(f[:10]) + "\n")
                stats["rows"] += 1
                stats["positives"] += positive
                stats["rows_seen_in_training"] += int((r["enh_id"], r["gene"]) in seen)
        name = f"{cell}.eQTL_{study}-Benchmark.v3.tsv.gz"
        if lines:
            write_rows(out / "eqtl_pairs" / name, lines)
        report["studies"][study] = {"description": label, "cell": cell, "match": match,
                                    "file": name if lines else None, **stats}
        print(f"  {study} {label:28s} -> {cell:14s} {stats['enhancers']:5d} enhancers, "
              f"{stats['rows']:6d} rows, {stats['positives']:5d} positive, "
              f"{stats['rows_seen_in_training']:6d} seen in training")
    return report


# -- Javierre --------------------------------------------------------------------
def build_javierre(data: Path, out: Path, coverage: Coverage, seen: Set) -> Dict:
    report = {}
    for cell in JAVIERRE_CELLS:
        path = data / "pairs" / "BENGI_extended" / f"{cell}.PCHiC_javierre2016-Benchmark.v3.tsv.gz"
        lines, stats = [], defaultdict(int)
        with open_text(str(path)) as handle:
            for line in handle:
                row = parse_line(line)
                if row is None:
                    continue
                stats["rows"] += 1
                stats["positives"] += row["label"]
                if (row["enh_id"], row["gene"]) in seen:
                    continue
                if not coverage.has(row):
                    stats["unseen_rows_without_embedding"] += 1
                    continue
                lines.append(line if line.endswith("\n") else line + "\n")
                stats["unseen_rows"] += 1
                stats["unseen_positives"] += row["label"]
        name = f"{cell}.PCHiC_javierre2016unseen-Benchmark.v3.tsv.gz"
        write_rows(out / "javierre_unseen" / name, lines)
        report[cell] = {"file": name, **stats}
        print(f"  Javierre {cell:9s} {stats['rows']:8d} rows ({stats['positives']} positive); "
              f"unseen in training {stats['unseen_rows']:8d} ({stats['unseen_positives']} positive)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="datasets/slim_data")
    parser.add_argument("--config", default="configs/phase2_dna_both.yaml",
                        help="the final configuration (which rows are trained on)")
    parser.add_argument("--feats", default=None,
                        help="feature list deciding which cells have complete tracks "
                             "(default: <data>/features_motif.json)")
    parser.add_argument("--embeddings", default=None,
                        help="DNA embedding folder (default: <data>/dna_features/embeddings); "
                             "'none' keeps every row (embed the new elements afterwards)")
    parser.add_argument("--out", default=None, help="default: <data>/eval")
    parser.add_argument("--flank", type=int, default=0,
                        help="bp added on each side of an enhancer when matching variants")
    parser.add_argument("--only", choices=["eqtl", "javierre"], default=None)
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "eval"
    feats = Path(args.feats) if args.feats else data / "features_motif.json"
    embeddings = args.embeddings or str(data / "dna_features" / "embeddings")
    coverage = Coverage(None if embeddings == "none" else embeddings)

    print("Training pairs of the final configuration ...")
    seen, cells, row_report = training_pairs(args.config, data, feats)
    print(f"  {len(seen):,} (enhancer, gene) pairs from {len(cells)} training cells")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "training_pairs.pkl", "wb") as handle:
        pickle.dump({"pairs": seen, "cells": cells, "config": args.config}, handle)

    report = {"config": args.config, "training_cells": cells, "flank": args.flank,
              "training_rows": {k: v for k, v in row_report.items() if isinstance(v, int)}}
    if args.only in (None, "eqtl"):
        print("eQTL target genes ...")
        report["eqtl"] = build_eqtl(data, out, args.flank, coverage, seen)
    if args.only in (None, "javierre"):
        print("Javierre capture Hi-C, unseen pairs ...")
        report["javierre"] = build_javierre(data, out, coverage, seen)
    (out / "extra_tests_report.json").write_text(json.dumps(report, indent=1))
    print(f"Report: {out / 'extra_tests_report.json'}")


if __name__ == "__main__":
    main()

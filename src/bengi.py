"""
Reading BENGI benchmark files.

One parser serves the training dataset, the leakage audit and the dataset
audit, so every script sees the same rows in the same order. Row i of the test
set here is row i of the saved test predictions.

Each BENGI row is one enhancer and one transcript start site:

    label  distance  chrom  enh_start  enh_end  enh_name
    prom_chrom  prom_start  prom_end  prom_name

`enh_name` is "chrom:start-end|CELL|ENHANCER_ID" and `prom_name` is
"chrom:tss-tss+1|CELL|GENE_ID|TRANSCRIPT_ID|STRAND".

GM12878 and HeLa each come as three files, one per assay (Hi-C, CTCF ChIA-PET
and RNA polymerase II ChIA-PET). The same pair can therefore appear two or
three times in one cell line, with the same inputs and sometimes different
labels. `deduplicate` merges those rows.
"""

from __future__ import annotations

import glob
import gzip
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple


def open_text(path: str):
    """Open a plain or gzipped text file."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def bengi_files(bengi_dir: str, cells: Sequence[str]) -> List[str]:
    """Benchmark files for the named cell lines, in a fixed sorted order."""
    found = sorted(glob.glob(os.path.join(bengi_dir, "*.tsv*")))
    return [f for f in found if os.path.basename(f).split(".")[0] in cells]


def assay_of(path: str) -> str:
    """Assay named in a BENGI file name, e.g. 'HiC' or 'CTCF-ChIAPET'."""
    parts = os.path.basename(path).split(".")
    return parts[1].replace("-Benchmark", "") if len(parts) > 1 else "unknown"


def parse_line(line: str, assay: str = "unknown") -> Dict | None:
    """Parse one BENGI row, or return None for a short or malformed line."""
    fields = [x for x in line.strip().split("\t") if x]
    if len(fields) < 10:
        return None
    (label, dist, chrom, enh_start, enh_end, enh_name,
     _prom_chrom, _prom_start, _prom_end, prom_name) = fields[:10]
    enh_parts = enh_name.split("|")
    prom_parts = prom_name.split("|")
    tss = prom_parts[0].split(":")[-1].split("-")
    enh_start, enh_end = int(enh_start), int(enh_end)
    return {
        "label": int(label),
        "dist": float(dist),
        "chrom": chrom,
        "enh_start": enh_start,
        "enh_end": enh_end,
        "enh_coord": (enh_start + enh_end) // 2,
        "tss_coord": (int(tss[0]) + int(tss[1])) // 2,
        "tss_start": int(tss[0]),
        "tss_end": int(tss[1]),
        "cell": enh_parts[1],
        "enh_id": enh_parts[2] if len(enh_parts) > 2 else enh_name,
        "gene": prom_parts[2].split(".")[0] if len(prom_parts) > 2 else prom_name,
        "transcript": prom_parts[3] if len(prom_parts) > 3 else "",
        "assay": assay,
    }


def read_rows(paths: Iterable[str]) -> List[Dict]:
    """Every row of the given files, file by file, in file order."""
    rows = []
    for path in paths:
        assay = assay_of(path)
        with open_text(path) as handle:
            for line in handle:
                row = parse_line(line, assay)
                if row is not None:
                    rows.append(row)
    return rows


def pair_key(row: Dict) -> Tuple:
    """Identity of a row: the same cell, enhancer and transcript start site."""
    return (row["cell"], row["chrom"], row["enh_coord"], row["tss_coord"])


def deduplicate(rows: List[Dict], mode: str = "none"
                ) -> Tuple[List[Dict], Dict[str, int]]:
    """Merge rows that describe the same pair in the same cell line.

    mode="none" returns the rows unchanged. mode="union" keeps one row per
    `pair_key`, at the position of its first occurrence, labelled positive if
    any of the merged rows is positive, and records the assays it came from.

    Returns the rows and a report of what was merged.
    """
    if mode not in {"none", "union"}:
        raise ValueError("dedup must be one of: none, union")
    groups: "OrderedDict[Tuple, List[Dict]]" = OrderedDict()
    for row in rows:
        groups.setdefault(pair_key(row), []).append(row)
    repeated = [g for g in groups.values() if len(g) > 1]
    report = {
        "rows_in": len(rows),
        "unique_pairs": len(groups),
        "duplicate_rows": len(rows) - len(groups),
        "pairs_repeated": len(repeated),
        "pairs_with_conflicting_labels": sum(
            1 for g in repeated if len({r["label"] for r in g}) > 1),
    }
    if mode == "none":
        report["rows_out"] = len(rows)
        return rows, report
    merged = []
    for group in groups.values():
        row = dict(group[0])
        row["label"] = max(r["label"] for r in group)
        row["assay"] = "+".join(sorted({r["assay"] for r in group}))
        merged.append(row)
    report["rows_out"] = len(merged)
    return merged, report


def load_pairs(bengi_dir: str, cells: Sequence[str], dedup: str = "none"
               ) -> Tuple[List[Dict], Dict[str, int]]:
    """Read and, if asked, deduplicate the pairs of the named cell lines."""
    return deduplicate(read_rows(bengi_files(bengi_dir, cells)), dedup)


def select_rows(rows: List[Dict], held: Iterable[str],
                train_assays: Sequence[str] | None = None,
                valid_assays: Sequence[str] | None = None,
                dedup: str = "none") -> Tuple[List[Dict], Dict[str, int]]:
    """Rows of the training cell lines that enter training and validation.

    Rows on the held-out chromosomes form the validation set and keep the
    assays in `valid_assays`; the others form the training set and keep the
    assays in `train_assays`. None keeps every assay. Then `dedup` merges
    repeated pairs. Pairs never span chromosomes, so the merge cannot move a
    pair between training and validation.
    """
    held = set(held)
    train_assays = list(train_assays) if train_assays else None
    valid_assays = list(valid_assays) if valid_assays else None

    def keep(row):
        wanted = valid_assays if row["chrom"] in held else train_assays
        return wanted is None or row["assay"] in wanted

    return deduplicate([row for row in rows if keep(row)], dedup)


def select_rows_for_config(rows: List[Dict], held: Iterable[str],
                           config: Dict) -> Tuple[List[Dict], Dict[str, int]]:
    """`select_rows` with the assays and dedup rule named in a configuration."""
    return select_rows(rows, held,
                       train_assays=config["data"].get("train_assays"),
                       valid_assays=config["training"].get("valid_assays"),
                       dedup=config["data"].get("dedup", "none"))

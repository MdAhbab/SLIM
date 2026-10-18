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
from array import array
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def open_text(path: str):
    """Open a plain or gzipped text file."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def bengi_files(bengi_dir: str | Sequence[str], cells: Sequence[str]) -> List[str]:
    """Benchmark files for the named cell lines: folder by folder in the order
    given, sorted within each folder."""
    folders = [bengi_dir] if isinstance(bengi_dir, str) else list(bengi_dir)
    found = [f for folder in folders
             for f in sorted(glob.glob(os.path.join(folder, "*.tsv*")))]
    return [f for f in found if os.path.basename(f).split(".")[0] in cells]


def split_assay(path: str) -> Tuple[str, str]:
    """Assay and label source named in a benchmark file name.

    BENGI's files name the assay only ("GM12878.HiC-Benchmark.v3.tsv.gz").
    Files built by scripts/data/build_pairs.py may add the source after an
    underscore ("GM12878.HiC_encode-Benchmark...", "H1.PCHiC_jung2019-...").
    Returns ("HiC", "encode") for the second and ("HiC", "") for the first.
    """
    parts = os.path.basename(path).split(".")
    if len(parts) < 2:
        return "unknown", ""
    assay, _, source = parts[1].replace("-Benchmark", "").partition("_")
    return assay, source


def assay_of(path: str) -> str:
    """Assay named in a benchmark file name, e.g. 'HiC' or 'CTCF-ChIAPET'."""
    return split_assay(path)[0]


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


def fraction_of_rows(indices: Sequence[int], labels: Sequence[int],
                     fraction: float, seed: int) -> List[int]:
    """A fixed, class-stratified fraction of the training rows.

    Used for learning curves over training-set size. Each class is shuffled
    once with the seed and the first `fraction` of it is kept, so at one seed
    the 25 percent subset lies inside the 50 percent subset, which lies inside
    the full set. The positive rate stays that of the full set.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"data.train_fraction must be in (0, 1], got {fraction}")
    indices = list(indices)
    if fraction == 1:
        return indices
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for cls in (0, 1):
        pool = [i for i in indices if int(labels[i]) == cls]
        order = rng.permutation(len(pool))
        n = max(1, int(round(fraction * len(pool))))
        keep += [pool[j] for j in order[:n]]
    return sorted(keep)


# ---------------------------------------------------------------------------
# Large row sets
# ---------------------------------------------------------------------------
class PairTable:
    """Benchmark rows as NumPy columns, for row sets too large for one dict
    per row.

    The extended training data has millions of rows. As dicts (read_rows) they
    take several gigabytes, and under Windows every DataLoader worker receives
    its own copy. Here a row costs about 40 bytes. Rows are parsed by
    `parse_line` in file order, as `read_rows` parses them, so both read the
    same rows.

    Columns: label (int8), dist (float32), enh_start, enh_end, tss_coord
    (int32), and int32 codes for chrom, cell, assay, source, enh (the enhancer
    ID) and gene, each an index into the list of the same name in `vocab`.
    """

    NUMERIC = (("label", "b"), ("dist", "f"), ("enh_start", "i"),
               ("enh_end", "i"), ("tss_coord", "i"))
    CODED = ("chrom", "cell", "assay", "source", "enh", "gene")

    def __init__(self, columns: Dict[str, np.ndarray], vocab: Dict[str, List[str]]):
        self.columns = columns
        self.vocab = vocab

    # -- building ------------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: Iterable[Dict]) -> "PairTable":
        """From rows parsed by `parse_line` (or `read_rows`)."""
        builder = _TableBuilder()
        for row in rows:
            builder.add(row)
        return builder.build()

    @classmethod
    def from_files(cls, paths: Iterable[str],
                   keep: Optional[Callable[[Dict], bool]] = None) -> "PairTable":
        """Every row of the files, file by file, that `keep` accepts. Each row
        passed to `keep` carries its "assay" and "source" (see split_assay)."""
        builder = _TableBuilder()
        for path in paths:
            assay, source = split_assay(path)
            with open_text(path) as handle:
                for line in handle:
                    row = parse_line(line, assay)
                    if row is None:
                        continue
                    row["source"] = source
                    if keep is None or keep(row):
                        builder.add(row)
        return builder.build()

    # -- access --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.columns["label"])

    def strings(self, name: str) -> np.ndarray:
        """A coded column as strings."""
        return np.asarray(self.vocab[name], dtype=object)[self.columns[name]]

    def row(self, i: int) -> Dict:
        """Row i in the form `parse_line` returns."""
        c, v = self.columns, self.vocab
        start, end = int(c["enh_start"][i]), int(c["enh_end"][i])
        return {"label": int(c["label"][i]), "dist": float(c["dist"][i]),
                "chrom": v["chrom"][c["chrom"][i]], "enh_start": start,
                "enh_end": end, "enh_coord": (start + end) // 2,
                "tss_coord": int(c["tss_coord"][i]), "cell": v["cell"][c["cell"][i]],
                "enh_id": v["enh"][c["enh"][i]], "gene": v["gene"][c["gene"][i]],
                "assay": v["assay"][c["assay"][i]],
                "source": v["source"][c["source"][i]]}

    def take(self, index) -> "PairTable":
        """The rows at `index` (an index array or a boolean mask)."""
        return PairTable({k: col[index] for k, col in self.columns.items()}, self.vocab)

    def pair_keys(self) -> np.ndarray:
        """One int64 per row naming its (enhancer ID, gene) inside this table."""
        return (self.columns["enh"].astype(np.int64) * len(self.vocab["gene"])
                + self.columns["gene"])

    def pairs(self, index=None) -> Set[Tuple[str, str]]:
        """The (enhancer ID, gene) pairs of the rows at `index` (all rows)."""
        keys = self.pair_keys() if index is None else self.pair_keys()[index]
        n = len(self.vocab["gene"])
        enh, gene = self.vocab["enh"], self.vocab["gene"]
        return {(enh[k // n], gene[k % n]) for k in np.unique(keys).tolist()}

    def pair_mask(self, pairs: Iterable[Tuple[str, str]]) -> np.ndarray:
        """True for each row whose (enhancer ID, gene) is in `pairs`."""
        enh = {e: i for i, e in enumerate(self.vocab["enh"])}
        gene = {g: i for i, g in enumerate(self.vocab["gene"])}
        n = len(self.vocab["gene"])
        wanted = np.array([enh[e] * n + gene[g] for e, g in pairs
                           if e in enh and g in gene], dtype=np.int64)
        return np.isin(self.pair_keys(), wanted)

    # -- merging repeats -----------------------------------------------------
    def deduplicate(self, mode: str = "none") -> Tuple["PairTable", Dict[str, int]]:
        """`deduplicate` for a table: the same rule, report and row order."""
        if mode not in {"none", "union"}:
            raise ValueError("dedup must be one of: none, union")
        c = self.columns
        n = len(self)
        enh_coord = (c["enh_start"].astype(np.int64) + c["enh_end"]) // 2
        order = np.lexsort((c["tss_coord"], enh_coord, c["chrom"], c["cell"]))
        key = np.stack([c["cell"][order], c["chrom"][order],
                        enh_coord[order], c["tss_coord"][order]], axis=1)
        new_group = np.ones(n, dtype=bool)
        if n:
            new_group[1:] = (key[1:] != key[:-1]).any(axis=1)
        starts = np.flatnonzero(new_group)
        sizes = np.diff(np.append(starts, n))
        labels = c["label"][order]
        high = np.maximum.reduceat(labels, starts) if n else labels
        low = np.minimum.reduceat(labels, starts) if n else labels
        repeated = sizes > 1
        report = {"rows_in": n, "unique_pairs": len(starts),
                  "duplicate_rows": n - len(starts),
                  "pairs_repeated": int(repeated.sum()),
                  "pairs_with_conflicting_labels": int((repeated & (high != low)).sum())}
        if mode == "none":
            report["rows_out"] = n
            return self, report
        # lexsort is stable, so each group starts at its first occurrence.
        first = order[starts]
        assays = list(self.vocab["assay"])
        if len(assays) > 62:
            raise ValueError("more than 62 assay names; cannot merge their sets")
        bits = np.left_shift(np.int64(1), c["assay"][order].astype(np.int64))
        masks = np.bitwise_or.reduceat(bits, starts) if n else bits
        joined: Dict[int, int] = {}
        merged_assay = np.empty(len(starts), dtype=np.int32)
        for i, mask in enumerate(masks.tolist()):
            if mask not in joined:
                name = "+".join(sorted(assays[b] for b in range(len(assays))
                                       if mask >> b & 1))
                if name not in assays:
                    assays.append(name)
                joined[mask] = assays.index(name)
            merged_assay[i] = joined[mask]
        position = np.argsort(first, kind="stable")
        table = PairTable({k: col[first[position]] for k, col in c.items()},
                          dict(self.vocab, assay=assays))
        table.columns["label"] = high[position].astype(np.int8)
        table.columns["assay"] = merged_assay[position]
        report["rows_out"] = len(table)
        return table, report


class _TableBuilder:
    """Collects rows into typed arrays and string codes."""

    def __init__(self):
        self.numeric = {name: array(code) for name, code in PairTable.NUMERIC}
        self.coded = {name: array("i") for name in PairTable.CODED}
        self.index = {name: {} for name in PairTable.CODED}

    def add(self, row: Dict) -> None:
        for name, _ in PairTable.NUMERIC:
            self.numeric[name].append(row[name])
        values = (("chrom", row["chrom"]), ("cell", row["cell"]),
                  ("assay", row["assay"]), ("source", row.get("source", "")),
                  ("enh", row["enh_id"]), ("gene", row["gene"]))
        for name, value in values:
            codes = self.index[name]
            code = codes.get(value)
            if code is None:
                code = codes[value] = len(codes)
            self.coded[name].append(code)

    def build(self) -> PairTable:
        types = {"b": np.int8, "f": np.float32, "i": np.int32}
        columns = {name: np.array(self.numeric[name], dtype=types[code])
                   for name, code in PairTable.NUMERIC}
        columns.update({name: np.array(self.coded[name], dtype=np.int32)
                        for name in PairTable.CODED})
        vocab = {name: list(codes) for name, codes in self.index.items()}
        return PairTable(columns, vocab)

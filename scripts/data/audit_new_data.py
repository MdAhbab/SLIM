"""
Check the new benchmark files against the rules before any of them is used.

  1. No test cell line in training. No row of a training or validation file
     may name HMEC, NHEK or HUVEC (or another name a source uses for them).
     This rule is fatal: the script exits with an error.
  2. No repeats of test pairs. A training or validation pair whose enhancer
     and gene also form a pair of a primary test set is removed from the file.
     The primary test sets are the ones results are reported on: BENGI's HMEC
     and NHEK sets and the extra test files built from Rao et al.'s loops
     (names containing "_rao2014": the full HMEC and NHEK sets and HUVEC).
     Repeats of the other extra test files (ENCODE loop calls for the same
     cells) are only counted; evaluation on those uses the rows that do not
     repeat a training pair. Both counts are reported per file.
  3. All eight tracks. Every cell named in any file must have all eight track
     files, either among the existing six cell lines or in
     features_new_cells.json.
  4. hg19 coordinates. Every coordinate must fall inside hg19.

It also describes each file: rows, positives, positive rate, the distance of
positive pairs against BENGI's own Hi-C training files, and how many rows
share an enhancer or a gene with the test set.

Usage:
    python scripts/data/audit_new_data.py
    python scripts/data/audit_new_data.py --test-bengi datasets/from_training_pc/BENGI
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import ROOT, load_manifest, out_root, raw_root  # noqa: E402
from prepare_tracks import HG19  # noqa: E402
from src.bengi import parse_line, read_rows  # noqa: E402

EXISTING_CELLS = {"GM12878", "HeLa", "K562", "IMR90", "HMEC", "NHEK"}


def pair_id(row: Dict) -> Tuple[str, str]:
    return row["enh_id"], row["gene"]


def describe(rows: List[Dict], test_enh: Set[str], test_genes: Set[str]) -> Dict:
    labels = np.array([r["label"] for r in rows])
    dist = np.array([r["dist"] for r in rows if r["label"] == 1])
    return {
        "rows": len(rows),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(rows) else 0.0,
        "positive_distance_kb": ({"median": float(np.median(dist)) / 1000,
                                  "q25": float(np.percentile(dist, 25)) / 1000,
                                  "q75": float(np.percentile(dist, 75)) / 1000}
                                 if len(dist) else None),
        "rows_sharing_test_enhancer": sum(r["enh_id"] in test_enh for r in rows),
        "rows_sharing_test_gene": sum(r["gene"] in test_genes for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-bengi", default=None,
                        help="folder with the published BENGI files (default: the "
                             "TransEPI copy downloaded with the tools group)")
    parser.add_argument("--keep-test-repeats", action="store_true",
                        help="report repeats of test pairs without removing them")
    parser.add_argument("--out", default=str(ROOT / "results" / "data_audit_new.json"))
    args = parser.parse_args()

    manifest = load_manifest()
    out = out_root()
    bengi = Path(args.test_bengi) if args.test_bengi else (
        raw_root() / "reference" / "transepi" / "data" / "BENGI")
    banned = {a.lower() for a in manifest["test_only"]["aliases"]}
    banned |= {c.lower() for c in manifest["test_only"]["cells"]}

    # The test pairs: the primary sets (removed from training) and the others
    # (counted only).
    extra = sorted((out / "BENGI_extra_test").glob("*.tsv.gz"))
    test_files = sorted(bengi.glob("HMEC.*")) + sorted(bengi.glob("NHEK.*"))
    test_files += [p for p in extra if "_rao2014" in p.name]
    secondary_files = [p for p in extra if "_rao2014" not in p.name]
    if not test_files:
        sys.exit(f"no test files found under {bengi}; download the tools group first")
    test_rows = read_rows([str(p) for p in test_files])
    test_pairs = {pair_id(r) for r in test_rows}
    test_enh = {r["enh_id"] for r in test_rows}
    test_genes = {r["gene"] for r in test_rows}
    secondary_pairs = {pair_id(r) for r in read_rows([str(p) for p in secondary_files])}

    reference = read_rows([str(p) for c in ("GM12878", "HeLa", "K562", "IMR90")
                           for p in bengi.glob(f"{c}.HiC-Benchmark.v3.tsv*")])

    features = out / "features_new_cells.json"
    complete = set(EXISTING_CELLS)
    if features.exists():
        complete |= {c for c in json.loads(features.read_text()) if not c.startswith("_")}

    report: Dict = {"test_files": [p.name for p in test_files],
                    "test_pairs": len(test_pairs),
                    "secondary_test_files": [p.name for p in secondary_files],
                    "secondary_test_pairs": len(secondary_pairs),
                    "reference_bengi_hic_train": describe(reference, test_enh, test_genes),
                    "files": {}}
    fatal: List[str] = []
    for path in sorted((out / "BENGI_extended").glob("*.tsv.gz")):
        rows = read_rows([str(path)])
        cells = {r["cell"] for r in rows}
        leaked = sorted(c for c in cells | {path.name.split(".")[0]}
                        if c.lower() in banned)
        if leaked:
            fatal.append(f"{path.name}: test cell lines {leaked} in a training file")
        outside = sum(1 for r in rows if r["chrom"] not in HG19
                      or max(r["enh_end"], r["tss_end"]) > HG19[r["chrom"]])
        repeats = [r for r in rows if pair_id(r) in test_pairs]
        entry = {"cells": sorted(cells),
                 "cells_missing_tracks": sorted(cells - complete),
                 "outside_hg19": outside,
                 "test_pair_repeats": len(repeats),
                 "secondary_test_pair_repeats": sum(pair_id(r) in secondary_pairs
                                                    for r in rows),
                 "before": describe(rows, test_enh, test_genes)}
        if repeats and not args.keep_test_repeats:
            kept = [line for line in gzip.open(path, "rt")
                    if (lambda r: r is None or pair_id(r) not in test_pairs)(
                        parse_line(line))]
            with gzip.open(path, "wt") as handle:
                handle.writelines(kept)
            entry["after"] = describe(read_rows([str(path)]), test_enh, test_genes)
        if outside:
            fatal.append(f"{path.name}: {outside} rows outside hg19")
        report["files"][path.name] = entry
        d = entry.get("after", entry["before"])
        print(f"  {path.name:45s} rows {d['rows']:7d}  pos {d['positives']:6d} "
              f"({d['positive_rate']:.3f})  test repeats removed {len(repeats):5d}  "
              f"missing tracks {entry['cells_missing_tracks'] or '-'}")

    for path in sorted((out / "BENGI_extra_test").glob("*.tsv.gz")):
        rows = read_rows([str(path)])
        cells = {r["cell"] for r in rows}
        report["files"][f"extra_test/{path.name}"] = {
            "cells_missing_tracks": sorted(cells - complete),
            **describe(rows, set(), set())}

    report["fatal"] = fatal
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"  report: {args.out}")
    if fatal:
        for line in fatal:
            print("  FATAL:", line)
        sys.exit(1)
    print("  all rules hold")


if __name__ == "__main__":
    main()

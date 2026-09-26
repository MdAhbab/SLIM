"""
Measure how much genomic territory the training and test splits share.

Cross-cell-line evaluation holds out whole cell lines, but the held-out cell
lines still contribute every chromosome. BENGI draws its enhancers from one
registry of candidate regulatory elements, so the same genomic locus can carry
a pair in a training cell line and another pair in a test cell line. Holding
out cell lines therefore does NOT by itself guarantee that no training locus
appears at test time. This script measures the overlap instead of assuming it.

It reports three kinds of reuse for the test set:

  enhancer reuse  the test enhancer locus also carries a training pair
  promoter reuse  the test promoter locus also carries a training pair
  pair reuse      both ends match the SAME training pair, that is, the exact
                  pair was seen in another cell line

Training territory is every locus the model is trained or validated on: the
rows of the training cell lines that the configuration keeps (its training
and validation assays, after merging repeats), with `--protocol` applied as
in training. It then writes a boolean index marking the test pairs that share
no locus with training. `scripts/evaluate.py` scores that subset from predictions that are
already saved, so the question is answered without retraining anything.

Usage:

    python scripts/audit_leakage.py --config configs/slim_ka.yaml
    python scripts/audit_leakage.py --min-overlap 0.5 --out results/leakage
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bengi import read_rows, select_rows_for_config
from src.config import load_config

DEFAULT_TRAIN_CELLS = ["GM12878", "HeLa", "K562", "IMR90"]
DEFAULT_TEST_CELLS = ["HMEC", "NHEK"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantify train/test genomic overlap across cell lines.")
    p.add_argument("--config", default="configs/slim_ka.yaml")
    p.add_argument("--bengi-dir", default=None,
                   help="overrides paths.bengi_dir from the config")
    p.add_argument("--train-cells", nargs="+", default=DEFAULT_TRAIN_CELLS)
    p.add_argument("--test-cells", nargs="+", default=DEFAULT_TEST_CELLS)
    p.add_argument("--protocol", default=None,
                   help="protocol JSON from scripts/choose_protocol.py; its "
                        "training assays define the training territory")
    p.add_argument("--min-overlap", type=float, default=0.5,
                   help="reciprocal overlap fraction counted as the same locus")
    p.add_argument("--out", default="results/leakage",
                   help="directory for the report and the index")
    return p.parse_args()


def open_maybe_gzip(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def bengi_files(bengi_dir, cells):
    """Files for the named cell lines, in the same order the trainer reads them."""
    import glob
    found = sorted(glob.glob(os.path.join(bengi_dir, "*.tsv*")))
    return [f for f in found if os.path.basename(f).split(".")[0] in cells]


def read_pairs(paths):
    """Parse BENGI pairs exactly as `EPIGenomicDataset` does.

    Keeping the parsing and the file order identical means row i here is row i
    of the test dataset, so the index this script writes lines up with the
    saved predictions.
    """
    pairs = []
    for path in paths:
        with open_maybe_gzip(path) as handle:
            for line in handle:
                fields = [x for x in line.strip().split("\t") if x]
                if len(fields) < 10:
                    continue
                (label, dist, chrom, enh_start, enh_end, enh_name,
                 _prom_chrom, _prom_start, _prom_end, prom_name) = fields[:10]
                cell = enh_name.split("|")[1]
                coords = prom_name.split("|")[0].split(":")[-1].split("-")
                pairs.append({
                    "chrom": chrom,
                    "enh": (int(enh_start), int(enh_end)),
                    "prom": (int(coords[0]), int(coords[1])),
                    "cell": cell,
                    "label": int(label),
                    "dist": float(dist),
                })
    return pairs


class IntervalIndex:
    """Per-chromosome interval index answering 'does this locus reuse one?'

    Intervals are kept sorted by start so a query only examines the few
    intervals that could reach the queried position.
    """

    def __init__(self, min_overlap=0.5):
        self.min_overlap = min_overlap
        self._by_chrom = defaultdict(list)
        self._starts = {}
        self._max_len = defaultdict(int)
        self._exact = set()

    def add(self, chrom, start, end):
        self._by_chrom[chrom].append((start, end))
        self._max_len[chrom] = max(self._max_len[chrom], end - start)
        self._exact.add((chrom, start, end))

    def build(self):
        for chrom, intervals in self._by_chrom.items():
            intervals.sort()
            self._by_chrom[chrom] = intervals
            self._starts[chrom] = [s for s, _ in intervals]

    def has_exact(self, chrom, start, end):
        return (chrom, start, end) in self._exact

    def has_overlap(self, chrom, start, end):
        """True when some stored interval reciprocally overlaps this one."""
        intervals = self._by_chrom.get(chrom)
        if not intervals:
            return False
        starts = self._starts[chrom]
        length = max(end - start, 1)
        # Any interval reaching `start` must begin no earlier than
        # start - max_len on this chromosome.
        lo = bisect.bisect_left(starts, start - self._max_len[chrom] - 1)
        for other_start, other_end in intervals[lo:]:
            if other_start >= end:
                break
            overlap = min(end, other_end) - max(start, other_start)
            if overlap <= 0:
                continue
            other_length = max(other_end - other_start, 1)
            if (overlap / length >= self.min_overlap
                    and overlap / other_length >= self.min_overlap):
                return True
        return False


def main():
    args = parse_args()
    config = load_config(args.config)
    bengi_dir = args.bengi_dir or config["paths"].get("bengi_dir", "./data/BENGI")

    train_paths = bengi_files(bengi_dir, args.train_cells)
    test_paths = bengi_files(bengi_dir, args.test_cells)
    if not train_paths or not test_paths:
        raise FileNotFoundError(
            f"no BENGI files found in {bengi_dir} for "
            f"{args.train_cells} / {args.test_cells}")

    print("=" * 70)
    print("  Cross-cell-line genomic overlap audit")
    print("=" * 70)
    print(f"  Training cells: {args.train_cells}")
    print(f"  Test cells:     {args.test_cells}")
    print(f"  Same locus if reciprocal overlap is at least "
          f"{args.min_overlap:.0%}")
    print()

    if args.protocol:
        with open(args.protocol) as handle:
            config["data"]["train_assays"] = json.load(handle).get("train_assays")
    held = set(config["training"].get("valid_chroms", ["chr11", "chr17"]))
    kept, _ = select_rows_for_config(read_rows(train_paths), held, config)
    train_pairs = [{"chrom": r["chrom"], "enh": (r["enh_start"], r["enh_end"]),
                    "prom": (r["tss_start"], r["tss_end"]), "cell": r["cell"],
                    "label": r["label"], "dist": r["dist"]} for r in kept]
    test_pairs = read_pairs(test_paths)
    print(f"  Training assays {config['data'].get('train_assays') or 'all'}, "
          f"validation assays {config['training'].get('valid_assays') or 'all'}")
    print(f"  Training pairs: {len(train_pairs):,}")
    print(f"  Test pairs:     {len(test_pairs):,}")

    enh_index = IntervalIndex(args.min_overlap)
    prom_index = IntervalIndex(args.min_overlap)
    train_pair_keys = set()
    for pair in train_pairs:
        enh_index.add(pair["chrom"], *pair["enh"])
        prom_index.add(pair["chrom"], *pair["prom"])
        train_pair_keys.add((pair["chrom"], pair["enh"], pair["prom"]))
    enh_index.build()
    prom_index.build()

    n = len(test_pairs)
    enh_seen = np.zeros(n, dtype=bool)
    prom_seen = np.zeros(n, dtype=bool)
    pair_seen = np.zeros(n, dtype=bool)
    enh_exact = np.zeros(n, dtype=bool)
    prom_exact = np.zeros(n, dtype=bool)

    for i, pair in enumerate(test_pairs):
        chrom = pair["chrom"]
        e_start, e_end = pair["enh"]
        p_start, p_end = pair["prom"]
        enh_exact[i] = enh_index.has_exact(chrom, e_start, e_end)
        prom_exact[i] = prom_index.has_exact(chrom, p_start, p_end)
        enh_seen[i] = enh_exact[i] or enh_index.has_overlap(chrom, e_start, e_end)
        prom_seen[i] = prom_exact[i] or prom_index.has_overlap(chrom, p_start, p_end)
        pair_seen[i] = (chrom, pair["enh"], pair["prom"]) in train_pair_keys

    either_seen = enh_seen | prom_seen
    both_seen = enh_seen & prom_seen
    disjoint = ~either_seen

    labels = np.array([p["label"] for p in test_pairs], dtype=int)
    chroms = np.array([p["chrom"] for p in test_pairs])
    cells = np.array([p["cell"] for p in test_pairs])

    def pct(mask):
        return 100.0 * float(mask.sum()) / max(n, 1)

    print("\n--- Test pairs reusing training territory ---")
    print(f"  enhancer locus reused        : {int(enh_seen.sum()):>7,}  "
          f"({pct(enh_seen):5.1f} percent)   exact {int(enh_exact.sum()):,}")
    print(f"  promoter locus reused        : {int(prom_seen.sum()):>7,}  "
          f"({pct(prom_seen):5.1f} percent)   exact {int(prom_exact.sum()):,}")
    print(f"  either end reused            : {int(either_seen.sum()):>7,}  "
          f"({pct(either_seen):5.1f} percent)")
    print(f"  both ends reused             : {int(both_seen.sum()):>7,}  "
          f"({pct(both_seen):5.1f} percent)")
    print(f"  identical pair in a train cell: {int(pair_seen.sum()):>7,}  "
          f"({pct(pair_seen):5.1f} percent)")
    print(f"  shares no locus with training : {int(disjoint.sum()):>7,}  "
          f"({pct(disjoint):5.1f} percent)")

    if disjoint.sum() > 0:
        print(f"\n  Disjoint subset positives     : "
              f"{int(labels[disjoint].sum()):,} "
              f"({100 * labels[disjoint].mean():.1f} percent), against "
              f"{100 * labels.mean():.1f} percent overall")

    print("\n--- By test cell line ---")
    for cell in sorted(set(cells.tolist())):
        sel = cells == cell
        count = max(int(sel.sum()), 1)
        reused_pct = 100.0 * float((either_seen & sel).sum()) / count
        print(f"  {cell:<8} n={int(sel.sum()):>6,}  "
              f"either end reused {reused_pct:5.1f} percent  "
              f"disjoint {int((disjoint & sel).sum()):>6,}")

    print("\n--- By chromosome (either end reused) ---")
    rows = []
    for chrom in sorted(set(chroms.tolist()),
                        key=lambda c: (len(c), c)):
        sel = chroms == chrom
        count = int(sel.sum())
        reused = int((either_seen & sel).sum())
        rows.append({"chrom": chrom, "n": count, "reused": reused,
                     "fraction": reused / max(count, 1)})
        print(f"  {chrom:<6} n={count:>6,}  reused {reused:>6,}  "
              f"({100 * reused / max(count, 1):5.1f} percent)")

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "leakage_index.npz")
    np.savez(
        index_path,
        enhancer_reused=enh_seen,
        promoter_reused=prom_seen,
        pair_reused=pair_seen,
        either_reused=either_seen,
        both_reused=both_seen,
        disjoint=disjoint,
        labels=labels,
        chrom=chroms,
        cell=cells,
        test_enh_start=np.array([p["enh"][0] for p in test_pairs], dtype=np.int64),
        test_enh_end=np.array([p["enh"][1] for p in test_pairs], dtype=np.int64),
        test_prom_start=np.array([p["prom"][0] for p in test_pairs], dtype=np.int64),
        test_prom_end=np.array([p["prom"][1] for p in test_pairs], dtype=np.int64),
        min_overlap=np.float64(args.min_overlap),
        train_cells=np.array(args.train_cells),
        test_cells=np.array(args.test_cells),
    )

    summary = {
        "train_cells": args.train_cells,
        "test_cells": args.test_cells,
        "min_overlap": args.min_overlap,
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": n,
        "enhancer_reused": int(enh_seen.sum()),
        "enhancer_reused_exact": int(enh_exact.sum()),
        "promoter_reused": int(prom_seen.sum()),
        "promoter_reused_exact": int(prom_exact.sum()),
        "either_reused": int(either_seen.sum()),
        "both_reused": int(both_seen.sum()),
        "pair_reused": int(pair_seen.sum()),
        "disjoint": int(disjoint.sum()),
        "disjoint_positive_rate": float(labels[disjoint].mean()) if disjoint.sum() else None,
        "overall_positive_rate": float(labels.mean()),
        "by_chromosome": rows,
    }
    report_path = os.path.join(args.out, "leakage_report.json")
    with open(report_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nIndex written to  {index_path}")
    print(f"Report written to {report_path}")
    print("\nNext: score the disjoint subset from predictions already saved,")
    print("  python scripts/evaluate.py --run results/ka --leakage results/leakage")


if __name__ == "__main__":
    main()

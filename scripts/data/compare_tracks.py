"""
Compare two sets of 500 bp track files, bin by bin.

Used to check that the conversion here reproduces a reference set (TransEPI's
public files, or the training PC's), before new cells are trusted. For each
file present in both folders it reports, over chromosomes 1-22 and X:

  bins       whether every chromosome has the same number of bins
  pearson    correlation of the binned values
  spearman   rank correlation of the nonzero bins of either file
  ratio      median of ours / reference over bins where both are above 0.1
  max_abs    largest absolute difference

Usage:
    python scripts/data/compare_tracks.py --ours datasets/processed/tracks \\
        --reference datasets/raw/reference/transepi/data/genomic_data --cell HMEC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def load(path: Path) -> dict:
    import torch
    return {c: v.numpy().astype(np.float64) for c, v in torch.load(path).items()}


def ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def compare(ours: dict, ref: dict) -> dict:
    same_bins = all(len(ours.get(c, [])) == len(ref.get(c, [])) for c in CHROMS)
    a = np.concatenate([ours[c][:len(ref[c])] for c in CHROMS if c in ours and c in ref])
    b = np.concatenate([ref[c][:len(ours[c])] for c in CHROMS if c in ours and c in ref])
    nonzero = (a > 0) | (b > 0)
    both = (a > 0.1) & (b > 0.1)
    return {
        "bins": "same" if same_bins else "different",
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(ranks(a[nonzero]), ranks(b[nonzero]))[0, 1]),
        "ratio": float(np.median(a[both] / b[both])) if both.any() else float("nan"),
        "max_abs": float(np.abs(a - b).max()),
        "mean_ours": float(a.mean()),
        "mean_ref": float(b.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cell", required=True,
                        help="cell name in our file names")
    parser.add_argument("--reference-cell",
                        help="cell name in the reference file names, if different")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()

    ref_cell = args.reference_cell or args.cell
    results = {}
    for path in sorted(args.ours.glob(f"*_{args.cell}_*.500bp.pt")):
        ref = args.reference / path.name.replace(f"_{args.cell}_", f"_{ref_cell}_")
        if not ref.exists():
            print(f"  {path.name}: no reference file")
            continue
        mark = path.name.split("_")[-1].split(".")[0]
        results[mark] = compare(load(path), load(ref))
        r = results[mark]
        print(f"  {mark:9s} bins {r['bins']:9s} pearson {r['pearson']:.4f}  "
              f"spearman {r['spearman']:.4f}  ratio {r['ratio']:.3f}  "
              f"max diff {r['max_abs']:.3f}", flush=True)
    if not results:
        sys.exit(f"no files for {args.cell} in {args.ours}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

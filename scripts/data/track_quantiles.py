"""
Genome-wide quantiles of every chromatin track, for per-cell normalisation.

The Roadmap p-value tracks of different epigenomes differ in scale, mostly
through sequencing depth: the 99th percentile of one mark varies two to four
times across the cells prepared here. With `data.track_norm: cell_quantile`
the dataset maps each cell's values onto a reference distribution (the mean
quantiles of the training cells) before the per-window normalisation. This
script computes the quantiles that mapping needs, from chromosomes 1-22 and X.

CTCF peaks are skipped by default: about 99% of their bins are zero, so their
quantiles say little, and peak heights are already on one scale.

Usage:
    python scripts/data/track_quantiles.py --features datasets/processed/features.json \\
        --out datasets/processed/track_quantiles.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import resolve_track_location  # noqa: E402

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def cell_quantiles(path: str, points: int = 1001) -> List[float]:
    """Quantiles of one track file at `points` evenly spaced levels."""
    import torch
    track = torch.load(path, map_location="cpu", weights_only=False)
    values = np.concatenate([track[c].numpy() for c in CHROMS if c in track])
    return np.quantile(values, np.linspace(0, 1, points)).round(6).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", required=True,
                        help="feature lists (cell -> mark -> track file)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--skip-marks", nargs="*", default=["CTCF"])
    args = parser.parse_args()

    out = Path(args.out)
    result: Dict[str, Dict[str, List[float]]] = (
        json.loads(out.read_text()) if out.exists() else {})
    for features in args.features:
        config = json.loads(Path(features).read_text())
        location = resolve_track_location(features, config.pop("_location", None))
        for cell, marks in config.items():
            for mark, name in marks.items():
                if mark in args.skip_marks or mark in result.get(cell, {}):
                    continue
                path = name if os.path.isabs(name) else os.path.join(location, name)
                result.setdefault(cell, {})[mark] = cell_quantiles(path, args.points)
            print(f"  {cell}: {sorted(result.get(cell, {}))}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"  {len(result)} cells written to {out}")


if __name__ == "__main__":
    main()

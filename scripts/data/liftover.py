"""
GRCh38 to hg19 conversion with UCSC liftOver.

Every interval is lifted on its own, tagged with its row number, so any
payload (a peak's signal, a loop's other anchor) is reattached afterwards by
that number rather than carried through liftOver's column handling. An
interval that fails to map, maps to another chromosome or to a span that
differs from the original by more than `max_change` is dropped, and the
counts are reported.

For loops (bedpe) both anchors are lifted separately and a loop is kept only
if both anchors land on the same chromosome.

Library use:
    from liftover import lift_intervals, lift_bedpe
Command line:
    python scripts/data/liftover.py in.bed.gz out.bed --chain hg38ToHg19.over.chain.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import open_text, raw_root  # noqa: E402

Interval = Tuple[str, int, int]


def default_chain() -> Path:
    return raw_root() / "tools" / "hg38ToHg19.over.chain.gz"


def liftover_binary() -> str:
    found = shutil.which("liftOver")
    if not found:
        raise FileNotFoundError(
            "UCSC liftOver is not on PATH. Activate the slimdata environment "
            "(scripts/data/setup_wsl.sh) or download the binary from "
            "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver")
    return found


def lift_intervals(intervals: Sequence[Interval], chain: Path | None = None,
                   max_change: float = 0.1) -> Tuple[Dict[int, Interval], Dict[str, int]]:
    """Lift intervals; returns {row: (chrom, start, end)} for those kept."""
    chain = Path(chain or default_chain())
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, bad = (os.path.join(tmp, n) for n in ("in.bed", "out.bed", "bad.bed"))
        with open(src, "w") as out:
            for i, (chrom, start, end) in enumerate(intervals):
                if not str(chrom).startswith("chr"):
                    chrom = f"chr{chrom}"
                out.write(f"{chrom}\t{int(start)}\t{int(end)}\t{i}\n")
        subprocess.run([liftover_binary(), src, str(chain), dst, bad],
                       check=True, capture_output=True)
        lifted: Dict[int, Interval] = {}
        counts = {"in": len(intervals), "unmapped": 0, "split": 0,
                  "span_changed": 0, "moved_chrom": 0, "kept": 0}
        seen, split = set(), set()
        with open(dst) as handle:
            for line in handle:
                chrom, start, end, row = line.split("\t")[:4]
                row = int(row)
                if row in seen:            # split mappings: drop the interval
                    lifted.pop(row, None)
                    split.add(row)
                    continue
                seen.add(row)
                lifted[row] = (chrom, int(start), int(end))
    for row in list(lifted):
        o_chrom, o_start, o_end = intervals[row]
        o_chrom = o_chrom if str(o_chrom).startswith("chr") else f"chr{o_chrom}"
        chrom, start, end = lifted[row]
        old, new = max(1, o_end - o_start), end - start
        if chrom != o_chrom:
            counts["moved_chrom"] += 1
            del lifted[row]
        elif abs(new - old) > max_change * old:
            counts["span_changed"] += 1
            del lifted[row]
    counts["unmapped"] = counts["in"] - len(seen)
    counts["split"] = len(split)
    counts["kept"] = len(lifted)
    return lifted, counts


def lift_bedpe(pairs: Sequence[Tuple[Interval, Interval]], chain: Path | None = None,
               max_change: float = 0.1) -> Tuple[List[Tuple[int, Interval, Interval]], Dict]:
    """Lift both anchors of each loop; keep loops whose anchors stay together."""
    flat = [p[0] for p in pairs] + [p[1] for p in pairs]
    lifted, counts = lift_intervals(flat, chain, max_change)
    n = len(pairs)
    kept = []
    for i in range(n):
        a, b = lifted.get(i), lifted.get(i + n)
        if a is None or b is None or a[0] != b[0]:
            continue
        kept.append((i, a, b))
    return kept, {"loops_in": n, "loops_kept": len(kept), "anchors": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="bed file (first three columns are used)")
    parser.add_argument("dest")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--max-change", type=float, default=0.1)
    args = parser.parse_args()
    rows = [line.rstrip("\n").split("\t") for line in open_text(args.source)
            if line.strip() and not line.startswith(("#", "track", "browser"))]
    lifted, counts = lift_intervals([(r[0], int(r[1]), int(r[2])) for r in rows],
                                    args.chain, args.max_change)
    with open(args.dest, "w") as out:
        for i, (chrom, start, end) in sorted(lifted.items()):
            out.write("\t".join([chrom, str(start), str(end)] + rows[i][3:]) + "\n")
    print(counts)


if __name__ == "__main__":
    main()

"""
Genome-wide tracks of a DNA motif, one per strand, in the 500 bp bins of the
chromatin tracks: the sequence information the model reads across the whole
2.5 Mb window.

Loops form mostly between CTCF sites whose motifs point towards each other
(convergent orientation). A CTCF ChIP-seq track shows where CTCF is bound in a
cell, not which way its motif points; the genome sequence does. This scans
hg19 with a JASPAR matrix on both strands and writes, per bin and strand, how
far the best match centred in the bin scores above the score of p-value
--pvalue, in bits (0 when no match passes). The tracks are the same in every
cell.

    python scripts/data/motif_tracks.py --genome hg19.fa --jaspar MA0139.2.jaspar --out DIR
    python scripts/data/motif_tracks.py --genome hg19.fa --jaspar MA0095.4.jaspar \
        --name YY1 --pvalue 1e-4 --out DIR

Writes DIR/motif_<NAME>_plus.500bp.pt and DIR/motif_<NAME>_minus.500bp.pt in
the layout of prepare_tracks.py (a dict from chromosome to a float32 tensor of
ceil(length / 500) + 1 bins; chrM, chrMT and chrY zero), and
DIR/motif_<NAME>.json with the threshold and the number of matches. NAME is
the matrix's name in its header unless --name gives it (JASPAR 2024 names
YY1's current matrix after the mouse gene, Yy1). A short matrix cannot reach
a small p-value: the best match of the 8 bp YY1 core already has p 1.5e-5, so
it is scanned at 1e-4, and a --pvalue no match can reach stops the scan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HG19 = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
    "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
    "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
    "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
    "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    "chrM": 16569, "chrMT": 16569,
}
SKIP = {"chrM", "chrMT", "chrY"}
BIN = 500
CODES = np.full(256, 4, dtype=np.uint8)         # anything else is N
for i, base in enumerate("ACGT"):
    CODES[ord(base)] = CODES[ord(base.lower())] = i


def read_jaspar(text: str):
    """(name, counts of shape (4, length)) from a JASPAR-format matrix."""
    lines = [l for l in text.splitlines() if l.strip()]
    name = lines[0].lstrip(">").split()[-1]
    rows = {}
    for line in lines[1:]:
        base, values = line.split("[", 1)
        rows[base.strip()] = [float(v) for v in values.rstrip("] \t").split()]
    return name, np.array([rows[b] for b in "ACGT"])


def log_odds(counts: np.ndarray, pseudocount: float = 0.8) -> np.ndarray:
    """Log2 odds against a uniform background, (4, length)."""
    p = (counts + pseudocount / 4) / (counts.sum(axis=0) + pseudocount)
    return np.log2(p / 0.25)


def threshold(lod: np.ndarray, pvalue: float, step: float = 0.01) -> float:
    """The lowest score a uniform random sequence reaches with probability at
    most `pvalue`, from the exact distribution of the discretised score."""
    ints = np.round(lod / step).astype(np.int64)
    low = int(ints.min(axis=0).sum())
    dist = np.zeros(int(ints.max(axis=0).sum()) - low + 1)
    dist[0] = 1.0
    offset = 0
    for column in ints.T:
        shifted = np.zeros_like(dist)
        for value in column:
            move = value - column.min()
            shifted[move:] += 0.25 * dist[:len(dist) - move]
        dist = shifted
        offset += column.min()
    tail = np.cumsum(dist[::-1])[::-1]         # P(score >= low + k)
    if not (tail <= pvalue).any():
        # argmax would return 0, the lowest score, and every position would
        # count as a match.
        raise ValueError(f"no score of this {lod.shape[1]} bp matrix has p <= {pvalue}; "
                         f"its best match has p {tail[-1]:.2g}")
    k = int(np.argmax(tail <= pvalue))
    return (offset + k) * step


def scan(codes: np.ndarray, lod: np.ndarray, chunk: int = 20_000_000) -> np.ndarray:
    """Score of the match starting at every position (N scores very low)."""
    width = lod.shape[1]
    # Row j: the score of each base (A, C, G, T, N) at motif position j.
    table = np.vstack([lod, np.full((1, width), -1e4)]).T.astype(np.float32).copy()
    n = len(codes) - width + 1
    out = np.empty(max(n, 0), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        score = np.zeros(stop - start, dtype=np.float32)
        for j in range(width):
            score += table[j][codes[start + j:stop + j]]
        out[start:stop] = score
    return out


def bin_best(scores: np.ndarray, width: int, cut: float, n_bins: int) -> np.ndarray:
    """Per bin, the best match centred in it, as bits above `cut` (0 below)."""
    centred = np.zeros(n_bins * BIN, dtype=np.float32)
    above = np.maximum(scores - cut, 0.0)
    centred[width // 2: width // 2 + len(above)] = above[:n_bins * BIN - width // 2]
    return centred.reshape(n_bins, BIN).max(axis=1)


def motif_tracks(genome, lod: np.ndarray, cut: float, chroms=HG19):
    """Plus- and minus-strand tracks of every chromosome, and match counts."""
    import torch
    rc = lod[::-1, ::-1]                       # complement rows (ACGT), reversed
    width = lod.shape[1]
    plus, minus, matches = {}, {}, {"plus": 0, "minus": 0}
    for chrom, length in chroms.items():
        n_bins = math.ceil(length / BIN) + 1
        if chrom in SKIP or chrom not in genome:
            plus[chrom] = torch.zeros(n_bins)
            minus[chrom] = torch.zeros(n_bins)
            continue
        sequence = str(genome[chrom][:length]).encode()
        codes = CODES[np.frombuffer(sequence, dtype=np.uint8)]
        for strand, matrix, out in (("plus", lod, plus), ("minus", rc, minus)):
            scores = scan(codes, matrix)
            matches[strand] += int((scores >= cut).sum())
            out[chrom] = torch.from_numpy(bin_best(scores, width, cut, n_bins))
        print(f"  {chrom}: {matches['plus']:,} plus and {matches['minus']:,} minus "
              f"matches so far", flush=True)
    return plus, minus, matches


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--genome", required=True, help="hg19 FASTA (indexed by pyfaidx)")
    p.add_argument("--jaspar", required=True, help="JASPAR-format matrix file")
    p.add_argument("--out", required=True)
    p.add_argument("--pvalue", type=float, default=1e-5)
    p.add_argument("--name", default=None,
                   help="motif name in the output files (default: the matrix's name)")
    args = p.parse_args()

    import pyfaidx
    import torch
    name, counts = read_jaspar(Path(args.jaspar).read_text())
    name = args.name or name
    lod = log_odds(counts)
    cut = threshold(lod, args.pvalue)
    print(f"{name}: {lod.shape[1]} bp, threshold {cut:.2f} bits (p {args.pvalue}), "
          f"best possible {lod.max(axis=0).sum():.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plus, minus, matches = motif_tracks(pyfaidx.Fasta(args.genome), lod, cut)
    torch.save(plus, out / f"motif_{name}_plus.500bp.pt")
    torch.save(minus, out / f"motif_{name}_minus.500bp.pt")
    report = {"motif": name, "matrix": Path(args.jaspar).name, "pvalue": args.pvalue,
              "threshold_bits": cut, "matches": matches}
    (out / f"motif_{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

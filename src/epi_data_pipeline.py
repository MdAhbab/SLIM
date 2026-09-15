"""
Epigenetic signal data loading for POCD-KansformerEPI.

Reads the BENGI benchmark TSV files and pre-processed .pt epigenetic signal files
(same format as the original Kansformer), then adds DNA sequence extraction for
the POCD-ND encoding branch.

Data layout expected:
  data/
    BENGI/
      GM12878.HiC-Benchmark.v3.tsv.gz   (or .tsv)
      ...
    genomic_data/
      processed/
        bigWig_GM12878_DNase.500bp.pt
        narrowPeak_GM12878_CTCF.500bp.pt
        ...
      CTCF_DNase_6histone_local.500.json  (feature config)
"""

import os
import sys
import csv
import gzip
import json
import math
import random
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# hg19 chromosome sizes (for boundary checking)
# ---------------------------------------------------------------------------
HG19_CHROMSIZE = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430,
    "chr4": 191154276, "chr5": 180915260, "chr6": 171115067,
    "chr7": 159138663, "chr8": 146364022, "chr9": 141213431,
    "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
    "chr16": 90354753, "chr17": 81195210, "chr18": 78077248,
    "chr19": 59128983, "chr20": 63025520, "chr21": 48129895,
    "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    "chrM": 16569, "chrMT": 16569,
}


def _open_file(fn: str):
    """Open plain text or gzipped file."""
    if fn.endswith(".gz"):
        return gzip.open(fn, "rt")
    return open(fn, "rt")


def _normalize_minmax(tensor: torch.Tensor) -> torch.Tensor:
    """Per-feature (row) min-max normalisation."""
    mn = tensor.min(dim=1, keepdim=True)[0]
    mx = tensor.max(dim=1, keepdim=True)[0]
    return (tensor - mn) / (mx - mn + 1e-8)


def _sym_log(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log10(1 + |x|).  Matches reference epi_dataset.py."""
    return torch.sign(x) * torch.log10(1 + torch.abs(x))


# ===================================================================
# Core data-loading class
# ===================================================================
class EPIGenomicDataset(Dataset):
    """
    PyTorch Dataset for Enhancer-Promoter Interaction prediction.

    Each sample returns:
        epi_features : Tensor of shape (num_feats, num_bins)
        enhancer_seq : str   (DNA sequence around enhancer center)
        promoter_seq : str   (DNA sequence around promoter center)
        distance     : float (log-scaled genomic distance)
        label        : float (0 or 1)
    """

    _FEATS_ORDER_8 = [
        "CTCF", "DNase", "H3K27ac", "H3K27me3",
        "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3",
    ]
    _FEATS_ORDER_6 = [
        "CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3", "H3K9ac",
    ]

    def __init__(
        self,
        bengi_paths: Union[str, List[str]],
        feats_config_path: str,
        feats_order: Optional[List[str]] = None,
        seq_len: int = 2_500_000,
        bin_size: int = 500,
        enhancer_window: int = 3000,
        promoter_window: int = 3000,
        ref_genome_path: Optional[str] = None,
        normalize_epi: bool = True,
    ):
        """
        Parameters
        ----------
        bengi_paths : path(s) to BENGI TSV / TSV.gz benchmark files
        feats_config_path : path to the JSON mapping cell→mark→.pt file
        feats_order : ordered list of epigenetic mark names to use
        seq_len : genomic window size (bp) for epigenetic features
        bin_size : bin size (bp) for the .pt signal files
        enhancer_window : bp length of DNA to extract around enhancer center
        promoter_window : bp length of DNA to extract around promoter center
        ref_genome_path : path to hg19 FASTA (indexed). If None, returns
                          'N'-padded dummy sequences.
        normalize_epi : apply per-feature min-max normalisation
        """
        super().__init__()

        if isinstance(bengi_paths, str):
            bengi_paths = [bengi_paths]
        self.bengi_paths = bengi_paths

        self.seq_len = int(seq_len)
        self.bin_size = int(bin_size)
        assert self.seq_len % self.bin_size == 0
        self.num_bins = self.seq_len // self.bin_size

        self.enhancer_window = enhancer_window
        self.promoter_window = promoter_window
        self.normalize_epi = normalize_epi

        # Determine feature order
        if feats_order is None:
            feats_order = list(self._FEATS_ORDER_8)
        self.feats_order = list(feats_order)
        self.num_feats = len(self.feats_order)

        # Load feature config JSON
        with open(feats_config_path, "r") as f:
            self.feats_config = json.load(f)
        base_location = self.feats_config.pop("_location", None)
        if base_location is None:
            base_location = os.path.dirname(os.path.abspath(feats_config_path))
        # Resolve absolute paths for each .pt file
        for cell, assays in list(self.feats_config.items()):
            if not isinstance(assays, dict):
                # Skip non-dict entries (e.g. metadata keys)
                del self.feats_config[cell]
                continue
            for mark, fn in assays.items():
                if not isinstance(fn, str):
                    continue
                if not os.path.isabs(fn):
                    assays[mark] = os.path.join(base_location, fn)

        # Chromosome → max bins
        self.chrom_bins = {
            ch: (sz // bin_size)
            for ch, sz in HG19_CHROMSIZE.items()
        }

        # Reference genome (optional). Store the path and open lazily: pyfaidx
        # holds an open file handle (_io.BufferedReader) that cannot be pickled,
        # which would break multi-worker DataLoaders on Windows (spawn). Each
        # process (main + every worker) opens its own handle on first use.
        self.ref_genome_path = (
            ref_genome_path if (ref_genome_path and os.path.exists(ref_genome_path)) else None
        )
        self.ref_genome = None
        if self.ref_genome_path:
            self._open_ref_genome(verbose=True)

        # Storage
        self.samples: List[dict] = []
        self.feats: Dict[str, Dict[str, dict]] = {}  # cell → mark → {chrom: tensor}

        self._load_datasets()
        print(f"  EPIGenomicDataset ready: {len(self.samples)} samples, "
              f"{self.num_feats} marks, {self.num_bins} bins")

    # ---------------------------------------------------------------
    def _open_ref_genome(self, verbose: bool = False):
        """Open the pyfaidx FASTA handle for this process (idempotent)."""
        if self.ref_genome is not None or not self.ref_genome_path:
            return
        try:
            import pyfaidx
            self.ref_genome = pyfaidx.Fasta(self.ref_genome_path)
            if verbose:
                print(f"  Reference genome loaded: {self.ref_genome_path}")
        except ImportError:
            if verbose:
                print("  WARNING: pyfaidx not installed – using dummy sequences")
            self.ref_genome_path = None

    def __getstate__(self):
        # Drop the unpicklable open FASTA handle; workers reopen lazily.
        state = self.__dict__.copy()
        state["ref_genome"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ref_genome = None  # reopened on first sequence access in this process

    # ---------------------------------------------------------------
    def _load_datasets(self):
        for fn in self.bengi_paths:
            with _open_file(fn) as f:
                for line in f:
                    fields = [x for x in line.strip().split("\t") if x]
                    if len(fields) < 10:
                        continue
                    (label, dist, chrom,
                     enh_start, enh_end, enh_name,
                     _prom_chrom,
                     prom_start, prom_end, prom_name) = fields[:10]

                    cell = enh_name.split("|")[1]

                    enh_coord = (int(enh_start) + int(enh_end)) // 2
                    p_coords = prom_name.split("|")[0].split(":")[-1].split("-")
                    tss_coord = (int(p_coords[0]) + int(p_coords[1])) // 2

                    mid = (enh_coord + tss_coord) // 2
                    seq_begin = mid - self.seq_len // 2
                    seq_end = mid + self.seq_len // 2

                    enh_bin = enh_coord // self.bin_size
                    prom_bin = tss_coord // self.bin_size
                    start_bin = seq_begin // self.bin_size
                    stop_bin = seq_end // self.bin_size

                    left_pad, right_pad = 0, 0
                    if start_bin < 0:
                        left_pad = abs(start_bin)
                        start_bin = 0
                    if chrom in self.chrom_bins and stop_bin > self.chrom_bins[chrom]:
                        right_pad = stop_bin - self.chrom_bins[chrom]
                        stop_bin = self.chrom_bins[chrom]

                    self.samples.append({
                        "start_bin": start_bin,
                        "stop_bin": stop_bin,
                        "left_pad": left_pad,
                        "right_pad": right_pad,
                        "enh_bin": enh_bin,
                        "prom_bin": prom_bin,
                        "enh_coord": enh_coord,
                        "prom_coord": tss_coord,
                        "cell": cell,
                        "chrom": chrom,
                        "label": int(label),
                        "dist": float(dist),
                    })

                    # Lazy-load .pt files per cell line
                    if cell not in self.feats:
                        self._load_cell_features(cell)

    def _load_cell_features(self, cell: str):
        """Load all epigenetic signal .pt files for a cell line."""
        if cell not in self.feats_config:
            print(f"  WARNING: cell '{cell}' not in feats_config – skipping")
            self.feats[cell] = {}
            return
        self.feats[cell] = {}
        for mark in self.feats_order:
            if mark not in self.feats_config[cell]:
                print(f"  WARNING: mark '{mark}' not in config for {cell}")
                continue
            pt_path = self.feats_config[cell][mark]
            if not os.path.exists(pt_path):
                print(f"  WARNING: {pt_path} not found")
                continue
            self.feats[cell][mark] = torch.load(pt_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(self.feats[cell])}/{self.num_feats} marks for {cell}")

    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        s = self.samples[idx]
        start_bin = s["start_bin"]
        stop_bin = s["stop_bin"]
        left_pad = s["left_pad"]
        right_pad = s["right_pad"]
        cell = s["cell"]
        chrom = s["chrom"]

        # --- Epigenetic features: (num_feats, num_bins) ---
        feat_rows = []
        for mark in self.feats_order:
            if cell in self.feats and mark in self.feats[cell] and chrom in self.feats[cell][mark]:
                row = self.feats[cell][mark][chrom][start_bin:stop_bin]
            else:
                row = torch.zeros(stop_bin - start_bin)
            feat_rows.append(row.unsqueeze(0))  # (1, bins)

        ar = torch.cat(feat_rows, dim=0)  # (num_feats, stop-start)

        # Pad if near chromosome edges
        if left_pad > 0 or right_pad > 0:
            ar = torch.cat([
                torch.zeros(self.num_feats, left_pad),
                ar,
                torch.zeros(self.num_feats, right_pad),
            ], dim=1)

        if self.normalize_epi:
            ar = _normalize_minmax(ar)

        # --- Enhancer / Promoter relative indices (for index-based feature extraction) ---
        enh_idx = s["enh_bin"] - start_bin + left_pad
        prom_idx = s["prom_bin"] - start_bin + left_pad

        # Clamp to valid range
        enh_idx = max(0, min(enh_idx, self.num_bins - 1))
        prom_idx = max(0, min(prom_idx, self.num_bins - 1))

        # --- Positional encoding channel (matches reference epi_dataset.py) ---
        # V-shaped signal encoding distance to nearest enh/prom boundary
        pos = torch.arange(self.num_bins).float()
        d1 = pos - min(enh_idx, prom_idx)       # distance from left boundary
        d2 = max(enh_idx, prom_idx) - pos        # distance from right boundary
        pos_enc_raw = torch.stack([d1, d2], dim=0)   # (2, num_bins)
        pos_enc = _sym_log(pos_enc_raw.min(dim=0)[0]).unsqueeze(0)  # (1, num_bins)

        # Prepend positional encoding to features (NOT normalized)
        ar = torch.cat([pos_enc, ar], dim=0)     # (num_feats + 1, num_bins)

        # --- DNA sequences ---
        enh_seq = self._extract_seq(chrom, s["enh_coord"], self.enhancer_window)
        prom_seq = self._extract_seq(chrom, s["prom_coord"], self.promoter_window)

        # Distance: log-scaled (same formula as original Kansformer)
        dist_scaled = math.log2(1 + 500_000 / max(s["dist"], 1.0))

        return {
            "epi": ar,                                           # (num_feats+1, num_bins)
            "enhancer_seq": enh_seq,                             # str
            "promoter_seq": prom_seq,                            # str
            "label": torch.tensor([s["label"]], dtype=torch.float),
            "dist": torch.tensor([dist_scaled], dtype=torch.float),
            "enh_idx": torch.tensor([enh_idx], dtype=torch.float),
            "prom_idx": torch.tensor([prom_idx], dtype=torch.float),
        }

    def _extract_seq(self, chrom: str, center: int, window: int) -> str:
        """Extract a DNA sequence of `window` bp centred on `center`."""
        half = window // 2
        start = max(0, center - half)
        end = start + window
        if self.ref_genome is None and self.ref_genome_path is not None:
            self._open_ref_genome()  # lazily reopen per-process (worker)
        if self.ref_genome is not None:
            try:
                chrom_rec = self.ref_genome[chrom]
                chrom_len = len(chrom_rec)
                end = min(end, chrom_len)
                seq = str(chrom_rec[start:end]).upper()
                if len(seq) < window:
                    seq = seq + "N" * (window - len(seq))
                return seq
            except (KeyError, ValueError):
                pass
        # Fallback: N-padded dummy
        return "N" * window

    # ---------------------------------------------------------------
    # Utility: split by chromosome for cross-validation
    # ---------------------------------------------------------------
    def get_chrom_groups(self) -> np.ndarray:
        """Return an array of chromosome strings for GroupKFold splitting."""
        return np.array([s["chrom"] for s in self.samples])

    def get_labels(self) -> np.ndarray:
        return np.array([s["label"] for s in self.samples])

    def get_distances(self) -> np.ndarray:
        return np.array([s["dist"] for s in self.samples])


# ===================================================================
# Alternative: Load from EPIPDLF CSV files (simpler pair format)
# ===================================================================
def load_epipdlf_csv(csv_path: str) -> List[dict]:
    """
    Read an EPIPDLF CSV file and return a list of dicts with keys:
        enhancer_chrom, enhancer_start, enhancer_end,
        promoter_chrom, promoter_start, promoter_end,
        label, distance
    """
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "enhancer_chrom": row["enhancer_chrom"],
                "enhancer_start": int(row["enhancer_start"]),
                "enhancer_end": int(row["enhancer_end"]),
                "promoter_chrom": row["promoter_chrom"],
                "promoter_start": int(row["promoter_start"]),
                "promoter_end": int(row["promoter_end"]),
                "label": int(row["label"]),
                "distance": int(row["enhancer_distance_to_promoter"]),
            })
    return pairs


# ===================================================================
# Quick self-test
# ===================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bengi", nargs="+", required=True)
    p.add_argument("--feats-config", required=True)
    p.add_argument("--feats-order", nargs="+", default=None)
    args = p.parse_args()

    ds = EPIGenomicDataset(
        bengi_paths=args.bengi,
        feats_config_path=args.feats_config,
        feats_order=args.feats_order,
    )
    sample = ds[0]
    print(f"epi shape: {sample['epi'].shape}")
    print(f"enhancer_seq length: {len(sample['enhancer_seq'])}")
    print(f"label: {sample['label']}, dist: {sample['dist']}")

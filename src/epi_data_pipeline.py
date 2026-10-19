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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import resolve_track_location
from src.bengi import PairTable, read_rows

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


def extract_sequence(ref_genome, chrom: str, center: int, window: int) -> str:
    """`window` bp of an open pyfaidx genome centred on `center`.

    Positions past the chromosome end are padded with N. Without a genome, or
    for an unknown chromosome, the whole window is N.
    """
    half = window // 2
    start = max(0, center - half)
    end = start + window
    if ref_genome is not None:
        try:
            chrom_rec = ref_genome[chrom]
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


# ===================================================================
# Core data-loading class
# ===================================================================
def dataset_options(config: dict) -> dict:
    """EPIGenomicDataset keyword arguments named by a configuration.

    Training, evaluation and the analysis scripts all build their datasets
    from this, so a run is always scored with the inputs it was trained on.
    """
    data, paths = config["data"], config.get("paths", {})
    return dict(
        feats_config_path=paths.get("feats_config", ""),
        feats_order=data.get("feats_order"),
        seq_len=data.get("seq_len_bp", 2_500_000),
        bin_size=data.get("bin_size", 500),
        enhancer_window=data.get("enhancer_window", 3000),
        promoter_window=data.get("promoter_window", 3000),
        ref_genome_path=paths.get("ref_genome") or None,
        presence_channels=data.get("presence_channels", False),
        track_quantiles=(paths.get("track_quantiles")
                         if data.get("track_norm", "none") == "cell_quantile" else None),
        quantile_reference_cells=data.get("quantile_reference_cells"),
    )


def load_track(path: str, mmap: bool = True) -> Dict[str, torch.Tensor]:
    """One track file: a dict from chromosome to a float tensor of bins.

    With `mmap` the file is memory-mapped, so processes that read the same
    file share one copy in the operating system's cache instead of holding
    one each. Files written in torch's legacy format cannot be mapped and are
    read normally.
    """
    if mmap:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError, TypeError):
            pass
    return torch.load(path, map_location="cpu", weights_only=False)


class _SampleView:
    """Read-only list-like access to the samples, one dict per index."""

    def __init__(self, dataset: "EPIGenomicDataset"):
        self._ds = dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        return self._ds.sample(int(idx))

    def __iter__(self):
        for i in range(len(self)):
            yield self._ds.sample(i)


class EPIGenomicDataset(Dataset):
    """
    PyTorch Dataset for Enhancer-Promoter Interaction prediction.

    Each sample returns:
        epi_features : Tensor of shape (num_feats, num_bins)
        enhancer_seq : str   (DNA sequence around enhancer center)
        promoter_seq : str   (DNA sequence around promoter center)
        distance     : float (log-scaled genomic distance)
        label        : float (0 or 1)

    The rows are held as NumPy columns (see src.bengi.PairTable), so millions
    of rows fit in memory and pickle cheaply to DataLoader workers. Track files
    are opened lazily in each process, memory-mapped where possible.
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
        bengi_paths: Union[str, List[str], None],
        feats_config_path: str,
        feats_order: Optional[List[str]] = None,
        seq_len: int = 2_500_000,
        bin_size: int = 500,
        enhancer_window: int = 3000,
        promoter_window: int = 3000,
        ref_genome_path: Optional[str] = None,
        normalize_epi: bool = True,
        rows: Optional[List[dict]] = None,
        table: Optional[PairTable] = None,
        presence_channels: bool = False,
        track_quantiles: Optional[str] = None,
        quantile_reference_cells: Optional[List[str]] = None,
        mmap_tracks: bool = True,
    ):
        """
        Parameters
        ----------
        bengi_paths : path(s) to BENGI TSV / TSV.gz benchmark files, read in
                      order. Ignored when `rows` or `table` is given.
        feats_config_path : path to the JSON mapping cell→mark→.pt file
        feats_order : ordered list of epigenetic mark names to use
        seq_len : genomic window size (bp) for epigenetic features
        bin_size : bin size (bp) for the .pt signal files
        enhancer_window : bp length of DNA to extract around enhancer center
        promoter_window : bp length of DNA to extract around promoter center
        ref_genome_path : path to hg19 FASTA (indexed). If None, returns
                          'N'-padded dummy sequences.
        normalize_epi : apply per-feature min-max normalisation
        rows : BENGI rows already parsed by `src.bengi` (and possibly filtered
               or deduplicated). Used instead of reading `bengi_paths`.
        table : the rows as a `src.bengi.PairTable`, for large row sets.
        presence_channels : append one channel per mark, 1 where the cell has
                            that track and 0 where it is missing (its bins
                            are then zero). Lets cells without some marks be
                            used, and is what mark dropout switches off.
        track_quantiles : JSON written by scripts/data/track_quantiles.py.
                          With it, each cell's track is mapped, value by value,
                          onto the mean quantiles of `quantile_reference_cells`
                          before the window normalisation, so cells whose
                          tracks differ in scale (sequencing depth) match.
        mmap_tracks : memory-map the track files (shared between processes).
        """
        super().__init__()

        if isinstance(bengi_paths, str):
            bengi_paths = [bengi_paths]
        self.bengi_paths = bengi_paths or []

        self.seq_len = int(seq_len)
        self.bin_size = int(bin_size)
        assert self.seq_len % self.bin_size == 0
        self.num_bins = self.seq_len // self.bin_size

        self.enhancer_window = enhancer_window
        self.promoter_window = promoter_window
        self.normalize_epi = normalize_epi
        self.presence_channels = bool(presence_channels)
        self.mmap_tracks = mmap_tracks

        # Determine feature order
        if feats_order is None:
            feats_order = list(self._FEATS_ORDER_8)
        self.feats_order = list(feats_order)
        self.num_feats = len(self.feats_order)

        # Load feature config JSON
        with open(feats_config_path, "r") as f:
            self.feats_config = json.load(f)
        declared_location = self.feats_config.pop("_location", None)
        base_location = resolve_track_location(feats_config_path,
                                               declared_location)
        if declared_location and os.path.normpath(
                str(declared_location)) != os.path.normpath(base_location):
            print(f"  Track directory '{declared_location}' from the feature "
                  f"config is not present; reading tracks from {base_location}")
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

        if table is None:
            table = PairTable.from_rows(rows if rows is not None
                                        else read_rows(self.bengi_paths))
        self._build_samples(table)

        # Track files per cell, opened lazily in each process (see _track).
        self.track_paths: Dict[str, Dict[str, str]] = {}
        self._feats: Dict[Tuple[str, str], dict] = {}
        for code in np.unique(self.cell_code).tolist():
            self._find_cell_tracks(self.cell_names[code])

        self.quantile_maps = None
        if track_quantiles:
            self.quantile_maps = self._quantile_maps(track_quantiles,
                                                     quantile_reference_cells)
        print(f"  EPIGenomicDataset ready: {len(self)} samples, "
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
        # Drop the unpicklable open FASTA handle and the opened tracks; each
        # worker reopens both lazily.
        state = self.__dict__.copy()
        state["ref_genome"] = None
        state["_feats"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ref_genome = None  # reopened on first sequence access in this process

    # ---------------------------------------------------------------
    def _build_samples(self, table: PairTable):
        """The window of every row, computed for all rows at once."""
        c = table.columns
        self.cell_names = list(table.vocab["cell"])
        self.chrom_names = list(table.vocab["chrom"])
        self.assay_names = list(table.vocab["assay"])
        enh_coord = (c["enh_start"].astype(np.int64) + c["enh_end"]) // 2
        tss_coord = c["tss_coord"].astype(np.int64)

        mid = (enh_coord + tss_coord) // 2
        seq_begin = mid - self.seq_len // 2
        seq_end = mid + self.seq_len // 2
        start_bin = seq_begin // self.bin_size
        stop_bin = seq_end // self.bin_size

        left_pad = np.where(start_bin < 0, -start_bin, 0)
        start_bin = np.maximum(start_bin, 0)
        limit = np.array([self.chrom_bins.get(name, np.iinfo(np.int64).max)
                          for name in self.chrom_names], dtype=np.int64)
        limit = limit[c["chrom"]] if len(table) else np.zeros(0, np.int64)
        right_pad = np.where(stop_bin > limit, stop_bin - limit, 0)
        stop_bin = np.minimum(stop_bin, limit)

        self.cols = {
            "start_bin": start_bin, "stop_bin": stop_bin,
            "left_pad": left_pad, "right_pad": right_pad,
            "enh_bin": enh_coord // self.bin_size,
            "prom_bin": tss_coord // self.bin_size,
            "enh_coord": enh_coord, "prom_coord": tss_coord,
            "label": c["label"].astype(np.int8),
            "dist": c["dist"].astype(np.float64),
        }
        # Logit of a bias model per row (training.debias); None when unused.
        self.bias = None
        self.cell_code = c["cell"].astype(np.int32)
        self.chrom_code = c["chrom"].astype(np.int32)
        self.assay_code = c["assay"].astype(np.int32)
        self.table = table

    def sample(self, idx: int) -> dict:
        """Everything known about one sample, as a dict."""
        c = self.cols
        return {"start_bin": int(c["start_bin"][idx]), "stop_bin": int(c["stop_bin"][idx]),
                "left_pad": int(c["left_pad"][idx]), "right_pad": int(c["right_pad"][idx]),
                "enh_bin": int(c["enh_bin"][idx]), "prom_bin": int(c["prom_bin"][idx]),
                "enh_coord": int(c["enh_coord"][idx]), "prom_coord": int(c["prom_coord"][idx]),
                "cell": self.cell_names[self.cell_code[idx]],
                "chrom": self.chrom_names[self.chrom_code[idx]],
                "label": int(c["label"][idx]), "dist": float(c["dist"][idx]),
                "assay": self.assay_names[self.assay_code[idx]]}

    @property
    def samples(self) -> _SampleView:
        return _SampleView(self)

    # ---------------------------------------------------------------
    def _find_cell_tracks(self, cell: str):
        """Check which track files of a cell exist; they are read later."""
        if cell not in self.feats_config:
            print(f"  WARNING: cell '{cell}' not in feats_config – skipping")
            self.track_paths[cell] = {}
            return
        found = {}
        for mark in self.feats_order:
            if mark not in self.feats_config[cell]:
                print(f"  WARNING: mark '{mark}' not in config for {cell}")
                continue
            pt_path = self.feats_config[cell][mark]
            if not os.path.exists(pt_path):
                print(f"  WARNING: {pt_path} not found")
                continue
            found[mark] = pt_path
        if not found:
            # Every chromatin channel would be zeros, and the run would report
            # numbers as if nothing were wrong. Stop instead.
            raise FileNotFoundError(
                f"no epigenetic track files could be loaded for '{cell}'. "
                f"Looked for the files named in the feature config under "
                f"{os.path.dirname(next(iter(self.feats_config[cell].values())))}")
        self.track_paths[cell] = found
        print(f"  Loaded {len(found)}/{self.num_feats} marks for {cell}")

    def _track(self, cell: str, mark: str) -> Optional[dict]:
        """The chromosome dict of one track, opened on first use here."""
        path = self.track_paths.get(cell, {}).get(mark)
        if path is None:
            return None
        key = (cell, mark)
        if key not in self._feats:
            self._feats[key] = load_track(path, self.mmap_tracks)
        return self._feats[key]

    def _quantile_maps(self, path: str, reference_cells: Optional[List[str]]):
        """Per cell and mark, (cell quantiles, reference quantiles) for np.interp."""
        with open(path) as handle:
            quantiles = json.load(handle)
        if not reference_cells:
            raise ValueError("track_quantiles needs quantile_reference_cells: the "
                             "cells whose mean quantiles define the reference")
        maps = {}
        for mark in self.feats_order:
            ref = [quantiles[c][mark] for c in reference_cells
                   if mark in quantiles.get(c, {})]
            if not ref:
                continue                     # this mark is left as it is
            reference = np.mean(np.array(ref, dtype=np.float64), axis=0)
            for cell in self.track_paths:
                q = quantiles.get(cell, {}).get(mark)
                if q is None:
                    if mark in self.track_paths[cell]:
                        raise KeyError(f"no quantiles for {cell} {mark} in {path}")
                    continue
                q = np.array(q, dtype=np.float64)
                # np.interp needs increasing points; ties (runs of zeros) are
                # broken by a negligible ramp.
                q = q + np.arange(len(q)) * 1e-9
                maps[(cell, mark)] = (q, reference)
        return maps

    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.cols["label"])

    def __getitem__(self, idx) -> dict:
        c = self.cols
        start_bin = int(c["start_bin"][idx])
        stop_bin = int(c["stop_bin"][idx])
        left_pad = int(c["left_pad"][idx])
        right_pad = int(c["right_pad"][idx])
        cell = self.cell_names[self.cell_code[idx]]
        chrom = self.chrom_names[self.chrom_code[idx]]

        # --- Epigenetic features: (num_feats, num_bins) ---
        feat_rows = []
        for mark in self.feats_order:
            track = self._track(cell, mark)
            if track is not None and chrom in track:
                row = track[chrom][start_bin:stop_bin]
                if self.quantile_maps is not None and (cell, mark) in self.quantile_maps:
                    q, reference = self.quantile_maps[(cell, mark)]
                    row = torch.from_numpy(np.interp(
                        row.numpy(), q, reference).astype(np.float32))
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
        enh_idx = int(c["enh_bin"][idx]) - start_bin + left_pad
        prom_idx = int(c["prom_bin"][idx]) - start_bin + left_pad

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

        if self.presence_channels:
            present = self.track_paths.get(cell, {})
            flags = torch.tensor([[1.0 if mark in present else 0.0]
                                  for mark in self.feats_order])
            ar = torch.cat([ar, flags.expand(-1, self.num_bins)], dim=0)

        # --- DNA sequences ---
        enh_coord = int(c["enh_coord"][idx])
        prom_coord = int(c["prom_coord"][idx])
        enh_seq = self._extract_seq(chrom, enh_coord, self.enhancer_window)
        prom_seq = self._extract_seq(chrom, prom_coord, self.promoter_window)

        # Distance: log-scaled (same formula as original Kansformer)
        dist_scaled = math.log2(1 + 500_000 / max(float(c["dist"][idx]), 1.0))

        extra = {}
        if self.bias is not None:
            extra["bias"] = torch.tensor([float(self.bias[idx])], dtype=torch.float)
        return {
            **extra,
            "epi": ar,                                           # (num_feats+1, num_bins)
            "enhancer_seq": enh_seq,                             # str
            "promoter_seq": prom_seq,                            # str
            "chrom": chrom,                                      # str
            "label": torch.tensor([int(c["label"][idx])], dtype=torch.float),
            "dist": torch.tensor([dist_scaled], dtype=torch.float),
            "enh_idx": torch.tensor([enh_idx], dtype=torch.float),
            "prom_idx": torch.tensor([prom_idx], dtype=torch.float),
        }

    def sequence_pair(self, idx) -> Tuple[str, str]:
        """Enhancer and promoter DNA of one sample, without the chromatin work.

        Fitting the sequence encoder needs only the DNA, so this skips the
        track slicing and normalisation that `__getitem__` does.
        """
        chrom = self.chrom_names[self.chrom_code[idx]]
        return (self._extract_seq(chrom, int(self.cols["enh_coord"][idx]),
                                  self.enhancer_window),
                self._extract_seq(chrom, int(self.cols["prom_coord"][idx]),
                                  self.promoter_window))

    def _extract_seq(self, chrom: str, center: int, window: int) -> str:
        """Extract a DNA sequence of `window` bp centred on `center`."""
        if self.ref_genome is None and self.ref_genome_path is not None:
            self._open_ref_genome()  # lazily reopen per-process (worker)
        return extract_sequence(self.ref_genome, chrom, center, window)

    # ---------------------------------------------------------------
    # Utility: split by chromosome for cross-validation
    # ---------------------------------------------------------------
    def get_chrom_groups(self) -> np.ndarray:
        """Return an array of chromosome strings for GroupKFold splitting."""
        return np.array(self.chrom_names)[self.chrom_code] if len(self) else np.array([])

    def get_cells(self) -> np.ndarray:
        """The cell line of every sample."""
        return np.array(self.cell_names)[self.cell_code] if len(self) else np.array([])

    def get_labels(self) -> np.ndarray:
        return self.cols["label"].astype(np.int64)

    def get_distances(self) -> np.ndarray:
        return self.cols["dist"].copy()


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

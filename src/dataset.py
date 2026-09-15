"""
PyTorch Dataset wrapper that applies POCD-ND encoding on-the-fly.

Can be used in two modes:
  1. Wrapping an EPIGenomicDataset (real genomic data)
  2. Standalone with synthetic data (for pipeline testing)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
# ---- Augmentation helpers ----
_COMP = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

def _reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA string."""
    return seq.translate(_COMP)[::-1]

class EPIDataset(Dataset):
    """
    Wrapper that applies POCD-ND encoding to raw DNA sequences.

    Each __getitem__ returns:
        seq      : Tensor (64, L)   – POCD-ND encoded sequence
        epi      : Tensor (n_feats+1, n_bins) – pos_enc + epigenetic features
        label    : Tensor (1,)
        dist     : Tensor (1,)
        enh_idx  : Tensor (1,)   – enhancer bin index (in 5000-bin space)
        prom_idx : Tensor (1,)   – promoter bin index (in 5000-bin space)
    """

    def __init__(
        self,
        config: dict,
        encoder,
        source_dataset=None,
        sequences=None,
        epi_features=None,
        labels=None,
        distances=None,
        num_synthetic: int = 0,
        concat_enh_prom: bool = True,
    ):
        """
        Parameters
        ----------
        config : dict from config.yaml
        encoder : fitted POCD_ND_Encoder
        source_dataset : EPIGenomicDataset instance (preferred for real data)
        sequences : list of DNA strings (alternative manual input)
        epi_features : array (N, n_feats, n_bins) or list of arrays
        labels, distances : arrays / lists
        num_synthetic : generate N synthetic samples for testing
        concat_enh_prom : if True, concatenate enhancer+promoter sequences
                          from source_dataset into a single input string
        """
        self.seq_len = config["data"]["sequence_length"]
        self.bins = config["data"]["epigenetic_bins"]
        self.n_epi = config["data"]["n_epigenetic_features"]
        self.encoder = encoder
        self.concat_enh_prom = concat_enh_prom

        # Augmentation settings (toggled per-phase in training loop)
        self.augment = False
        aug_cfg = config.get("augmentation", {})
        self.aug_rc_prob = aug_cfg.get("rc_prob", 0.5)
        self.aug_epi_noise_std = aug_cfg.get("epi_noise_std", 0.05)
        self.aug_shift_max = aug_cfg.get("shift_max_bins", 3)

        self.mode = None
        self._source = None
        self._manual_data = []

        if source_dataset is not None:
            self.mode = "pipeline"
            self._source = source_dataset
            print(f"EPIDataset: wrapping {len(self._source)} pipeline samples")

        elif sequences is not None:
            self.mode = "manual"
            self._build_manual(sequences, epi_features, labels, distances)

        elif num_synthetic > 0:
            self.mode = "synthetic"
            self._build_synthetic(num_synthetic)

        else:
            raise ValueError(
                "Provide source_dataset, sequences, or num_synthetic > 0"
            )

    # ------------------------------------------------------------------
    def _build_manual(self, sequences, epi_features, labels, distances):
        n = len(sequences)
        if epi_features is None:
            epi_features = [
                np.zeros((self.n_epi, self.bins), dtype=np.float32)
                for _ in range(n)
            ]
        if labels is None:
            labels = [0.0] * n
        if distances is None:
            distances = [0.0] * n

        for i in range(n):
            seq = self._pad_or_trim(sequences[i])
            epi = self._ensure_epi_shape(epi_features[i])
            self._manual_data.append(
                (seq, epi, float(labels[i]), float(distances[i]))
            )
        print(f"EPIDataset: {n} manual samples loaded")

    def _build_synthetic(self, n):
        bases = list("ACGT")
        for _ in range(n):
            seq = "".join(np.random.choice(bases, size=self.seq_len))
            epi = np.random.rand(self.n_epi, self.bins).astype(np.float32)
            label = float(np.random.randint(0, 2))
            dist = float(np.random.rand() * 10.0)
            self._manual_data.append((seq, epi, label, dist))
        print(f"EPIDataset: {n} synthetic samples generated")

    # ------------------------------------------------------------------
    def _pad_or_trim(self, seq: str) -> str:
        if len(seq) > self.seq_len:
            return seq[: self.seq_len]
        if len(seq) < self.seq_len:
            return seq + "N" * (self.seq_len - len(seq))
        return seq

    def _ensure_epi_shape(self, epi):
        """Ensure epi is float32 with shape (n_epi, bins)."""
        epi = np.asarray(epi, dtype=np.float32)
        if epi.shape == (self.n_epi, self.bins):
            return epi
        if epi.ndim == 2 and epi.shape == (self.bins, self.n_epi):
            return epi.T
        out = np.zeros((self.n_epi, self.bins), dtype=np.float32)
        r = min(epi.shape[0], self.n_epi)
        c = min(epi.shape[1] if epi.ndim > 1 else self.bins, self.bins)
        if epi.ndim == 1:
            out[0, : min(len(epi), self.bins)] = epi[: self.bins]
        else:
            out[:r, :c] = epi[:r, :c]
        return out

    # ------------------------------------------------------------------
    def __len__(self):
        if self.mode == "pipeline":
            return len(self._source)
        return len(self._manual_data)

    def __getitem__(self, idx):
        if self.mode == "pipeline":
            return self._getitem_pipeline(idx)
        return self._getitem_manual(idx)

    def _getitem_pipeline(self, idx):
        """Retrieve from EPIGenomicDataset and apply POCD-ND encoding."""
        raw = self._source[idx]
        epi = raw["epi"].clone()  # (num_feats, num_bins) from pipeline

        # Build DNA input string
        if self.concat_enh_prom:
            dna = raw["enhancer_seq"] + raw["promoter_seq"]
        else:
            dna = raw["enhancer_seq"]
        dna = self._pad_or_trim(dna)

        # --- Data augmentation (training only) ---
        if self.augment:
            # 1. Reverse complement with probability rc_prob
            if np.random.random() < self.aug_rc_prob:
                dna = _reverse_complement(dna)

            # 2. Small Gaussian noise on epigenetic features
            if self.aug_epi_noise_std > 0:
                epi = epi + torch.randn_like(epi) * self.aug_epi_noise_std
                epi = epi.clamp(min=0)  # epigenetic signals are non-negative

            # 3. Random circular shift of epigenetic bins (simulates coord jitter)
            if self.aug_shift_max > 0:
                shift = np.random.randint(-self.aug_shift_max, self.aug_shift_max + 1)
                if shift != 0:
                    epi = torch.roll(epi, shifts=shift, dims=1)

        seq_enc = self.encoder.transform(dna)  # Tensor (64, L)

        return {
            "seq": seq_enc,
            "epi": epi.float(),
            "label": raw["label"],
            "dist": raw["dist"],
            "enh_idx": raw["enh_idx"],
            "prom_idx": raw["prom_idx"],
        }

    def _getitem_manual(self, idx):
        seq, epi, label, dist = self._manual_data[idx]
        seq_enc = self.encoder.transform(seq)  # Tensor (64, L)
        return {
            "seq": seq_enc,
            "epi": torch.tensor(epi, dtype=torch.float32),
            "label": torch.tensor([label], dtype=torch.float32),
            "dist": torch.tensor([dist], dtype=torch.float32),
            "enh_idx": torch.tensor([2500.0], dtype=torch.float32),  # default midpoint
            "prom_idx": torch.tensor([2500.0], dtype=torch.float32),
        }
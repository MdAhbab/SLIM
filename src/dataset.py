"""
PyTorch Dataset wrapper that applies POCD-ND encoding on-the-fly.

Can be used in two modes:
  1. Wrapping an EPIGenomicDataset (real genomic data)
  2. Standalone with synthetic data (for pipeline testing)
"""

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
# ---- Augmentation helpers ----
_COMP = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

def _reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA string."""
    return seq.translate(_COMP)[::-1]

def shift_tracks(epi: torch.Tensor, shift: int, mode: str = "tracks",
                 n_tracks: int | None = None) -> torch.Tensor:
    """Shift chromatin bins by `shift` positions.

    mode "tracks" moves channels 1 onward (channels 1 to n_tracks when the
    presence channels follow them) and fills the vacated bins with zeros,
    leaving channel 0 (the position channel, aligned with the anchor indices)
    where it is. mode "all" is the original circular roll of every channel.
    """
    if shift == 0:
        return epi
    if mode == "all":
        return torch.roll(epi, shifts=shift, dims=1)
    last = epi.shape[0] if n_tracks is None else 1 + n_tracks
    out = epi.clone()
    tracks = epi[1:last]
    moved = torch.zeros_like(tracks)
    if shift > 0:
        moved[:, shift:] = tracks[:, :-shift]
    else:
        moved[:, :shift] = tracks[:, -shift:]
    out[1:last] = moved
    return out


class PairedRows(Dataset):
    """Two rows of a dataset per item, a positive and a negative, for a
    ranking loss (training.rank_weight)."""

    def __init__(self, dataset, pairs):
        self.dataset = dataset
        self.pairs = np.asarray(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        positive, negative = self.pairs[i]
        return {"pos": self.dataset[int(positive)], "neg": self.dataset[int(negative)]}


class CellBalancedSampler(Sampler):
    """A fresh sample of the training rows every epoch, shared fairly between
    cell lines.

    Each epoch draws `num_samples` rows without replacement. Every cell gets
    the same share, and a cell with fewer rows than its share gives all of
    them, its unused share going to the others. So the largest cells cannot
    dominate, and every row of a small cell is seen every epoch. Within a cell
    rows are drawn uniformly, which keeps its natural positive rate. Call
    `set_epoch` before each epoch; the draw depends only on the seed and the
    epoch, so a resumed run sees the same rows.
    """

    def __init__(self, cells, num_samples: int, seed: int = 0):
        cells = np.asarray(cells)
        self.groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
        sizes = np.array([len(g) for g in self.groups])
        self.num_samples = int(min(num_samples, sizes.sum()))
        self.quota = self.shares(sizes, self.num_samples)
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def shares(sizes: np.ndarray, total: int) -> np.ndarray:
        """Equal shares of `total`, capped at each size, the rest spread."""
        quota = np.zeros(len(sizes), dtype=np.int64)
        left, open_ = total, np.ones(len(sizes), dtype=bool)
        while left > 0 and open_.any():
            share = max(1, left // int(open_.sum()))
            for i in np.flatnonzero(open_):
                take = min(share, sizes[i] - quota[i], left)
                quota[i] += take
                left -= take
                if quota[i] >= sizes[i]:
                    open_[i] = False
                if left == 0:
                    break
        return quota

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        picked = [rng.choice(g, size=q, replace=False)
                  for g, q in zip(self.groups, self.quota) if q]
        order = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)
        return iter(rng.permutation(order).tolist())

    def __len__(self) -> int:
        return self.num_samples


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
        # How the two augmentations that move things around are applied.
        #   rc_mode "segment": each of the enhancer and promoter sequences is
        #       reverse-complemented in place, so the enhancer stays first,
        #       which is the layout the position-specific encoding and the
        #       encoder's segments assume. "concat" (the original rule)
        #       reverse-complements the joined string, which moves the
        #       promoter into the enhancer's half.
        #   shift_mode "tracks": the chromatin tracks move by up to
        #       shift_max_bins relative to the anchors, with zeros filling
        #       the gap. "all" (the original rule) rolls every channel,
        #       including the position channel, circularly, so the anchor
        #       indices no longer match the position channel.
        self.aug_rc_mode = aug_cfg.get("rc_mode", "concat")
        self.aug_shift_mode = aug_cfg.get("shift_mode", "all")
        # With data.presence_channels, the last n_tracks channels say which
        # tracks the cell has. Mark dropout hides whole tracks during training
        # (zeroes the track and its presence channel), so the model learns to
        # predict when a mark is missing, as it is for some cells.
        self.presence = config["data"].get("presence_channels", False)
        self.n_tracks = len(config["data"].get("feats_order") or [])
        self.aug_mark_dropout = aug_cfg.get("mark_dropout", 0.0)
        if self.aug_mark_dropout and not self.presence:
            raise ValueError("augmentation.mark_dropout needs data.presence_channels")
        expected = 1 + self.n_tracks * (2 if self.presence else 1)
        if self.n_tracks and self.n_epi != expected:
            raise ValueError(
                f"data.n_epigenetic_features is {self.n_epi} but the position "
                f"channel, {self.n_tracks} tracks"
                f"{' and their presence channels' if self.presence else ''} "
                f"make {expected}")
        if self.aug_rc_mode not in {"segment", "concat"}:
            raise ValueError("augmentation.rc_mode must be segment or concat")
        if self.aug_shift_mode not in {"tracks", "all"}:
            raise ValueError("augmentation.shift_mode must be tracks or all")
        # With the DNA branch switched off, the model never looks at the
        # encoded sequence, so skip the encoding work and send zeros.
        self.skip_sequence = config["data"].get("modalities", "all") == "epi"

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
        enhancer, promoter = raw["enhancer_seq"], raw["promoter_seq"]
        flip = self.augment and np.random.random() < self.aug_rc_prob

        # Build DNA input string
        if flip and self.aug_rc_mode == "segment":
            enhancer = _reverse_complement(enhancer)
            promoter = _reverse_complement(promoter)
        if self.concat_enh_prom:
            dna = enhancer + promoter
        else:
            dna = enhancer
        dna = self._pad_or_trim(dna)

        # --- Data augmentation (training only) ---
        if self.augment:
            # 1. Reverse complement with probability rc_prob
            if flip and self.aug_rc_mode == "concat":
                dna = _reverse_complement(dna)

            # 2. Small Gaussian noise on epigenetic features (not on the
            #    presence channels, which are exact)
            if self.aug_epi_noise_std > 0:
                last = 1 + self.n_tracks if self.presence else epi.shape[0]
                noisy = epi[:last] + torch.randn_like(epi[:last]) * self.aug_epi_noise_std
                epi[:last] = noisy.clamp(min=0)  # signals are non-negative

            # 3. Random shift of epigenetic bins (simulates coord jitter)
            if self.aug_shift_max > 0:
                shift = np.random.randint(-self.aug_shift_max, self.aug_shift_max + 1)
                if shift != 0:
                    epi = shift_tracks(epi, shift, self.aug_shift_mode,
                                       self.n_tracks if self.presence else None)

            # 4. Mark dropout: hide whole tracks, as for a cell that lacks them
            if self.aug_mark_dropout > 0:
                for j in np.flatnonzero(np.random.random(self.n_tracks)
                                        < self.aug_mark_dropout):
                    epi[1 + j] = 0.0
                    epi[1 + self.n_tracks + j] = 0.0

        if self.skip_sequence:
            seq_enc = torch.zeros(4 ** self.encoder.k, len(dna) - self.encoder.k + 1)
        else:
            seq_enc = self.encoder.transform(dna, chrom=raw.get("chrom"))  # (64, L)

        item = {
            "seq": seq_enc,
            "epi": epi.float(),
            "label": raw["label"],
            "dist": raw["dist"],
            "enh_idx": raw["enh_idx"],
            "prom_idx": raw["prom_idx"],
        }
        if "bias" in raw:
            item["bias"] = raw["bias"]
        return item

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
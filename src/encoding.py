import torch
import numpy as np
import itertools


class POCD_ND_Encoder:
    """
    Implements Position-Aware Oligonucleotide Composition Density with Negative Density (POCD-ND).
    Ref: Section 3.2.2 of Project Design.

    The densities are estimated from labelled sequences, so the encoder is a
    supervised feature. A sequence that was part of the fit is encoded with
    statistics that include its own label; see `CrossFitEncoder` for the way
    training avoids that.
    """
    def __init__(self, k=3):
        self.k = k
        self.kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.kmers)}
        self.num_kmers = len(self.kmers)
        self.pos_density_matrix = None
        self.neg_density_matrix = None

    def _compute_freq(self, sequences, seq_len):
        num_positions = seq_len - self.k + 1
        freq_matrix = np.zeros((self.num_kmers, num_positions))

        for seq in sequences:
            # Simple padding handling if seq is short
            loop_len = min(len(seq), seq_len) - self.k + 1
            for i in range(loop_len):
                kmer = seq[i : i + self.k]
                if kmer in self.kmer_to_idx:
                    freq_matrix[self.kmer_to_idx[kmer], i] += 1
        return freq_matrix

    def fit(self, pos_sequences, neg_sequences, seq_len):
        """Calculates global density matrices A^pos and A^neg."""
        print("Fitting POCD-ND Encoder...")
        A_pos = self._compute_freq(pos_sequences, seq_len)
        A_neg = self._compute_freq(neg_sequences, seq_len)

        epsilon = 1e-9
        # Normalize densities (Step 3)
        self.pos_density_matrix = (A_pos / (len(pos_sequences) + epsilon)) + epsilon
        self.neg_density_matrix = (A_neg / (len(neg_sequences) + epsilon)) + epsilon

    def transform(self, sequence, chrom=None):
        """Encodes a single sequence using Eq: Ratio * Min(Densities).

        `chrom` is accepted for interface parity with `CrossFitEncoder` and is
        not used here.
        """
        seq_len = len(sequence)
        num_positions = seq_len - self.k + 1

        # Calculate global POCD map
        ratio = self.pos_density_matrix[:, :num_positions] / self.neg_density_matrix[:, :num_positions]
        min_den = np.minimum(self.pos_density_matrix[:, :num_positions], self.neg_density_matrix[:, :num_positions])
        global_map = ratio * min_den

        # Mask: Only activate k-mers present in the sequence
        encoded = np.zeros_like(global_map)
        for i in range(num_positions):
            kmer = sequence[i : i + self.k]
            if kmer in self.kmer_to_idx:
                idx = self.kmer_to_idx[kmer]
                encoded[idx, i] = global_map[idx, i]

        return torch.FloatTensor(encoded)

    def log_ratio_score(self, sequence):
        """Naive-Bayes score sum_i log(p/n) at the k-mers the sequence holds.

        A plain function of the fitted densities, used by the leakage audit:
        if it separates the labels of the sequences the encoder was fitted on
        much better than those of held-out sequences, the densities carry the
        fitted labels into the encoding.
        """
        log_ratio = np.log(self.pos_density_matrix) - np.log(self.neg_density_matrix)
        total = 0.0
        for i in range(min(len(sequence), log_ratio.shape[1] + self.k - 1)
                       - self.k + 1):
            idx = self.kmer_to_idx.get(sequence[i:i + self.k])
            if idx is not None:
                total += log_ratio[idx, i]
        return total


def chromosome_half(chrom):
    """0 for odd-numbered chromosomes, 1 for even-numbered ones and the rest.

    chrX, chrY and anything unnumbered join the even half. The split only has
    to be fixed and label-free; odd and even numbers spread the large and the
    small chromosomes evenly between the two halves.
    """
    tail = str(chrom).replace("chr", "")
    if tail.isdigit() and int(tail) % 2 == 1:
        return 0
    return 1


class CrossFitEncoder:
    """Two POCD-ND encoders, each fitted on one half of the chromosomes.

    A sequence is always encoded by the encoder fitted on the OTHER half, so
    no training example is encoded with densities that counted its own label.
    Validation and test pairs are encoded the same way, by the half their
    chromosome belongs to, so every split sees the same kind of encoding.
    Pairs never cross chromosomes, so the rule is exact.
    """

    def __init__(self, k=3):
        self.k = k
        self.encoders = [POCD_ND_Encoder(k=k), POCD_ND_Encoder(k=k)]

    def fit(self, pos_by_half, neg_by_half, seq_len):
        """Fit encoder h on the positive and negative sequences of half h."""
        for half in (0, 1):
            if not pos_by_half[half] or not neg_by_half[half]:
                raise ValueError(
                    f"chromosome half {half} has no positive or no negative "
                    f"training sequences to fit the encoder on")
            self.encoders[half].fit(pos_by_half[half], neg_by_half[half], seq_len)

    def encoder_for(self, chrom):
        return self.encoders[1 - chromosome_half(chrom)]

    def transform(self, sequence, chrom=None):
        if chrom is None:
            raise ValueError("CrossFitEncoder.transform needs the chromosome")
        return self.encoder_for(chrom).transform(sequence)

    def log_ratio_score(self, sequence, chrom):
        return self.encoder_for(chrom).log_ratio_score(sequence)

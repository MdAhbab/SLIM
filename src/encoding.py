import torch
import numpy as np
import itertools

class POCD_ND_Encoder:
    """
    Implements Position-Aware Oligonucleotide Composition Density with Negative Density (POCD-ND).
    Ref: Section 3.2.2 of Project Design.
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

    def transform(self, sequence):
        """Encodes a single sequence using Eq: Ratio * Min(Densities)."""
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
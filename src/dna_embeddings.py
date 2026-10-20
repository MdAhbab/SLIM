"""
Embeddings of enhancers and promoters from a DNA language model, written by
scripts/data/dna_embeddings.py and read by the DNA branch when data.dna_input
is "embedding".

An element is named by its chromosome and position (the enhancer midpoint or
the gene start), packed into one int64 key. A pair's DNA input is the
enhancer's (segments, dim) array followed by the promoter's.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
_INDEX = {name: i for i, name in enumerate(CHROMS)}


def element_key(chroms, positions) -> np.ndarray:
    """Keys of elements given chromosome names and positions (arrays)."""
    codes = np.array([_INDEX[c] for c in chroms], dtype=np.int64)
    return (codes << 32) | np.asarray(positions, dtype=np.int64)


class ElementEmbeddings:
    """Lookup of pair DNA inputs in a folder of embeddings.

    The array is memory-mapped on first use in each process, so DataLoader
    workers share one copy and the object pickles cheaply.
    """

    def __init__(self, folder: str):
        self.folder = folder
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"no DNA embeddings in {folder} (meta.json missing)")
        with open(meta_path) as handle:
            self.meta = json.load(handle)
        self.segments = int(self.meta["segments"])
        self.dim = int(self.meta["dim"])
        self.keys = np.load(os.path.join(folder, "keys.npy"))
        self._array: Optional[np.ndarray] = None
        self.missing = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(os.path.join(self.folder, "embeddings.npy"), mmap_mode="r")
        return self._array

    def element(self, chrom: str, position: int) -> np.ndarray:
        """(segments, dim) of one element; zeros if it was not embedded."""
        if chrom in _INDEX:
            key = (_INDEX[chrom] << 32) | int(position)
            row = int(np.searchsorted(self.keys, key))
            if row < len(self.keys) and self.keys[row] == key:
                return np.asarray(self.array[row], dtype=np.float32)
        self.missing += 1
        return np.zeros((self.segments, self.dim), dtype=np.float32)

    def pair(self, chrom: str, enhancer: int, promoter: int) -> torch.Tensor:
        """(2 * segments, dim): the enhancer's parts, then the promoter's."""
        return torch.from_numpy(np.concatenate([self.element(chrom, enhancer),
                                                self.element(chrom, promoter)]))

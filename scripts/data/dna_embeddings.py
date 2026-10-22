"""
Embeddings of every enhancer and promoter from a pretrained DNA language
model, for the DNA branch (data.dna_input "embedding").

The elements are the enhancer midpoints and gene starts of every pair in the
given files: training, validation and test alike, since the embedding uses
the genome only, never a label or a cell. For each element the window of
--window bp centred on it is read from hg19 and passed through the model
(Nucleotide Transformer v2 by default) on both strands. The last layer's token
states are averaged into --segments equal parts along the window, and the two
strands are averaged (the reverse strand's parts in reverse order), so an
element gets a (segments, dim) array that does not depend on the strand.

    python scripts/data/dna_embeddings.py --pairs 'BENGI/*.tsv*' --genome hg19.fa \
        --out DIR [--shard 0 --shards 2]
    python scripts/data/dna_embeddings.py --pairs ... --out DIR --merge --shards 2

DIR then holds embeddings.npy (float16, elements x segments x dim), keys.npy
(int64 per element: chromosome index << 32 | position, sorted), meta.json
(model, window, segments, dim, chromosome names). src/dna_embeddings.py reads
them. With --shards each process embeds every n-th element and --merge joins
the parts. --model dummy uses a fixed random projection of the bases instead
of a language model, for tests.

With --extend OLD, only the elements missing from the table in OLD are
embedded, and --merge writes the union of OLD and them to --out (a new
folder; OLD is not changed). The model settings must match OLD's.

Nucleotide Transformer v2's own model code needs transformers 4.x (tested
with 4.46.3). Install it apart from the training environment, for example
  pip install --target /tmp/tf4 --no-deps transformers==4.46.3 \
      tokenizers==0.20.3 huggingface_hub==0.36.2
and run this script with PYTHONPATH=/tmp/tf4.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bengi import PairTable  # noqa: E402
from src.dna_embeddings import CHROMS, element_key  # noqa: E402

DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def elements(patterns) -> np.ndarray:
    """Sorted unique keys of every enhancer midpoint and gene start."""
    files = sorted({f for pattern in patterns for f in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"no pair files match {patterns}")
    table = PairTable.from_files(files)
    c = table.columns
    names = np.array(table.vocab["chrom"], dtype=object)[c["chrom"]]
    known = np.isin(names, CHROMS)
    enh = (c["enh_start"].astype(np.int64) + c["enh_end"]) // 2
    keys = np.concatenate([element_key(names[known], enh[known]),
                           element_key(names[known], c["tss_coord"][known].astype(np.int64))])
    print(f"{len(files)} files, {len(table):,} rows: ", end="")
    return np.unique(keys)


def window(genome, key: int, size: int) -> str:
    chrom, center = CHROMS[int(key) >> 32], int(key) & 0xFFFFFFFF
    start = max(0, center - size // 2)
    seq = str(genome[chrom][start:start + size]).upper()
    return seq + "N" * (size - len(seq))


def _unused_head_pruning(*args, **kwargs):
    raise NotImplementedError("attention-head pruning is not supported here")


class LanguageModel:
    """Last-layer token states of a Hugging Face DNA language model."""

    def __init__(self, name: str, device: str):
        import torch
        import transformers.pytorch_utils as utils
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        if not hasattr(utils, "find_pruneable_heads_and_indices"):
            # Nucleotide Transformer v2's own code imports this helper, which
            # newer transformers releases dropped; it is used only to prune
            # attention heads, never here.
            utils.find_pruneable_heads_and_indices = _unused_head_pruning
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            name, trust_remote_code=True).to(device).eval()
        if device.startswith("cuda"):
            self.model = self.model.half()
        self.dim = self.model.config.hidden_size

    def states(self, sequences):
        """(token states without the first special token, mask) as arrays."""
        torch = self.torch
        enc = self.tokenizer(sequences, return_tensors="pt", padding="longest")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(ids, attention_mask=mask, output_hidden_states=True)
        states = out.hidden_states[-1][:, 1:].float().cpu().numpy()
        return states, mask[:, 1:].cpu().numpy().astype(bool)


class DummyModel:
    """A fixed random projection of one-hot bases, 6 bp per token (tests)."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.weights = np.random.default_rng(0).normal(size=(5 * 6, dim))

    def states(self, sequences):
        lookup = {b: i for i, b in enumerate("ACGTN")}
        n = max(len(s) for s in sequences) // 6
        states = np.zeros((len(sequences), n, self.dim))
        for i, s in enumerate(sequences):
            for t in range(len(s) // 6):
                onehot = np.zeros(30)
                for j, base in enumerate(s[6 * t:6 * t + 6]):
                    onehot[5 * j + lookup.get(base, 4)] = 1
                states[i, t] = onehot @ self.weights
        return states, np.ones(states.shape[:2], dtype=bool)


def pool(states: np.ndarray, mask: np.ndarray, segments: int) -> np.ndarray:
    """Mean of the valid tokens in each of `segments` equal parts."""
    out = np.zeros((len(states), segments, states.shape[2]), dtype=np.float32)
    for i in range(len(states)):
        valid = states[i][mask[i]]
        for s, part in enumerate(np.array_split(valid, segments)):
            if len(part):
                out[i, s] = part.mean(axis=0)
    return out


def embed(model, sequences, segments: int) -> np.ndarray:
    """Strand-averaged pooled states, (len(sequences), segments, dim)."""
    reverse = [s.translate(COMPLEMENT)[::-1] for s in sequences]
    states, mask = model.states(list(sequences) + reverse)
    pooled = pool(states, mask, segments)
    n = len(sequences)
    return (pooled[:n] + pooled[n:, ::-1]) / 2


def extend_table(old: Path, new_keys, new_rows, meta, out: Path):
    """The union of the table in `old` and the new rows, sorted by key, written
    to <out>/embeddings.npy. Returns the keys and the written array."""
    old_meta = json.loads((old / "meta.json").read_text())
    for name in ("model", "window", "segments", "dim"):
        if old_meta.get(name) != meta.get(name):
            raise SystemExit(f"{name} differs from {old}: {meta.get(name)} vs {old_meta.get(name)}")
    old_keys = np.load(old / "keys.npy")
    old_rows = np.load(old / "embeddings.npy", mmap_mode="r")
    keys = np.union1d(old_keys, new_keys)
    table = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float16,
                                      shape=(len(keys),) + old_rows.shape[1:])
    for source_keys, rows in ((old_keys, old_rows), (new_keys, new_rows)):
        at = np.searchsorted(keys, source_keys)
        for s in range(0, len(at), 20000):
            table[at[s:s + 20000]] = rows[s:s + 20000]
    table.flush()
    return keys, table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", nargs="+", required=True, help="pair files (globs)")
    p.add_argument("--genome", help="hg19 FASTA (indexed by pyfaidx)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=3000)
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--extend", default=None,
                   help="an existing embedding folder: embed only what it lacks")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = elements(args.pairs)
    print(f"{len(keys):,} elements")
    if args.extend:
        if Path(args.extend).resolve() == out.resolve():
            raise SystemExit("--out must be a new folder, not the one --extend names")
        keys = keys[~np.isin(keys, np.load(Path(args.extend) / "keys.npy"))]
        print(f"  {len(keys):,} of them are not in {args.extend}")
    if args.merge:
        parts = [np.load(out / f"part{i}.npy", mmap_mode="r") for i in range(args.shards)]
        meta = json.loads((out / "part0.json").read_text())
        target = out / ("new_elements.npy" if args.extend else "embeddings.npy")
        merged = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                           shape=(len(keys),) + parts[0].shape[1:])
        for i, part in enumerate(parts):
            merged[i::args.shards] = part
        merged.flush()
        if args.extend:
            keys, merged = extend_table(Path(args.extend), keys, merged, meta, out)
            new_file = target
        np.save(out / "keys.npy", keys)
        meta.update(elements=len(keys), shards=args.shards)
        if args.extend:
            meta["extends"] = str(args.extend)
            del merged
            new_file.unlink()
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"Merged {args.shards} parts into {out / 'embeddings.npy'} "
              f"({len(keys):,} elements)")
        return

    import pyfaidx
    genome = pyfaidx.Fasta(args.genome)
    mine = keys[args.shard::args.shards]
    model = DummyModel() if args.model == "dummy" else LanguageModel(args.model, args.device)
    part = np.lib.format.open_memmap(out / f"part{args.shard}.npy", mode="w+",
                                     dtype=np.float16,
                                     shape=(len(mine), args.segments, model.dim))
    started = time.time()
    for b in range(0, len(mine), args.batch):
        batch = [window(genome, k, args.window) for k in mine[b:b + args.batch]]
        part[b:b + len(batch)] = embed(model, batch, args.segments)
        if (b // args.batch) % 200 == 0:
            done = b + len(batch)
            print(f"  shard {args.shard}: {done:,}/{len(mine):,} in "
                  f"{time.time() - started:.0f} s", flush=True)
    part.flush()
    meta = {"model": args.model, "window": args.window, "segments": args.segments,
            "dim": model.dim, "chroms": list(CHROMS)}
    (out / f"part{args.shard}.json").write_text(json.dumps(meta, indent=1))
    print(f"Shard {args.shard}: {len(mine):,} elements in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

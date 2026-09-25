"""
Test whether the survival gate writes biologically meaningful positions.

The gate keeps a small number of positions per layer and discards the rest.
If that selection reflects regulatory biology rather than an arbitrary
compression, the kept positions should carry more architectural and
open-chromatin signal than the positions that were dropped.

The test is paired and within-sample, which matters: comparing written bins in
one window against unwritten bins in a different window would be confounded by
how active each window is overall. Here every window is its own control, and a
permutation test reshuffles which bins count as written inside that same
window, so the null keeps each window's signal distribution intact.

It also reports where the writes go. The first 128 tokens carry DNA sequence
and the remaining 500 carry chromatin bins, so the split says which branch the
memory is summarising, and the distance from each write to the enhancer and
promoter anchors says whether the gate simply relocates to the two positions
the task already points at.

Enrichment alone cannot tell a selective gate from a collapsed one: a gate
that writes the same positions for every input can still land on
signal-rich bins. The collapse check therefore asks how often each position
is chosen across inputs, and how much any two inputs' written sets overlap
compared with sets of the same size placed at random. It also reports how
many memory slots are ever written.

Needs a trained checkpoint and the prepared dataset. One inference pass.

Usage:

    python scripts/memory_writelog.py --run results/ka/seed0
    python scripts/memory_writelog.py --run results/ga/seed0 --max-batches 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import EPIDataset
from src.epi_data_pipeline import EPIGenomicDataset
from src.slim_model import build_model

# Channel 0 of the chromatin tensor holds the position profile; the tracks
# follow in `data.feats_order` order.
TRACKS_TO_TEST = ("CTCF", "DNase", "H3K27ac", "H3K4me3", "H3K27me3")


def parse_args():
    p = argparse.ArgumentParser(
        description="Correlate memory writes with chromatin signal.")
    p.add_argument("--run", required=True,
                   help="run directory holding checkpoint.pt and encoder.pkl")
    p.add_argument("--config", default=None,
                   help="defaults to the config stored in the checkpoint")
    p.add_argument("--test-cells", nargs="+", default=["HMEC", "NHEK"])
    p.add_argument("--max-batches", type=int, default=40,
                   help="batches to trace; the statistics converge quickly")
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="defaults to <run>/writelog.json")
    return p.parse_args()


def collect_traces(model, loader, device, track_channel, max_batches):
    """One inference pass, recording which tokens each window wrote.

    Returns:
        masks: list of (n_epi_tokens,) boolean arrays, True where written.
        signals: track name -> list of (n_epi_tokens,) mean signal per token.
        branch: counts of writes landing in each branch.
        anchor_distance: mean tokens from a write to the nearer anchor.
        histogram: how often each token position was written.
        written_sets: per window, the distinct token positions written.
        slot_counts: how often each memory slot was written.
    """
    n_seq = model.n_seq_tokens
    n_epi = model.n_epi_tokens
    pool = model.epi_pool_factor

    masks = []
    signals = {name: [] for name in track_channel}
    branch = {"sequence": 0, "chromatin": 0}
    anchor_distance = []
    histogram = np.zeros(n_seq + n_epi, dtype=np.int64)
    written_sets = []
    slot_counts = np.zeros(model.memory_config.bin_slots, dtype=np.int64)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            seq = batch["seq"].float().to(device)
            epi = batch["epi"].float().to(device)
            enh_idx = batch["enh_idx"].float().to(device)
            prom_idx = batch["prom_idx"].float().to(device)

            out = model(seq, epi, enh_idx, prom_idx, return_trace=True)
            trace = out[-1]
            written = torch.cat(trace["tokens"], dim=1).cpu().numpy()  # (B, layers*k)
            slots = torch.cat(trace["slots"], dim=1).cpu().numpy().ravel()
            slot_counts += np.bincount(slots[slots >= 0],
                                       minlength=slot_counts.size)
            epi_np = epi.cpu().numpy()
            enh_bins = enh_idx.view(-1).cpu().numpy()
            prom_bins = prom_idx.view(-1).cpu().numpy()

            for i in range(written.shape[0]):
                tokens = np.unique(written[i])
                histogram[tokens] += 1
                written_sets.append(tokens)
                branch["sequence"] += int((tokens < n_seq).sum())
                branch["chromatin"] += int((tokens >= n_seq).sum())

                chrom_tokens = tokens[tokens >= n_seq] - n_seq
                # A window that wrote nothing, or everything, carries no
                # paired contrast.
                if chrom_tokens.size == 0 or chrom_tokens.size == n_epi:
                    continue

                mask = np.zeros(n_epi, dtype=bool)
                mask[chrom_tokens] = True
                masks.append(mask)

                for name, channel in track_channel.items():
                    row = epi_np[i, channel][: n_epi * pool]
                    signals[name].append(row.reshape(n_epi, pool).mean(axis=1))

                enh_token = min(int(enh_bins[i]) // pool, n_epi - 1)
                prom_token = min(int(prom_bins[i]) // pool, n_epi - 1)
                nearest = np.minimum(np.abs(chrom_tokens - enh_token),
                                     np.abs(chrom_tokens - prom_token))
                anchor_distance.append(float(nearest.mean()))

    return (masks, signals, branch, anchor_distance, histogram, written_sets,
            slot_counts)


def selection_collapse(written_sets, n_tokens, rng, pairs=20000):
    """How strongly the same positions win for every input.

    A gate that selects by content writes different positions for different
    inputs. A collapsed gate writes the same few positions whatever it sees.
    Three views:

      frequency  the fraction of windows that wrote each position; positions
                 written by at least 90 percent of windows are fixed winners.
      overlap    the Jaccard overlap between the written sets of random pairs
                 of windows, next to the overlap of random sets of the same
                 sizes. A ratio near one means input-driven selection; an
                 overlap near one means every input writes the same positions.
      entropy    of the pooled write histogram over all positions, normalised
                 so 1 is uniform use and 0 is a single position.
    """
    n = len(written_sets)
    frequency = np.zeros(n_tokens)
    for tokens in written_sets:
        frequency[tokens] += 1
    frequency /= max(n, 1)

    first = rng.integers(0, n, size=pairs)
    second = rng.integers(0, n, size=pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    observed, null = [], []
    for a, b in zip(first, second):
        set_a, set_b = written_sets[a], written_sets[b]
        observed.append(np.intersect1d(set_a, set_b).size
                        / np.union1d(set_a, set_b).size)
        rand_a = rng.choice(n_tokens, size=set_a.size, replace=False)
        rand_b = rng.choice(n_tokens, size=set_b.size, replace=False)
        null.append(np.intersect1d(rand_a, rand_b).size
                    / np.union1d(rand_a, rand_b).size)
    observed_mean = float(np.mean(observed)) if observed else float("nan")
    null_mean = float(np.mean(null)) if null else float("nan")

    pooled = frequency / frequency.sum() if frequency.sum() > 0 else frequency
    nonzero = pooled[pooled > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(n_tokens))
    top = np.argsort(frequency)[::-1][:10]
    return {
        "n_windows": n,
        "max_position_frequency": float(frequency.max()),
        "fixed_winners": int((frequency >= 0.9).sum()),
        "positions_ever_written": int((frequency > 0).sum()),
        "mean_pairwise_jaccard": observed_mean,
        "null_pairwise_jaccard": null_mean,
        "jaccard_ratio": (observed_mean / null_mean if null_mean > 0
                          else float("nan")),
        "identical_pair_fraction": (float(np.mean(np.array(observed) == 1.0))
                                    if observed else float("nan")),
        "normalised_entropy": entropy,
        "top_positions": [{"token": int(t), "frequency": float(frequency[t])}
                          for t in top],
        # Flag stated in advance rather than tuned: half of any two inputs'
        # writes shared means the gate is mostly ignoring its input.
        "collapse_warning": bool(observed_mean >= 0.5),
    }


def permutation_test(signals, masks, rng, permutations):
    """Paired permutation test on written minus unwritten signal.

    The statistic is the mean over windows of (mean signal in written bins)
    minus (mean signal in unwritten bins). Each permutation redraws the
    written set uniformly inside the same window, so window-level differences
    in overall signal cannot produce an effect.
    """
    written = np.array([row[m].mean() for row, m in zip(signals, masks)])
    unwritten = np.array([row[~m].mean() for row, m in zip(signals, masks)])
    observed = float(np.mean(written - unwritten))

    n_tokens = signals[0].shape[0]
    sizes = [int(m.sum()) for m in masks]
    at_least_as_extreme = 0
    for _ in range(permutations):
        diffs = np.empty(len(signals))
        for i, row in enumerate(signals):
            picked = rng.choice(n_tokens, size=sizes[i], replace=False)
            shuffled = np.zeros(n_tokens, dtype=bool)
            shuffled[picked] = True
            diffs[i] = row[shuffled].mean() - row[~shuffled].mean()
        if abs(float(diffs.mean())) >= abs(observed):
            at_least_as_extreme += 1

    return {
        "written_mean": float(written.mean()),
        "unwritten_mean": float(unwritten.mean()),
        "difference": observed,
        "ratio": (float(written.mean() / unwritten.mean())
                  if unwritten.mean() > 0 else float("nan")),
        "p_value": (at_least_as_extreme + 1) / (permutations + 1),
        "n_windows": len(masks),
        "permutations": permutations,
    }


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.run, "checkpoint.pt")
    encoder_path = os.path.join(args.run, "encoder.pkl")
    for path in (checkpoint_path, encoder_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. This analysis needs a trained run; "
                f"train one with scripts/train.py first.")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = load_config(args.config) if args.config else state["config"]

    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if not hasattr(model, "encoder"):
        raise SystemExit(
            "This run is the global-attention baseline, which has no survival "
            "gate and so no write log. Point --run at a SLIM run.")

    with open(encoder_path, "rb") as handle:
        encoder = pickle.load(handle)

    bengi_dir = config["paths"].get("bengi_dir", "./data/BENGI")
    files = [f for f in sorted(glob.glob(os.path.join(bengi_dir, "*.tsv*")))
             if os.path.basename(f).split(".")[0] in args.test_cells]
    if not files:
        raise FileNotFoundError(f"no BENGI files for {args.test_cells}")

    genomic = EPIGenomicDataset(
        bengi_paths=files,
        feats_config_path=config["paths"]["feats_config"],
        feats_order=config["data"].get("feats_order"),
        seq_len=config["data"].get("seq_len_bp", 2_500_000),
        bin_size=config["data"].get("bin_size", 500),
        enhancer_window=config["data"].get("enhancer_window", 3000),
        promoter_window=config["data"].get("promoter_window", 3000),
        ref_genome_path=config["paths"].get("ref_genome") or None,
    )
    dataset = EPIDataset(config, encoder, source_dataset=genomic)
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"],
                        shuffle=False, num_workers=0)

    feats_order = list(config["data"].get("feats_order", []))
    track_channel = {name: i + 1 for i, name in enumerate(feats_order)
                     if name in TRACKS_TO_TEST}
    n_seq, n_epi = model.n_seq_tokens, model.n_epi_tokens
    pool = model.epi_pool_factor

    print("=" * 74)
    print(f"  Memory write log: {args.run}")
    print("=" * 74)
    print(f"  Device: {device}")
    print(f"  Tokens: {n_seq} sequence, {n_epi} chromatin")
    print(f"  Tracing up to {args.max_batches} batches")

    (masks, signals, branch, anchor_distance, histogram, written_sets,
     slot_counts) = collect_traces(model, loader, device, track_channel,
                                   args.max_batches)

    total = branch["sequence"] + branch["chromatin"]
    if total == 0 or not masks:
        raise SystemExit("No writes were traced; check the run directory.")

    seq_share = branch["sequence"] / total
    seq_token_share = n_seq / (n_seq + n_epi)
    print("\n--- Where the writes go ---")
    print(f"  sequence branch : {branch['sequence']:>8,} writes "
          f"({100 * seq_share:5.1f} percent), from "
          f"{100 * seq_token_share:.1f} percent of tokens")
    print(f"  chromatin branch: {branch['chromatin']:>8,} writes "
          f"({100 * (1 - seq_share):5.1f} percent), from "
          f"{100 * (1 - seq_token_share):.1f} percent of tokens")
    enrichment = seq_share / seq_token_share if seq_token_share else float("nan")
    print(f"  The sequence branch is written "
          f"{enrichment:.2f} times its token share.")

    mean_distance = float(np.mean(anchor_distance)) if anchor_distance else None
    if mean_distance is not None:
        print(f"\n  Mean distance from a chromatin write to the nearer anchor: "
              f"{mean_distance:.1f} tokens "
              f"({mean_distance * pool * 500 / 1000:.0f} kb).")
        print(f"  Writes scattered uniformly would average about "
              f"{n_epi / 4:.0f} tokens.")

    rng = np.random.default_rng(args.seed)
    collapse = selection_collapse(written_sets, n_seq + n_epi, rng)
    print("\n--- Does the gate write the same positions for every input? ---")
    print(f"  positions ever written        : "
          f"{collapse['positions_ever_written']} of {n_seq + n_epi}")
    print(f"  most-written position         : in "
          f"{100 * collapse['max_position_frequency']:.1f} percent of windows")
    print(f"  positions in >=90% of windows : {collapse['fixed_winners']}")
    print(f"  pairwise overlap (Jaccard)    : "
          f"{collapse['mean_pairwise_jaccard']:.3f}, against "
          f"{collapse['null_pairwise_jaccard']:.3f} for random sets "
          f"({collapse['jaccard_ratio']:.1f}x)")
    print(f"  identical written sets        : "
          f"{100 * collapse['identical_pair_fraction']:.1f} percent of pairs")
    print(f"  write entropy (1 = uniform)   : "
          f"{collapse['normalised_entropy']:.3f}")
    if collapse["collapse_warning"]:
        print("  WARNING: any two inputs share at least half their writes, so "
              "the\n  gate is largely ignoring its input.")
    used = int((slot_counts > 0).sum())
    print(f"  memory slots ever written     : {used} of {slot_counts.size}")

    print("\n--- Chromatin signal in written versus unwritten bins ---")
    print("  Paired inside each window; the permutation test reshuffles which")
    print("  bins count as written within that same window.\n")
    print(f"  {'track':<10}{'written':>10}{'unwritten':>11}"
          f"{'difference':>12}{'ratio':>8}{'p':>10}")

    tracks = {}
    for name in track_channel:
        rows = signals[name]
        if not rows:
            continue
        stats = permutation_test(rows, masks, rng, args.permutations)
        tracks[name] = stats
        print(f"  {name:<10}{stats['written_mean']:>10.4f}"
              f"{stats['unwritten_mean']:>11.4f}{stats['difference']:>12.4f}"
              f"{stats['ratio']:>8.3f}{stats['p_value']:>10.4f}")

    results = {
        "run": args.run,
        "test_cells": args.test_cells,
        "n_windows": len(masks),
        "writes_total": int(total),
        "writes_sequence_branch": int(branch["sequence"]),
        "writes_chromatin_branch": int(branch["chromatin"]),
        "sequence_token_fraction": seq_token_share,
        "sequence_write_fraction": float(seq_share),
        "sequence_branch_enrichment": float(enrichment),
        "mean_tokens_to_nearest_anchor": mean_distance,
        "tracks": tracks,
        "collapse": collapse,
        "slot_write_counts": slot_counts.tolist(),
        "token_histogram": histogram.tolist(),
    }
    out_path = args.out or os.path.join(args.run, "writelog.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\n  Windows analysed: {len(masks):,}")
    print(f"  Written to {out_path}")
    print("\n  A ratio above one with a small p-value means the gate prefers "
          "bins")
    print("  carrying that mark. Report the effect size, not only the p-value.")


if __name__ == "__main__":
    main()

"""
Measure inference latency, training step time and peak GPU memory.

The manuscript argues that local windowed attention plus a bounded memory is
cheaper than global self-attention. That argument is analytical until somebody
measures it. This script measures it, on one machine, with identical batch
shapes for every model, using random tensors of the true input sizes so no
dataset is needed.

Latency is timed with CUDA events rather than wall-clock, because kernel
launches are asynchronous and a plain timer measures the queue, not the work.
Peak memory comes from the allocator's own high-water mark, reset before each
model.

Usage:

    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --batch-size 64 --repeats 30
    python scripts/benchmark_efficiency.py --variants baseline KA --no-train
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.slim_model import build_model

CONFIG_FOR = {
    "baseline": "configs/baseline.yaml",
    "A": "configs/slim_a.yaml",
    "KA": "configs/slim_ka.yaml",
    "GA": "configs/slim_ga.yaml",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark latency, throughput and peak memory.")
    p.add_argument("--variants", nargs="+",
                   default=["baseline", "A", "KA", "GA"],
                   choices=list(CONFIG_FOR))
    p.add_argument("--batch-size", type=int, default=None,
                   help="defaults to the batch size in each config")
    p.add_argument("--repeats", type=int, default=30,
                   help="timed iterations after the warm-up")
    p.add_argument("--warmup", type=int, default=5,
                   help="untimed iterations, which let cuDNN autotune settle")
    p.add_argument("--device", default=None)
    p.add_argument("--no-train", action="store_true",
                   help="skip the training-step measurement")
    p.add_argument("--out", default="results/efficiency.json")
    return p.parse_args()


def make_batch(config, batch_size, device):
    """Random inputs with the exact shapes the real pipeline produces."""
    seq_len = config["data"]["sequence_length"]
    n_bins = config["data"]["epigenetic_bins"]
    n_feats = config["data"]["n_epigenetic_features"]
    seq = torch.randn(batch_size, 64, seq_len - 2, device=device)
    epi = torch.rand(batch_size, n_feats, n_bins, device=device)
    enh = torch.randint(0, n_bins, (batch_size, 1), device=device).float()
    prom = torch.randint(0, n_bins, (batch_size, 1), device=device).float()
    return seq, epi, enh, prom


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_inference(model, batch, device, repeats, warmup):
    """Median and spread of forward-pass latency, in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(*batch)
        synchronize(device)

        timings = []
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                start.record()
                model(*batch)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
        else:
            for _ in range(repeats):
                began = time.perf_counter()
                model(*batch)
                timings.append((time.perf_counter() - began) * 1000.0)
    return np.asarray(timings)


def time_training_step(model, batch, device, repeats, warmup):
    """Median time of a full training step: forward, backward and update."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seq, epi, enh, prom = batch
    labels = torch.randint(0, 2, (seq.shape[0], 1), device=device).float()
    distances = torch.randn(seq.shape[0], 1, device=device)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(seq, epi, enh, prom)
        cls_out, reg_out, attention = out[0], out[1], out[2]
        loss = nn.functional.binary_cross_entropy_with_logits(cls_out, labels)
        loss = loss + nn.functional.mse_loss(reg_out, distances)
        loss = loss + 0.1 * model.attention_penalty(attention)
        if len(out) == 4:
            loss = loss + model.memory_auxiliary_loss(out[3])
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    synchronize(device)

    timings = []
    for _ in range(repeats):
        began = time.perf_counter()
        one_step()
        synchronize(device)
        timings.append((time.perf_counter() - began) * 1000.0)
    return np.asarray(timings)


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("  Efficiency benchmark")
    print("=" * 78)
    print(f"  Device:   {device}")
    if device.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("  Running on CPU. Latencies are not comparable to GPU numbers,")
        print("  but the ordering between models is still informative.")
    print(f"  torch:    {torch.__version__}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Repeats:  {args.repeats} timed, {args.warmup} warm-up")

    results = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "models": {},
    }

    for variant in args.variants:
        config = load_config(CONFIG_FOR[variant])
        batch_size = args.batch_size or config["data"]["batch_size"]

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = build_model(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        batch = make_batch(config, batch_size, device)

        infer = time_inference(model, batch, device, args.repeats, args.warmup)
        peak_infer = (torch.cuda.max_memory_allocated() / 2 ** 20
                      if device.type == "cuda" else None)

        entry = {
            "parameters": int(params),
            "batch_size": int(batch_size),
            "inference_ms_median": float(np.median(infer)),
            "inference_ms_p05": float(np.percentile(infer, 5)),
            "inference_ms_p95": float(np.percentile(infer, 95)),
            "inference_pairs_per_second":
                float(batch_size / (np.median(infer) / 1000.0)),
            "peak_memory_mib_inference": peak_infer,
        }

        if not args.no_train:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            step = time_training_step(model, batch, device,
                                      max(args.repeats // 3, 3), args.warmup)
            entry["train_step_ms_median"] = float(np.median(step))
            entry["train_pairs_per_second"] = float(
                batch_size / (np.median(step) / 1000.0))
            entry["peak_memory_mib_training"] = (
                torch.cuda.max_memory_allocated() / 2 ** 20
                if device.type == "cuda" else None)

        results["models"][variant] = entry
        del model, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\n  {variant}")
        print(f"    parameters              {params:>12,}")
        print(f"    inference latency       "
              f"{entry['inference_ms_median']:>9.2f} ms per batch of "
              f"{batch_size}")
        print(f"    inference throughput    "
              f"{entry['inference_pairs_per_second']:>9.1f} pairs per second")
        if entry["peak_memory_mib_inference"] is not None:
            print(f"    peak memory, inference  "
                  f"{entry['peak_memory_mib_inference']:>9.1f} MiB")
        if "train_step_ms_median" in entry:
            print(f"    training step           "
                  f"{entry['train_step_ms_median']:>9.2f} ms")
            if entry["peak_memory_mib_training"] is not None:
                print(f"    peak memory, training   "
                      f"{entry['peak_memory_mib_training']:>9.1f} MiB")

    if "baseline" in results["models"]:
        base = results["models"]["baseline"]["inference_ms_median"]
        print("\n  Inference latency relative to the baseline "
              "(below 1.00 is faster)")
        for variant, entry in results["models"].items():
            ratio = entry["inference_ms_median"] / base
            entry["inference_latency_vs_baseline"] = ratio
            print(f"    {variant:<10} {ratio:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()

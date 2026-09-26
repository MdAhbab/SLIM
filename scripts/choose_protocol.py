"""
Choose the training protocol from the pilot runs, on validation only.

Four pilot runs of variant KA at seed 0 differ in two choices:

    training assays   Hi-C rows only, or every BENGI assay (Hi-C and both
                      ChIA-PET assays, repeats merged)
    DNA branch        on (modalities all) or off (modalities epi)

Every pilot is validated on the same rows: Hi-C pairs of the training cell
lines on the held-out chromosomes, the assay the test cell lines use. This
script reads each pilot's validation AUPR of its selected checkpoint, picks
the highest, and writes the winning choices to a protocol file that every
later training stage applies with `--protocol`.

Test scores are never read here, so the choice cannot be tuned to the test
set. The pilots that lose are kept as ablations.

Usage:

    python scripts/choose_protocol.py --pilots results/pilot/* \\
        --out results/protocol.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Pick the protocol from the pilots.")
    p.add_argument("--pilots", nargs="+", required=True,
                   help="pilot run directories")
    p.add_argument("--metric", default="val_aupr",
                   choices=["val_aupr", "val_auroc"])
    p.add_argument("--keep-dna", action="store_true",
                   help="only consider pilots with the DNA branch on")
    p.add_argument("--out", default="results/protocol.json")
    return p.parse_args()


def describe(run_dir):
    """The two protocol choices and the validation scores of one pilot."""
    npz = np.load(os.path.join(run_dir, "eval_results.npz"), allow_pickle=True)
    with open(os.path.join(run_dir, "config_snapshot.yaml")) as handle:
        snapshot = yaml.safe_load(handle)
    train_assays = snapshot["data"].get("train_assays") or None
    return {
        "run": run_dir,
        "train_assays": train_assays,
        "modalities": snapshot["data"].get("modalities", "all"),
        "valid_assays": snapshot["training"].get("valid_assays") or None,
        "val_rows": int(npz["val_rows"]) if "val_rows" in npz.files else None,
        "val_aupr": float(npz["val_aupr"]),
        "val_auroc": float(npz["val_auroc"]),
    }


def main():
    args = parse_args()
    pilots = [describe(d) for d in args.pilots
              if os.path.exists(os.path.join(d, "eval_results.npz"))]
    if len(pilots) < 2:
        raise SystemExit(f"need at least two finished pilots, found {len(pilots)}")

    # The comparison is only fair if every pilot was scored on the same rows.
    validation = {(json.dumps(p["valid_assays"]), p["val_rows"]) for p in pilots}
    if len(validation) != 1:
        raise SystemExit(
            "the pilots were validated on different rows, so their scores "
            f"cannot be compared: {sorted(validation)}")

    candidates = [p for p in pilots if not args.keep_dna or p["modalities"] != "epi"]
    if not candidates:
        raise SystemExit("no pilot is left to choose from")
    best = max(candidates, key=lambda p: p[args.metric])

    print(f"Pilots, scored on {pilots[0]['val_rows']:,} validation rows "
          f"(assays {pilots[0]['valid_assays'] or 'all'}):")
    for p in sorted(pilots, key=lambda p: -p[args.metric]):
        mark = "  <- chosen" if p is best else ""
        print(f"  {os.path.basename(p['run']):<16} training assays "
              f"{str(p['train_assays'] or 'all'):<10} inputs {p['modalities']:<4} "
              f"val AUPR {p['val_aupr']:.4f}  val AUROC {p['val_auroc']:.4f}{mark}")

    protocol = {
        "train_assays": best["train_assays"],
        "modalities": best["modalities"],
        "chosen": os.path.basename(best["run"]),
        "metric": args.metric,
        "keep_dna": args.keep_dna,
        "pilots": {os.path.basename(p["run"]): {
            k: p[k] for k in ("train_assays", "modalities", "val_aupr", "val_auroc")}
            for p in pilots},
        "note": "chosen on validation only; test scores were not read",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(protocol, handle, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

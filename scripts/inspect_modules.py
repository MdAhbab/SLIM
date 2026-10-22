"""
Read what the Phase 2 modules (src/pair_modules.py) learned, for the figures.

For one run folder written by scripts/train.py and one test file in the BENGI
format, it writes to --out:

  modules.json   with model.chromatin_stem "groups": the mean gate weight of
                 each chromatin group over the tokens of a row, averaged over
                 all rows, the positives and the negatives, and the same at
                 the enhancer and promoter tokens; with model.extrusion: the
                 learned processivity of each head and the tracks it reads.
  modules.npz    the per-row group weights with the labels and predictions,
                 and, for --examples positive and negative pairs, the per-bin
                 hazards of both extrusion legs (learned, and with the base
                 rate and the anchor margin), the anchors and logQ.

The model is rebuilt as scripts/evaluate_external.py rebuilds it, and nothing
about the run changes.

    python scripts/inspect_modules.py --run results/phase2_c/seed0 \
        --test datasets/processed/BENGI_extra_test/HUVEC.HiC_rao2014-Benchmark.v3.tsv.gz \
        --feats-config datasets/processed/features_new_cells.json \
        --out results/inspect/phase2_c_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import EPIDataset  # noqa: E402
from src.epi_data_pipeline import EPIGenomicDataset, dataset_options  # noqa: E402
from src.pair_modules import GroupedChromatinStem, track_names  # noqa: E402
from evaluate_external import load_run, merged_feature_config  # noqa: E402


def class_means(values: np.ndarray, labels: np.ndarray, names) -> dict:
    """Mean of each column over all rows, the positives and the negatives."""
    out = {}
    for key, rows in (("all", slice(None)), ("positives", labels == 1),
                      ("negatives", labels == 0)):
        chosen = values[rows]
        out[key] = ({n: float(v) for n, v in zip(names, chosen.mean(axis=0))}
                    if len(chosen) else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--feats-config", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="score a fixed random sample of this many rows (0: all)")
    parser.add_argument("--examples", type=int, default=4,
                        help="positive and negative pairs whose hazard maps are kept")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dna-embeddings", default=None)
    parser.add_argument("--track-quantiles", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    run = Path(args.run)
    config, model, encoder = load_run(run, device)
    stem = model.epi_cnn if isinstance(model.epi_cnn, GroupedChromatinStem) else None
    extrusion = (model.pair_features.extrusion
                 if getattr(model, "pair_features", None) is not None else None)
    if stem is None and extrusion is None:
        raise SystemExit(f"{run} was trained without the chromatin groups and the "
                         f"extrusion module; there is nothing to inspect")

    feats = merged_feature_config(args.feats_config)
    options = dataset_options(config)
    options.update(feats_config_path=feats,
                   ref_genome_path=config["paths"].get("ref_genome") or None)
    if options["track_quantiles"] and args.track_quantiles:
        options["track_quantiles"] = args.track_quantiles
    genomic = EPIGenomicDataset(bengi_paths=[args.test], **options)
    if args.dna_embeddings:
        config["paths"]["dna_embeddings"] = args.dna_embeddings
    dataset = EPIDataset(config, encoder, source_dataset=genomic)
    rows = np.arange(len(dataset))
    if args.max_rows and args.max_rows < len(rows):
        rows = np.sort(np.random.default_rng(0).choice(rows, args.max_rows, replace=False))
    loader = DataLoader(Subset(dataset, rows.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)

    alpha, alpha_anchor, labels, preds = [], [], [], []
    examples = {1: [], 0: []}
    done = 0
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].float().to(device)
            epi = batch["epi"].float().to(device)
            enh = batch["enh_idx"].float().to(device)
            prom = batch["prom_idx"].float().to(device)
            label = batch["label"].view(-1).numpy()
            out = model(seq, epi, enh, prom)
            pred = torch.sigmoid(out[0].float()).view(-1).cpu().numpy()
            labels.append(label)
            preds.append(pred)
            if stem is not None:
                a = stem.last_alpha.float()                          # (B, G, T)
                alpha.append(a.mean(dim=2).cpu().numpy())
                tokens = a.shape[2]
                rows_b = torch.arange(a.shape[0], device=a.device)
                at = lambda idx: a[rows_b, :, (idx.view(-1).long()
                                               // model.epi_pool_factor).clamp(0, tokens - 1)]
                alpha_anchor.append(((at(enh) + at(prom)) / 2).cpu().numpy())
            if extrusion is not None and min(len(v) for v in examples.values()) < args.examples:
                terms = model.pair_features.extrusion.terms(
                    epi * model.epi_channel_mask.to(epi.dtype), enh, prom)
                for i, y in enumerate(label.astype(int)):
                    if len(examples[y]) < args.examples:
                        examples[y].append(dict(
                            row=int(rows[done + i]), label=y, prediction=float(pred[i]),
                            **{k: terms[k][i].cpu().numpy() for k in (
                                "hazard_left", "hazard_right", "learned_left",
                                "learned_right", "left", "right", "log_q")}))
            done += len(label)
    os.unlink(feats)
    labels, preds = np.concatenate(labels), np.concatenate(preds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run": str(run), "test": str(args.test), "rows": int(len(labels)),
               "positives": int(labels.sum())}
    arrays = {"rows": rows, "labels": labels, "predictions": preds}
    if stem is not None:
        alpha, alpha_anchor = np.concatenate(alpha), np.concatenate(alpha_anchor)
        summary["groups"] = {
            "names": stem.group_names,
            "channels": dict(zip(stem.group_names, stem.group_channels)),
            "alpha_mean": class_means(alpha, labels, stem.group_names),
            "alpha_at_anchors": class_means(alpha_anchor, labels, stem.group_names)}
        arrays.update(alpha=alpha, alpha_at_anchors=alpha_anchor,
                      group_names=np.array(stem.group_names))
    if extrusion is not None:
        names = track_names(config)
        summary["extrusion"] = {
            "barriers": extrusion.barriers,
            "tracks": [names[c - 1] for c in extrusion.channels.tolist()],
            "processivity_kb": extrusion.processivity_kb.cpu().tolist()}
        kept = examples[1] + examples[0]
        for key in ("row", "label", "prediction", "left", "right", "log_q",
                    "hazard_left", "hazard_right", "learned_left", "learned_right"):
            arrays[f"example_{key}"] = np.stack([np.asarray(e[key]) for e in kept]) \
                if kept else np.zeros(0)
    (out_dir / "modules.json").write_text(json.dumps(summary, indent=1))
    np.savez(out_dir / "modules.npz", **arrays)
    print(json.dumps(summary, indent=1))
    print(f"Saved {out_dir / 'modules.json'} and {out_dir / 'modules.npz'}")


if __name__ == "__main__":
    main()

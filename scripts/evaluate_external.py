"""
Score saved models on test files they were not evaluated on.

Each run directory written by scripts/train.py holds the selected weights
(checkpoint.pt), the fitted sequence encoder (encoder.pkl) and the merged
configuration (config_snapshot.yaml). This script rebuilds the model from
those, reads any test file in the BENGI format (a new cell line, the CRISPR
benchmark, an extra label set), and scores every row. No training happens and
nothing about the run changes, so it is safe to run on a laptop processor.

The feature configuration must name tracks for every cell in the test files.
Several can be given; they are merged, each file's relative track names being
resolved against its own directory.

For each run and test file it writes the per-row predictions next to the
labels and coordinates (<out>/<test name>/<variant>_seed<N>.npz), and one
summary over all of them (<out>/summary.json). When an ABC score file exists
for a test file (convert_eval.py abc), ABC is scored on the same rows.

Usage:
    python scripts/evaluate_external.py \
        --runs results/ga/seed0 results/baseline/seed0 \
        --tests datasets/processed/BENGI_extra_test/HUVEC.HiC_rao2014-Benchmark.v3.tsv.gz \
        --feats-config datasets/from_training_pc/genomic_data/CTCF_DNase_6histone_local.500.json \
                       datasets/processed/features_new_cells.json \
        --out results/external
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_track_location  # noqa: E402
from src.dataset import EPIDataset  # noqa: E402
from src.epi_data_pipeline import EPIGenomicDataset, dataset_options  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.slim_model import build_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import evaluate_loader  # noqa: E402


def merged_feature_config(paths) -> str:
    """One feature configuration with absolute track paths, in a temp file."""
    merged = {}
    for path in paths:
        config = json.loads(Path(path).read_text())
        location = resolve_track_location(path, config.pop("_location", None))
        for cell, marks in config.items():
            if not isinstance(marks, dict):
                continue
            if cell in merged:
                raise ValueError(f"cell {cell} is named in more than one feature file")
            merged[cell] = {m: f if os.path.isabs(f) else os.path.join(location, f)
                            for m, f in marks.items()}
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(merged, handle)
    handle.close()
    return handle.name


def load_run(run: Path, device):
    config = yaml.safe_load((run / "config_snapshot.yaml").read_text())
    config.pop("_run", None)
    state = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
    model = build_model(state.get("config", config)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    with open(run / "encoder.pkl", "rb") as handle:
        encoder = pickle.load(handle)
    return config, model, encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--feats-config", nargs="+", required=True)
    parser.add_argument("--abc-dir", default="datasets/processed/eval/abc")
    parser.add_argument("--out", default="results/external")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true",
                        help="mixed precision on a GPU, as train.py scores its test set")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--track-quantiles", default=None,
                        help="quantile file on this machine, for runs trained "
                             "with data.track_norm cell_quantile")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    feats = merged_feature_config(args.feats_config)
    out = Path(args.out)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for run in map(Path, args.runs):
        config, model, encoder = load_run(run, device)
        variant = str(config["model"].get("variant", run.parent.name))
        seed = run.name.replace("seed", "")
        tag = f"{run.parent.name}_seed{seed}"
        for test in map(Path, args.tests):
            name = test.name.replace("-Benchmark.v3.tsv.gz", "")
            dest = out / name / f"{tag}.npz"
            if dest.exists():
                print(f"  {tag} on {name}: done before, skipped")
                continue
            options = dataset_options(config)
            options.update(feats_config_path=feats,
                           ref_genome_path=config["paths"].get("ref_genome") or None)
            if options["track_quantiles"] and args.track_quantiles:
                options["track_quantiles"] = args.track_quantiles
            genomic = EPIGenomicDataset(bengi_paths=[str(test)], **options)
            dataset = EPIDataset(config, encoder, source_dataset=genomic)
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
            metrics = evaluate_loader(model, loader, device, config, use_amp=args.amp)
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, predictions=metrics["predictions"], labels=metrics["labels"],
                     cell=genomic.get_cells(), chrom=genomic.get_chrom_groups(),
                     enh_coord=genomic.cols["enh_coord"],
                     prom_coord=genomic.cols["prom_coord"],
                     variant=np.array(variant), run=np.array(str(run)))
            entry = {k: metrics.get(k) for k in ("auroc", "aupr", "f1", "mcc")}
            entry.update(rows=len(genomic), positives=int(metrics["labels"].sum()))
            summary.setdefault(name, {})[tag] = entry
            abc = Path(args.abc_dir) / test.name.replace(".tsv.gz", ".abc.npy")
            if abc.exists() and "ABC" not in summary[name]:
                scores = np.load(abc)
                m = compute_all_metrics(metrics["labels"], scores)
                summary[name]["ABC"] = {k: m.get(k) for k in ("auroc", "aupr")}
            print(f"  {tag:22s} {name:40s} AUROC {entry['auroc']:.3f}  "
                  f"AUPR {entry['aupr']:.3f}  ({entry['rows']} rows)", flush=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=1))
    os.unlink(feats)
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()

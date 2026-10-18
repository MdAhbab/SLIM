"""
Bin chromatin tracks into the 500 bp tensor files the training code reads.

This reproduces the conversion behind the existing track files (TransEPI,
src/prepare_bw_signals.py and src/prepare_bed_signals.py), so a new cell's
tracks are on the same scale as the six BENGI cell lines':

  bigWig       the mean signal of each 500 bp bin (pyBigWig, exact), with the
               chromosome's tail in one last partial bin, then arcsinh.
               Missing values are 0.
  narrowPeak   column 7 (signalValue), arcsinh, spread over the bins a peak
               covers in proportion to the overlap; a bin keeps the largest
               value of any peak covering it.

Each chromosome of hg19 gets ceil(length / 500) + 1 bins; chrM, chrMT and chrY
stay zero. A file is a dict from chromosome name to a float32 tensor, written
with torch.save, named like the existing ones:
  bigWig_<CELL>_<MARK>.500bp.pt and narrowPeak_<CELL>_CTCF.500bp.pt

GRCh38 inputs are converted first: peaks with liftOver, bigWigs with CrossMap.

Usage:
    python scripts/data/prepare_tracks.py --groups tracks-check tracks-core
    python scripts/data/prepare_tracks.py --groups all --workers 6 --stage ~/stage
    python scripts/data/prepare_tracks.py --cells HUVEC HepG2
    python scripts/data/prepare_tracks.py --groups tracks-pretrain --download --delete-raw

The output goes to <out>/tracks (the GM12878 check files to <out>/check), and
<out>/features_new_cells.json lists every cell whose eight files are complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_manifest, load_resolved, open_text, out_root, raw_root  # noqa: E402

HG19 = {
    "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
    "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
    "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
    "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
    "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    "chrM": 16569, "chrMT": 16569,
}
SKIP = {"chrM", "chrMT", "chrY"}
BIN = 500


def empty_signals(bin_size: int = BIN) -> Dict[str, np.ndarray]:
    return {chrom: np.zeros(math.ceil(length / bin_size) + 1, dtype=np.float64)
            for chrom, length in HG19.items()}


def bin_bigwig(path: str, bin_size: int = BIN) -> Dict[str, np.ndarray]:
    import pyBigWig
    signals = empty_signals(bin_size)
    with pyBigWig.open(path) as bw:
        present = bw.chroms()
        for chrom, length in HG19.items():
            if chrom in SKIP or chrom not in present:
                continue
            length = min(length, present[chrom])
            full = length // bin_size
            values = bw.stats(chrom, 0, full * bin_size, nBins=full, exact=True)
            values = np.array([0.0 if v is None else v for v in values])
            signals[chrom][:full] = np.arcsinh(values)
            if length > full * bin_size:
                tail = bw.stats(chrom, full * bin_size, length, exact=True)[0]
                signals[chrom][full] = np.arcsinh(tail or 0.0)
    return signals


def bin_peaks(path: str, bin_size: int = BIN) -> Dict[str, np.ndarray]:
    signals = empty_signals(bin_size)
    for line in open_text(path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 7 or fields[0] not in signals:
            continue
        chrom, start, stop = fields[0], int(fields[1]), int(fields[2])
        value = float(np.arcsinh(float(fields[6])))
        row = signals[chrom]
        for i in range(start // bin_size, (stop - 1) // bin_size + 1):
            if i >= len(row):
                break
            overlap = min((i + 1) * bin_size, stop) - max(i * bin_size, start)
            row[i] = max(overlap * value / bin_size, row[i])
    return signals


def save(signals: Dict[str, np.ndarray], dest: Path):
    import torch
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save({c: torch.as_tensor(v, dtype=torch.float) for c, v in signals.items()}, tmp)
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
def lift_peaks(path: Path, workdir: Path) -> Path:
    """GRCh38 narrowPeak to hg19, keeping column 7."""
    from liftover import lift_intervals
    rows = [l.rstrip("\n").split("\t") for l in open_text(path) if l.strip()]
    lifted, counts = lift_intervals([(r[0], int(r[1]), int(r[2])) for r in rows])
    out = workdir / (path.name.replace(".GRCh38", ".lifted-hg19").replace(".gz", ""))
    with open(out, "w") as handle:
        for i, (chrom, start, end) in sorted(lifted.items()):
            handle.write("\t".join([chrom, str(start), str(end)] + rows[i][3:]) + "\n")
    print(f"    lifted {path.name}: {counts}")
    return out


def crossmap_bigwig(path: Path, workdir: Path) -> Path:
    from liftover import default_chain
    prefix = workdir / path.stem
    subprocess.run(["CrossMap", "bigwig", str(default_chain()), str(path), str(prefix)],
                   check=True, capture_output=True)
    candidates = sorted(workdir.glob(path.stem + "*.bw"))
    if not candidates:
        raise FileNotFoundError(f"CrossMap wrote no bigWig for {path}")
    return candidates[0]


def convert(job: Dict) -> Dict:
    """One input file to one track file. Runs in a worker process."""
    src, dest = Path(job["src"]), Path(job["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists() and job.get("url"):
        from common import download
        fetched = download(job["url"], src, size=job.get("size"), md5=job.get("md5"))
        if fetched["status"] == "failed":
            return {**job, "status": "failed", "error": fetched["error"]}
    staged = None
    try:
        if job.get("stage"):
            staged = Path(job["stage"]) / src.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, staged)
        read = staged or src
        with tempfile.TemporaryDirectory(dir=job.get("stage") or dest.parent) as tmp:
            tmp = Path(tmp)
            if job["kind"] == "peaks":
                path = lift_peaks(read, tmp) if job.get("assembly") == "GRCh38" else read
                signals = bin_peaks(str(path))
            else:
                path = crossmap_bigwig(read, tmp) if job.get("assembly") == "GRCh38" else read
                signals = bin_bigwig(str(path))
            save(signals, dest)
        values = np.concatenate([v for c, v in signals.items() if c not in SKIP])
        summary = {"median": float(np.median(values)),
                   "p99": float(np.percentile(values, 99)),
                   "max": float(values.max()),
                   "nonzero": float((values > 0).mean())}
        if job.get("delete_raw"):
            src.unlink()
        return {**job, "status": "ok", "summary": summary}
    except Exception as error:
        return {**job, "status": "failed", "error": repr(error)}
    finally:
        if staged is not None and staged.exists():
            staged.unlink()


def plan_jobs(resolved: Dict, cells: List[str], groups: List[str],
              raw: Path, out: Path, delete_raw: bool, stage: str = None) -> List[Dict]:
    manifest = load_manifest()
    jobs = []
    for record in resolved["files"]:
        if record["kind"] not in ("track", "ctcf", "hct116"):
            continue
        cell = record["cell"]
        if cells and cell not in cells:
            continue
        if groups and record["group"] not in groups:
            continue
        src = raw / record["dest"]
        mark = record.get("mark", "CTCF") if record["kind"] != "ctcf" else "CTCF"
        peaks = record["kind"] == "ctcf" or src.name.endswith(".bed.gz")
        prefix = "narrowPeak" if peaks else "bigWig"
        folder = "check" if manifest["cells"].get(cell, {}).get("existing") else "tracks"
        dest = out / folder / f"{prefix}_{cell}_{mark}.500bp.pt"
        jobs.append({"cell": cell, "mark": mark, "src": str(src), "dest": str(dest),
                     "kind": "peaks" if peaks else "bigwig",
                     "assembly": record.get("assembly", "hg19"),
                     "url": record["url"], "size": record.get("size"),
                     "md5": record.get("md5"), "delete_raw": delete_raw,
                     "stage": stage})
    return jobs


def write_feature_config(out: Path, marks: List[str]) -> Dict[str, List[str]]:
    """List every cell in <out>/tracks whose eight files exist."""
    tracks = out / "tracks"
    config, incomplete = {}, {}
    files = {p.name for p in tracks.glob("*.500bp.pt")} if tracks.exists() else set()
    cells = sorted({name.split("_", 1)[1].rsplit("_", 1)[0] for name in files})
    for cell in cells:
        entry = {}
        for mark in marks:
            prefix = "narrowPeak" if mark == "CTCF" else "bigWig"
            name = f"{prefix}_{cell}_{mark}.500bp.pt"
            if name in files:
                entry[mark] = name
        if len(entry) == len(marks):
            config[cell] = entry
        else:
            incomplete[cell] = [m for m in marks if m not in entry]
    payload = {"_location": "./processed"}
    payload.update(config)
    (out / "features_new_cells.json").write_text(json.dumps(payload, indent=2))
    return incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=[],
                        help="download groups to convert (tracks-check, "
                             "tracks-core, tracks-pretrain, eval for HCT116), or all")
    parser.add_argument("--cells", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--delete-raw", action="store_true",
                        help="remove each downloaded file once converted")
    parser.add_argument("--download", action="store_true",
                        help="download each missing input just before converting "
                             "it; with --delete-raw only a few inputs are on disk "
                             "at any time (for notebooks with small scratch disks)")
    parser.add_argument("--stage", help="copy each input to this local folder before "
                             "reading it; much faster when the inputs sit on a "
                             "slow or network drive (for WSL, a folder under ~)")
    parser.add_argument("--redo", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    groups = [] if "all" in args.groups else args.groups
    raw, out = raw_root(), out_root()
    jobs = plan_jobs(load_resolved(), args.cells, groups, raw, out, args.delete_raw,
                     args.stage)
    pending = [j for j in jobs if args.redo or not Path(j["dest"]).exists()]
    missing = [j for j in pending if not Path(j["src"]).exists()]
    ready = pending if args.download else [j for j in pending if Path(j["src"]).exists()]
    print(f"  {len(jobs)} track files planned, {len(jobs) - len(pending)} done, "
          f"{len(ready)} to convert, {len(missing)} not downloaded yet")

    log = out / "prepare_tracks_log.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool, open(log, "a") as handle:
        futures = [pool.submit(convert, job) for job in ready]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if result["status"] != "ok":
                failed += 1
                print(f"  [{n}/{len(ready)}] FAILED {result['cell']} {result['mark']}: "
                      f"{result['error']}")
            else:
                s = result["summary"]
                print(f"  [{n}/{len(ready)}] {result['cell']:20s} {result['mark']:9s} "
                      f"median {s['median']:.2f}  p99 {s['p99']:.2f}  "
                      f"nonzero {s['nonzero']:.2f}", flush=True)

    incomplete = write_feature_config(out, load_manifest()["marks"])
    if incomplete:
        print("  cells still missing tracks:")
        for cell, marks in incomplete.items():
            print(f"   - {cell}: {', '.join(marks)}")
    print(f"  feature list: {out / 'features_new_cells.json'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

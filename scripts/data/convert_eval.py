"""
Convert the evaluation sources into the benchmark format or into score tables.

  crispr        ENCODE-rE2G CRISPR benchmark (GRCh38). Valid tested
                element-gene pairs are lifted to hg19 and written as benchmark
                rows, label = Regulated, one file per cell type.
  targetfinder  TargetFinder pairs (hg19). Promoters are regions without gene
                IDs, so the gene field holds the promoter's coordinates.
  cboep2        CBOEP2 pair sets (hg19), every folder, every cell.
  abc           ABC predictions (hg19), scored on the rows of every test file:
                for each row, the largest ABC score of an element overlapping
                the enhancer and predicted to regulate the same gene, else 0
                (the published table only lists scores of at least 0.015).
  eqtl          eQTL Catalogue credible sets (GRCh38). Variants with posterior
                inclusion probability above 0.5, lifted to hg19.

Benchmark rows use the same ten columns as BENGI (see build_pairs.py), so
scripts that read test files read these unchanged. Everything is written
under <out>/eval/.

Usage:
    python scripts/data/convert_eval.py all
    python scripts/data/convert_eval.py abc --list-celltypes
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import load_manifest, open_text, out_root, raw_root  # noqa: E402
from build_pairs import gene_symbols  # noqa: E402
from src.bengi import read_rows  # noqa: E402

CELL_NAMES = {"HeLa-S3": "HeLa", "IMR-90": "IMR90"}


def row(label, chrom, enh_start, enh_end, enh_id, cell, tss, gene, transcript="",
        strand="+") -> str:
    centre = (enh_start + enh_end) / 2
    prom_start, prom_end = (tss - 1499, tss + 501) if strand == "+" else (tss - 499, tss + 1501)
    return "\t".join([
        str(int(label)), f"{abs(tss + 0.5 - centre):.1f}", chrom, str(enh_start),
        str(enh_end), f"{chrom}:{enh_start}-{enh_end}|{cell}|{enh_id}", chrom,
        str(prom_start), str(prom_end),
        f"{chrom}:{tss}-{tss + 1}|{cell}|{gene}|{transcript}|{strand}"])


def write(lines: List[str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.writelines(line + "\n" for line in lines)
    return len(lines)


# ---------------------------------------------------------------------------
def crispr(raw: Path, out: Path) -> Dict:
    from liftover import lift_intervals
    report = {}
    rows: Dict[str, List[str]] = defaultdict(list)
    for path in sorted((raw / "eval" / "crispr").glob("*.tsv.gz")):
        records = list(csv.DictReader(open_text(path), delimiter="\t"))
        records = [r for r in records if r.get("ValidConnection", "TRUE") == "TRUE"]
        elements = [(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in records]
        tss = [(r["chrTSS"], int(r["startTSS"]), int(r["startTSS"]) + 1) for r in records]
        lifted, counts = lift_intervals(elements + tss)
        n = len(records)
        kept = 0
        for i, r in enumerate(records):
            e, t = lifted.get(i), lifted.get(i + n)
            if e is None or t is None or e[0] != t[0]:
                continue
            cell = CELL_NAMES.get(r["CellType"], r["CellType"])
            rows[cell].append(row(r["Regulated"] == "TRUE", e[0], e[1], e[2],
                                  f"CRISPR{i}", cell, t[1], r["measuredGeneEnsemblId"]))
            kept += 1
        report[path.name] = {"valid_pairs": n, "lifted": kept, "liftover": counts}
    for cell, lines in rows.items():
        name = f"{cell}.CRISPR-Benchmark.v3.tsv.gz"
        positives = sum(line.startswith("1\t") for line in lines)
        report[name] = {"rows": write(lines, out / "eval" / "crispr" / name),
                        "positives": positives}
    return report


def targetfinder(raw: Path, out: Path) -> Dict:
    report = {}
    for path in sorted((raw / "eval" / "targetfinder").rglob("pairs.csv")):
        lines = []
        for r in csv.DictReader(open(path)):
            cell = CELL_NAMES.get(r["enhancer_name"].split("|")[0],
                                  r["enhancer_name"].split("|")[0])
            p_start, p_end = int(r["promoter_start"]), int(r["promoter_end"])
            tss = (p_start + p_end) // 2
            gene = f"TFP_{r['promoter_chrom']}_{p_start}_{p_end}"
            lines.append(row(r["label"] == "1", r["enhancer_chrom"],
                             int(r["enhancer_start"]), int(r["enhancer_end"]),
                             f"TFE_{r['enhancer_chrom']}_{r['enhancer_start']}",
                             cell, tss, gene))
        name = f"{cell}.TargetFinder-Benchmark.v3.tsv.gz"
        report[name] = {"rows": write(lines, out / "eval" / "targetfinder" / name),
                        "positives": sum(l.startswith("1\t") for l in lines)}
    return report


def cboep2(raw: Path, out: Path) -> Dict:
    report = {}
    for path in sorted((raw / "eval" / "cboep2").rglob("*.csv")):
        if path.stem.endswith("_prev"):
            continue
        lines = []
        for r in csv.DictReader(open(path)):
            cell = CELL_NAMES.get(r["enhancer_name"].split("|")[0],
                                  r["enhancer_name"].split("|")[0])
            tss = int(r["promoter_name"].split(":")[-1].split("-")[0])
            lines.append(row(r["label"] == "1", r["enhancer_chrom"],
                             int(r["enhancer_start"]), int(r["enhancer_end"]),
                             f"CBE_{r['enhancer_chrom']}_{r['enhancer_start']}",
                             cell, tss, f"CBP_{r['promoter_chrom']}_{tss}"))
        folder = path.parent.name
        name = f"{cell}.{folder.strip('_')}-Benchmark.v3.tsv.gz"
        report[f"{folder}/{name}"] = {
            "rows": write(lines, out / "eval" / "cboep2" / folder / name),
            "positives": sum(l.startswith("1\t") for l in lines)}
    return report


def abc(raw: Path, out: Path, list_only: bool) -> Dict:
    path = raw / "eval" / "abc" / "AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz"
    mapping = load_manifest().get("abc_celltypes", {})
    if list_only:
        counts = defaultdict(int)
        for line in open_text(path):
            counts[line.rstrip("\n").rsplit("\t", 1)[-1]] += 1
        for name, n in sorted(counts.items()):
            print(f"  {n:9d}  {name}")
        return {}
    wanted = {v: k for k, v in mapping.items()}
    elements: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in handle:
            f = line.rstrip("\n").split("\t")
            cell = wanted.get(f[col["CellType"]])
            if cell is None:
                continue
            elements[cell][(f[col["chr"]], f[col["TargetGene"]])].append(
                (int(f[col["start"]]), int(f[col["end"]]), float(f[col["ABC.Score"]])))
    symbol_of = {}
    annotation = raw / "reference" / "transepi" / "data" / "annotation" / "gencode.v19.tss.bed"
    for symbol, genes in gene_symbols(annotation).items():
        for gene in genes:
            symbol_of[gene.split(".")[0]] = symbol
    report = {}
    test_files = sorted((raw / "reference" / "transepi" / "data" / "BENGI").glob("*.tsv.gz"))
    test_files += sorted((out / "BENGI_extra_test").glob("*.tsv.gz"))
    for path in test_files:
        cell = path.name.split(".")[0]
        if cell not in elements:
            continue
        rows = read_rows([str(path)])
        scores = np.zeros(len(rows))
        for i, r in enumerate(rows):
            for start, end, score in elements[cell].get(
                    (r["chrom"], symbol_of.get(r["gene"], "")), []):
                if start < r["enh_end"] and end > r["enh_start"]:
                    scores[i] = max(scores[i], score)
        dest = out / "eval" / "abc" / (path.name.replace(".tsv.gz", ".abc.npy"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, scores)
        report[path.name] = {"rows": len(rows), "scored": int((scores > 0).sum()),
                             "abc_celltype": mapping[cell]}
    return report


def eqtl(raw: Path, out: Path, min_pip: float = 0.5) -> Dict:
    from liftover import lift_intervals
    report = {}
    for path in sorted((raw / "eval" / "eqtl").glob("*.credible_sets.tsv.gz")):
        records = [r for r in csv.DictReader(open_text(path), delimiter="\t")
                   if float(r["pip"]) > min_pip]
        positions = []
        for r in records:
            chrom, pos = r["variant"].split("_")[:2]
            positions.append((chrom, int(pos) - 1, int(pos)))
        lifted, counts = lift_intervals(positions)
        lines = ["chrom\tpos\tgene_id\tpip\tcs_id\tvariant"]
        for i, r in enumerate(records):
            if i in lifted:
                chrom, start, end = lifted[i]
                lines.append(f"{chrom}\t{end}\t{r['gene_id']}\t{r['pip']}\t"
                             f"{r['cs_id']}\t{r['variant']}")
        name = path.name.replace(".credible_sets.tsv.gz", ".pip0.5.hg19.tsv.gz")
        dest = out / "eval" / "eqtl" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt") as handle:
            handle.write("\n".join(lines) + "\n")
        report[name] = {"variants": len(records), "lifted": len(lines) - 1,
                        "liftover": counts}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["crispr", "targetfinder", "cboep2", "abc",
                                           "eqtl", "all"])
    parser.add_argument("--list-celltypes", action="store_true",
                        help="abc only: print the biosample names in the ABC table")
    args = parser.parse_args()
    raw, out = raw_root(), out_root()
    jobs = ["crispr", "targetfinder", "cboep2", "abc", "eqtl"] if args.source == "all" \
        else [args.source]
    log = out / "eval" / "convert_report.json"
    report = json.loads(log.read_text()) if log.exists() else {}
    for job in jobs:
        print(f"  {job} ...", flush=True)
        if job == "abc":
            result = abc(raw, out, args.list_celltypes)
        else:
            result = globals()[job](raw, out)
        report[job] = result
        for name, entry in result.items():
            print(f"    {name}: {entry}")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(report, indent=1))
    print(f"  report: {log}")


if __name__ == "__main__":
    main()

"""
Build enhancer-gene benchmark files from chromatin loops, the way BENGI did.

A Python 3 port of BENGI's Generate-Benchmark.sh (Moore et al. 2020,
github.com/weng-lab/BENGI, Scripts/Generate-Benchmark) for 3D-contact data,
in the "all pairs, natural ratio" form (the v3 files used here):

  1. Enhancers are the cell type's enhancer-like cCREs (Registry v1, hg19),
     minus any within 2 kb of a GENCODE v19 transcript start (TSS).
  2. A loop links an enhancer to a gene when one anchor overlaps the enhancer
     and the other overlaps a TSS of that gene (+/- 2 kb). Loops whose anchors
     lie on different chromosomes are ignored.
  3. The distance of a link is the smallest distance between the enhancer's
     edges and any TSS of the gene. The cutoff is the 95th percentile of the
     link distances, and links beyond it are dropped.
  4. For every enhancer with at least one link, each other gene with a TSS
     within the cutoff of the enhancer is a negative.

Each pair is then written as one row in the ten-column format src/bengi.py
reads, at the gene's TSS closest to the enhancer centre:

  label distance chrom enh_start enh_end chrom:start-end|CELL|cCRE
  chrom prom_start prom_end chrom:tss-tss+1|CELL|ENSG|ENST|strand

Sources (all coordinates hg19 by the time they reach `build`):
  rao        Rao et al. 2014 HiCCUPS loop lists
  bedpe      ENCODE loop calls (lifted from GRCh38 when needed)
  javierre   Javierre et al. 2016 peak matrix, CHiCAGO score >= 5 per cell
  jung       Jung et al. 2019 significant promoter capture Hi-C interactions,
             promoter named by gene symbol

Usage:
    python scripts/data/build_pairs.py --all
    python scripts/data/build_pairs.py --cells HUVEC --sources rao
    python scripts/data/build_pairs.py --check-hmec   # rebuild BENGI's HMEC set
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_manifest, load_resolved, open_text, out_root, raw_root  # noqa: E402

ELS_COLOUR = "255,205,0"      # enhancer-like state in the cell-type cCRE files
PROXIMAL = 2000               # BENGI's TSS +/- 2 kb ("TSS.2019.4K")
JAVIERRE_MIN_SCORE = 5.0

Interval = Tuple[str, int, int]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
class TSSIndex:
    """GENCODE v19 transcript starts, searchable by position per chromosome."""

    def __init__(self, path: Path):
        rows = defaultdict(list)
        for line in open_text(path):
            chrom, start, _end, transcript, _dot, strand, gene = \
                line.rstrip("\n").split("\t")[:7]
            rows[chrom].append((int(start), gene, transcript, strand))
        self.pos: Dict[str, np.ndarray] = {}
        self.rec: Dict[str, List[Tuple[int, str, str, str]]] = {}
        self.by_gene: Dict[str, List[Tuple[str, int, str, str]]] = defaultdict(list)
        for chrom, items in rows.items():
            items.sort()
            self.rec[chrom] = items
            self.pos[chrom] = np.array([p for p, *_ in items])
            for p, gene, transcript, strand in items:
                self.by_gene[gene].append((chrom, p, transcript, strand))

    def genes_in(self, chrom: str, start: int, end: int) -> Set[str]:
        """Genes with a TSS in [start, end]."""
        if chrom not in self.pos:
            return set()
        lo = np.searchsorted(self.pos[chrom], start, "left")
        hi = np.searchsorted(self.pos[chrom], end, "right")
        return {self.rec[chrom][i][1] for i in range(lo, hi)}

    def closest(self, gene: str, centre: float) -> Tuple[str, int, str, str]:
        return min(self.by_gene[gene], key=lambda t: (abs(t[1] - centre), t[1]))

    def distance(self, gene: str, start: int, end: int) -> int:
        return min(min(abs(start - p), abs(end - p)) for _, p, _, _ in self.by_gene[gene])


class EnhancerIndex:
    """Enhancer intervals per chromosome, with overlap queries."""

    def __init__(self, enhancers: Iterable[Tuple[str, int, int, str]]):
        rows = defaultdict(list)
        for chrom, start, end, name in enhancers:
            rows[chrom].append((start, end, name))
        self.rows = {c: sorted(v) for c, v in rows.items()}
        self.starts = {c: [r[0] for r in v] for c, v in self.rows.items()}
        self.maxlen = {c: max(r[1] - r[0] for r in v) for c, v in self.rows.items()}
        self.coords = {r[2]: (c, r[0], r[1]) for c, v in self.rows.items() for r in v}

    def overlapping(self, chrom: str, start: int, end: int) -> List[str]:
        if chrom not in self.rows:
            return []
        rows, starts = self.rows[chrom], self.starts[chrom]
        lo = bisect.bisect_left(starts, start - self.maxlen[chrom])
        hi = bisect.bisect_left(starts, end)
        return [r[2] for r in rows[lo:hi] if r[1] > start and r[0] < end]


def cell_enhancers(ccre_path: Path, tss: TSSIndex) -> EnhancerIndex:
    """Enhancer-like cCREs of one cell type, minus the TSS-proximal ones."""
    kept = []
    for line in open_text(ccre_path):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9 or fields[8] != ELS_COLOUR:
            continue
        chrom, start, end, name = fields[0], int(fields[1]), int(fields[2]), fields[3]
        if tss.genes_in(chrom, start - PROXIMAL, end + PROXIMAL - 1):
            continue
        kept.append((chrom, start, end, name))
    return EnhancerIndex(kept)


# ---------------------------------------------------------------------------
# The BENGI procedure
# ---------------------------------------------------------------------------
def link_pairs(loops: Sequence[Tuple[Interval, Interval]], enh: EnhancerIndex,
               tss: TSSIndex) -> List[Tuple[str, str]]:
    """(enhancer, gene) links, one per loop that supports them (duplicates kept,
    as BENGI kept them when it computed the distance percentile)."""
    links = []
    for (c1, s1, e1), (c2, s2, e2) in loops:
        if c1 != c2:
            continue
        x0 = enh.overlapping(c1, s1, e1)
        x1 = tss.genes_in(c1, s1 - PROXIMAL, e1 + PROXIMAL - 1)
        x2 = enh.overlapping(c2, s2, e2)
        x3 = tss.genes_in(c2, s2 - PROXIMAL, e2 + PROXIMAL - 1)
        links += [(e, g) for e in x0 for g in x3]
        links += [(e, g) for e in x2 for g in x1]
    return links


def build(cell: str, links: List[Tuple[str, str]], enh: EnhancerIndex,
          tss: TSSIndex) -> Tuple[List[Dict], Dict]:
    """Steps 3 and 4 of the procedure, from (enhancer, gene) links."""
    links = [(e, g) for e, g in links if g in tss.by_gene]
    if not links:
        return [], {"links": 0}
    dist = {(e, g): tss.distance(g, *enh.coords[e][1:]) for e, g in set(links)}
    cutoff = float(np.percentile([dist[k] for k in links], 95))
    positives: Dict[str, Set[str]] = defaultdict(set)
    for e, g in set(links):
        if dist[(e, g)] <= cutoff:
            positives[e].add(g)
    pairs = []
    window = int(round(cutoff))
    for e in sorted(positives):
        chrom, start, end = enh.coords[e]
        for g in sorted(tss.genes_in(chrom, max(0, start - window), end + window)):
            pairs.append((e, g, int(g in positives[e])))
    rows = [to_row(cell, e, g, label, enh, tss) for e, g, label in pairs]
    report = {"links": len(links),
              "distinct_links": len(set(links)), "cutoff_bp": cutoff,
              "enhancers_linked": len(positives), "pairs": len(rows),
              "positives": sum(r["label"] for r in rows)}
    report["positive_rate"] = report["positives"] / max(1, report["pairs"])
    return rows, report


def to_row(cell: str, e: str, g: str, label: int, enh: EnhancerIndex,
           tss: TSSIndex) -> Dict:
    chrom, start, end = enh.coords[e]
    centre = (start + end) / 2
    _, t, transcript, strand = tss.closest(g, centre)
    prom_start, prom_end = (t - 1499, t + 501) if strand == "+" else (t - 499, t + 1501)
    return {
        "label": label,
        "distance": abs((t + t + 1) / 2 - centre),
        "fields": [str(label), f"{abs((t + t + 1) / 2 - centre):.1f}", chrom,
                   str(start), str(end), f"{chrom}:{start}-{end}|{cell}|{e}",
                   chrom, str(prom_start), str(prom_end),
                   f"{chrom}:{t}-{t + 1}|{cell}|{g}|{transcript}|{strand}"],
    }


def write_rows(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as out:
        for row in rows:
            out.write("\t".join(row["fields"]) + "\n")


# ---------------------------------------------------------------------------
# Loop readers
# ---------------------------------------------------------------------------
def chrom_name(value: str) -> str:
    return value if value.startswith("chr") else f"chr{value}"


def read_rao(path: Path) -> List[Tuple[Interval, Interval]]:
    loops = []
    for i, line in enumerate(open_text(path)):
        f = line.rstrip("\n").split("\t")
        if i == 0 and not f[1].isdigit():
            continue
        loops.append(((chrom_name(f[0]), int(f[1]), int(f[2])),
                      (chrom_name(f[3]), int(f[4]), int(f[5]))))
    return loops


def read_bedpe(path: Path, assembly: str) -> Tuple[List[Tuple[Interval, Interval]], Dict]:
    loops = []
    for line in open_text(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6 or not f[1].isdigit():
            continue
        loops.append(((chrom_name(f[0]), int(f[1]), int(f[2])),
                      (chrom_name(f[3]), int(f[4]), int(f[5]))))
    if assembly == "GRCh38":
        from liftover import lift_bedpe
        kept, report = lift_bedpe(loops)
        return [(a, b) for _, a, b in kept], report
    return loops, {"loops_in": len(loops), "loops_kept": len(loops)}


def read_javierre(path: Path, columns: Sequence[str]) -> List[Tuple[Interval, Interval]]:
    """Bait and other end of every interaction scoring >= 5 in any of `columns`."""
    loops = []
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        wanted = [idx[c] for c in columns]
        for line in handle:
            f = line.rstrip("\n").split("\t")
            if max(float(f[i]) for i in wanted) < JAVIERRE_MIN_SCORE:
                continue
            loops.append(((chrom_name(f[idx["baitChr"]]), int(f[idx["baitStart"]]),
                           int(f[idx["baitEnd"]])),
                          (chrom_name(f[idx["oeChr"]]), int(f[idx["oeStart"]]),
                           int(f[idx["oeEnd"]]))))
    return loops


def gene_symbols(path: Path) -> Dict[str, Set[str]]:
    """Gene symbol to GENCODE v19 gene IDs, from TransEPI's TSS annotation
    (name column 'ENSG|SYMBOL|type|ENST')."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for line in open_text(path):
        gene, symbol = line.split("\t")[3].split("|")[:2]
        out[symbol].add(gene)
    return out


def read_jung(path: Path, sample: str, enh: EnhancerIndex,
              symbols: Dict[str, Set[str]]) -> Tuple[List[Tuple[str, str]], Dict]:
    """Links from one sample of Jung et al.'s significant interactions.

    The table (columns cell, chr, gene, target, raw, dist, ...) names the
    captured promoter by its gene symbols (",GENE1,GENE2") and gives the other
    end as a fragment "chr.start.end". Every row is already significant. An
    enhancer overlapping the fragment is linked to each named gene.
    """
    links, rows, unknown = [], 0, set()
    with open_text(path) as handle:
        header = handle.readline().split()
        col = {name: i for i, name in enumerate(header)}
        for line in handle:
            f = line.rstrip("\n").split("\t")
            if f[col["cell"]] != sample:
                continue
            rows += 1
            chrom, start, end = f[col["target"]].split(".")
            enhancers = enh.overlapping(chrom_name(chrom), int(start), int(end))
            if not enhancers:
                continue
            for symbol in filter(None, f[col["gene"]].split(",")):
                genes = symbols.get(symbol)
                if not genes:
                    unknown.add(symbol)
                    continue
                links += [(e, g) for e in enhancers for g in genes]
    return links, {"interactions": rows, "unknown_symbols": len(unknown)}


# ---------------------------------------------------------------------------
def gather_sources(cell: str, info: Dict, resolved: Dict, raw: Path,
                   sources: Set[str], enh: EnhancerIndex, tss: TSSIndex,
                   symbols: Dict[str, Set[str]]) -> List[Tuple[str, str, List, Dict]]:
    """(assay, label-source name, links, report) for each source of one cell."""
    out = []
    for assay, source, loops, report in gather_loops(cell, info, resolved, raw, sources):
        report["loops"] = len(loops)
        out.append((assay, source, link_pairs(loops, enh, tss), report))
    jung = raw / "labels" / "jung2019" / "GSE86189_all_interaction.po.txt.gz"
    if "jung" in sources and info.get("jung") and jung.exists():
        links, report = read_jung(jung, info["jung"], enh, symbols)
        out.append(("PCHiC", "jung2019", links, report))
    return out


def gather_loops(cell: str, info: Dict, resolved: Dict, raw: Path,
                 sources: Set[str]) -> List[Tuple[str, str, List, Dict]]:
    out = []
    rao = raw / "labels" / "rao2014" / f"GSE63525_{cell}_HiCCUPS_looplist.txt.gz"
    if "rao" in sources and rao.exists() and cell in ("HUVEC", "HMEC", "NHEK"):
        loops = read_rao(rao)
        out.append(("HiC", "rao2014", loops, {"loops_in": len(loops)}))
    if "bedpe" in sources:
        by_assay: Dict[str, List] = defaultdict(list)
        reports: Dict[str, List] = defaultdict(list)
        for record in resolved["files"]:
            if record["kind"] == "loops" and record["cell"] == cell:
                path = raw / record["dest"]
                if not path.exists():
                    continue
                loops, report = read_bedpe(path, record.get("assembly", "hg19"))
                by_assay[record["assay"]] += loops
                reports[record["assay"]].append({record["dest"]: report})
        for assay, loops in by_assay.items():
            out.append((assay, "encode", loops, {"files": reports[assay]}))
    javierre = raw.parent / "manual" / "javierre" / "PCHiC_peak_matrix_cutoff5.tsv"
    if "javierre" in sources and info.get("javierre") and javierre.exists():
        loops = read_javierre(javierre, info["javierre"])
        out.append(("PCHiC", "javierre2016", loops, {"loops_in": len(loops)}))
    return out


def output_folder(cell: str, role: str, manifest: Dict, source: str) -> str:
    test = set(manifest["test_only"]["cells"])
    if cell in test:
        return "BENGI_extra_test"
    if role in ("valid", "train"):
        return "BENGI_extended"
    return "BENGI_unused"


def check_hmec(raw: Path, tss: TSSIndex, resolved: Dict) -> Dict:
    """Rebuild BENGI's HMEC Hi-C set and compare pairs with the published one."""
    ccre = [raw / f["dest"] for f in resolved["files"]
            if f["kind"] == "ccre" and f["cell"] == "HMEC"]
    enh = cell_enhancers(ccre[0], tss)
    loops = read_rao(raw / "labels" / "rao2014" / "GSE63525_HMEC_HiCCUPS_looplist.txt.gz")
    rows, report = build("HMEC", link_pairs(loops, enh, tss), enh, tss)
    report["loops"] = len(loops)
    ours = {(r["fields"][5].split("|")[2], r["fields"][9].split("|")[2]): r["label"]
            for r in rows}
    published = {}
    for line in open_text(raw / "labels" / "bengi" / "Benchmark" /
                          "All-Pairs.Natural-Ratio" / "HMEC.HiC-Benchmark.v3.txt"):
        e, g, label = [x.strip() for x in line.split("\t")][:3]
        published[(e, g)] = int(label)
    shared = set(ours) & set(published)
    pos_o = {k for k, v in ours.items() if v}
    pos_p = {k for k, v in published.items() if v}
    return {
        "ours": report,
        "published_pairs": len(published), "published_positives": len(pos_p),
        "pair_jaccard": len(shared) / max(1, len(set(ours) | set(published))),
        "positive_jaccard": len(pos_o & pos_p) / max(1, len(pos_o | pos_p)),
        "label_agreement_on_shared": (sum(ours[k] == published[k] for k in shared)
                                      / max(1, len(shared))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sources", nargs="*",
                        default=["rao", "bedpe", "javierre", "jung"])
    parser.add_argument("--check-hmec", action="store_true")
    args = parser.parse_args()

    manifest, resolved = load_manifest(), load_resolved()
    raw, out = raw_root(), out_root()
    tss = TSSIndex(raw / "bengi" / "GENCODEv19-TSSs.bed.gz")
    symbols = gene_symbols(raw / "reference" / "transepi" / "data" / "annotation" /
                           "gencode.v19.tss.bed")
    reports: Dict = {}

    if args.check_hmec:
        reports["hmec_check"] = check_hmec(raw, tss, resolved)
        print(json.dumps(reports["hmec_check"], indent=1))

    cells = list(manifest["cells"]) if args.all else args.cells
    for cell in cells:
        info = manifest["cells"][cell]
        ccre = [raw / f["dest"] for f in resolved["files"]
                if f["kind"] == "ccre" and f["cell"] == cell]
        if not ccre or not ccre[0].exists():
            print(f"  {cell}: no cCRE file yet, skipped")
            continue
        enh = cell_enhancers(ccre[0], tss)
        for assay, source, links, source_report in gather_sources(
                cell, info, resolved, raw, set(args.sources), enh, tss, symbols):
            rows, report = build(cell, links, enh, tss)
            report.update(source=source, inputs=source_report,
                          enhancers=len(enh.coords))
            folder = output_folder(cell, info["role"], manifest, source)
            name = f"{cell}.{assay}-Benchmark.v3.tsv.gz"
            if source in ("javierre2016", "jung2019") or (cell in ("HMEC", "NHEK")
                                                          and source == "rao2014"):
                name = f"{cell}.{assay}_{source}-Benchmark.v3.tsv.gz"
            if rows:
                write_rows(rows, out / folder / name)
            reports[f"{folder}/{name}"] = report
            print(f"  {cell:16s} {assay:10s} {source:13s} links {report['links']:7d}  "
                  f"pairs {report.get('pairs', 0):7d}  positives "
                  f"{report.get('positives', 0):6d}  cutoff "
                  f"{report.get('cutoff_bp', 0) / 1000:7.1f} kb")

    log = out / "build_pairs_report.json"
    previous = json.loads(log.read_text()) if log.exists() else {}
    previous.update(reports)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(previous, indent=1))
    print(f"  report: {log}")


if __name__ == "__main__":
    main()

"""
Turn configs/data_sources.yaml into a frozen list of files to download.

Most sources are fixed URLs. The ENCODE ones are chosen by rule: the CTCF peak
file, the cell-type cCRE set and the Hi-C loop calls for each cell. This
script queries the portal once, applies the rules, and writes every choice,
with its accession, size and md5, to datasets/resolved.json. A later download
(here or on another machine given that file) fetches exactly the same files,
and the choices can be read and questioned.

Rules:
  CTCF   one peak file per cell. A fixed accession in the source list wins
         (TransEPI's choices for the cells it used). Otherwise hg19 before
         GRCh38, then the output types in the listed order, then the largest
         file (most peaks). GRCh38 peaks are lifted to hg19 later.
  cCRE   the full hg19 cell-type set of Registry v1 for the first matching
         biosample term (not the "high H3K27ac/CTCF/..." subsets); the largest
         if several donors exist.
  loops  every in situ or intact Hi-C loop file, one per experiment (the
         portal's default file when it marks one, else the largest).

Usage:
    python scripts/data/resolve_sources.py            # everything
    python scripts/data/resolve_sources.py --skip-sizes
    python scripts/data/resolve_sources.py --parts static github
                                  # refresh only these parts of resolved.json
    python scripts/data/resolve_sources.py --parts roadmap encode --cells NHEK
                                  # add or refresh one cell, keep the rest
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESOLVED, encode_file, encode_json, encode_search,  # noqa: E402
                    load_manifest)


PARTS = ("static", "roadmap", "encode", "eqtl", "github")
PART_OF = {"static": "static", "track": "roadmap", "ctcf": "encode", "ccre": "encode",
           "loops": "encode", "hct116": "encode", "eqtl": "eqtl", "github": "github"}


def head_size(url: str) -> Optional[int]:
    import requests
    try:
        response = requests.head(url, allow_redirects=True, timeout=60)
        if response.ok and response.headers.get("Content-Length"):
            return int(response.headers["Content-Length"])
    except Exception:
        pass
    return None


def biosample_name(record: Dict) -> Optional[str]:
    term = record.get("biosample_ontology")
    if isinstance(term, list):
        term = term[0] if term else {}
    return (term or {}).get("term_name")


# ---------------------------------------------------------------------------
def choose_ctcf(cell: str, info: Dict, rules: Dict) -> Optional[Dict]:
    if info.get("ctcf"):
        record = encode_file(info["ctcf"])
        record["rule"] = "fixed accession in configs/data_sources.yaml"
        return record
    order = rules["output_types"]
    candidates = []
    for term in info.get("encode", []):
        for record in encode_search(**{
                "type": "File", "file_format": "bed",
                "target.label": "CTCF", "assay_title": "TF ChIP-seq",
                "biosample_ontology.term_name": term,
                "field": ["accession", "assembly", "output_type", "file_size",
                          "preferred_default", "biosample_ontology.term_name"]}):
            if record.get("assembly") not in ("hg19", "GRCh38"):
                continue
            if record.get("output_type") not in order:
                continue
            candidates.append(record)
        if candidates:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r["assembly"] != "hg19",
                                   order.index(r["output_type"]),
                                   -(r.get("file_size") or 0)))
    best = encode_file(candidates[0]["accession"])
    best["rule"] = (f"best of {len(candidates)} by assembly, output type, size; "
                    f"term '{biosample_name(candidates[0])}'")
    return best


def choose_ccre(cell: str, info: Dict) -> Optional[Dict]:
    for term in info.get("encode", []):
        annotations = encode_search(**{
            "type": "Annotation",
            "annotation_type": "candidate Cis-Regulatory Elements",
            "assembly": "hg19", "biosample_ontology.term_name": term,
            "field": ["accession", "description"]})
        full = {f"/annotations/{a['accession']}/": a for a in annotations
                if not str(a.get("description", "")).lower().startswith("high ")}
        files = []
        if full:
            for f in encode_search(**{
                    "type": "File", "file_format": "bed", "assembly": "hg19",
                    "dataset": sorted(full),
                    "field": ["accession", "file_size", "dataset"]}):
                annotation = full.get(f.get("dataset"), {})
                files.append((f.get("file_size") or 0, f["accession"],
                              annotation.get("accession"),
                              annotation.get("description")))
        if files:
            files.sort(reverse=True)
            size, accession, annotation, description = files[0]
            record = encode_file(accession)
            record["annotation"] = annotation
            record["description"] = description
            record["rule"] = f"largest of {len(files)} full hg19 sets for '{term}'"
            return record
    return None


def choose_loops(cell: str, info: Dict, rules: Dict) -> List[Dict]:
    chosen = []
    for term in info.get("encode", []):
        records = [r for r in encode_search(**{
            "type": "File", "output_type": "loops",
            "biosample_ontology.term_name": term,
            "field": ["accession", "assembly", "assay_title", "file_size",
                      "dataset", "preferred_default", "file_format"]})
            if r.get("assay_title") in rules["assays"]
            and r.get("assembly") in ("hg19", "GRCh38")]
        by_experiment: Dict[str, List[Dict]] = {}
        for r in records:
            by_experiment.setdefault(r["dataset"], []).append(r)
        for experiment, files in sorted(by_experiment.items()):
            files.sort(key=lambda r: (not r.get("preferred_default", False),
                                      -(r.get("file_size") or 0)))
            record = encode_file(files[0]["accession"])
            record["assay_title"] = files[0]["assay_title"]
            record["term"] = term
            record["rule"] = (f"default or largest of {len(files)} loop files in "
                              f"{experiment}")
            chosen.append(record)
        if chosen:
            break
    return chosen


def hct116_tracks(rules: Dict) -> List[Dict]:
    out = []
    wanted = [("Histone ChIP-seq", mark) for mark in rules["histone_marks"]]
    wanted += [("DNase-seq", None), ("TF ChIP-seq", "CTCF")]
    for assay, target in wanted:
        params = {"type": "File", "assay_title": assay,
                  "biosample_ontology.term_name": rules["biosample"],
                  "field": ["accession", "assembly", "output_type", "file_size",
                            "file_format", "preferred_default"]}
        if target:
            params["target.label"] = target
        records = encode_search(**params)
        if assay == "TF ChIP-seq":
            records = [r for r in records if r.get("file_format") == "bed"
                       and "IDR" in str(r.get("output_type"))]
        else:
            records = [r for r in records if r.get("file_format") == "bigWig"]
            pval = [r for r in records if r.get("output_type") == "signal p-value"]
            records = pval or records
        records = [r for r in records if r.get("assembly") in ("hg19", "GRCh38")]
        if not records:
            out.append({"mark": target or "DNase", "missing": True})
            continue
        records.sort(key=lambda r: (r["assembly"] != "hg19",
                                    not r.get("preferred_default", False),
                                    -(r.get("file_size") or 0)))
        record = encode_file(records[0]["accession"])
        record["mark"] = target or "DNase"
        out.append(record)
    return out


def github_files(entry: Dict) -> List[Dict]:
    import requests
    out = []

    def walk(path: str):
        url = f"https://api.github.com/repos/{entry['repo']}/contents/{path}"
        listing = requests.get(url, params={"ref": entry["ref"]}, timeout=60).json()
        if not isinstance(listing, list):
            raise RuntimeError(
                f"GitHub listing of {entry['repo']}/{path} failed: "
                f"{listing.get('message')}. Unauthenticated requests are limited "
                f"to 60 an hour; rerun later with --parts github")
        for item in listing:
            if item["type"] == "dir" and entry.get("recursive"):
                walk(item["path"])
            elif item["type"] == "file":
                if entry.get("only") and item["name"] not in entry["only"]:
                    continue
                if entry.get("suffixes") and not item["name"].endswith(
                        tuple(entry["suffixes"])):
                    continue
                out.append(item)

    for path in entry["paths"]:
        walk(path)
    return out


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-sizes", action="store_true",
                        help="do not ask servers for file sizes")
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS,
                        help="re-resolve only these parts and keep the rest of an "
                             "existing resolved.json")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="re-resolve only these cells (their tracks, CTCF, "
                             "cCREs and loops) and keep everything else in an "
                             "existing resolved.json unchanged")
    args = parser.parse_args()
    parts = set(args.parts)
    named = set(args.cells or [])

    manifest = load_manifest()
    files: List[Dict] = []
    choices: Dict[str, Dict] = {}
    missing: List[str] = []
    if named:
        if not RESOLVED.exists():
            sys.exit("--cells updates an existing resolved.json; run without it first")
        previous = json.loads(RESOLVED.read_text())
        files = [f for f in previous["files"]
                 if not (PART_OF[f["kind"]] in parts and f.get("cell") in named)]
        choices = {k: v for k, v in previous["choices"].items() if k not in named}
        missing = [m for m in previous["missing"] if m.split(":")[0] not in named]
    elif parts != set(PARTS) and RESOLVED.exists():
        previous = json.loads(RESOLVED.read_text())
        files = [f for f in previous["files"] if PART_OF[f["kind"]] not in parts]
        choices = previous["choices"] if "encode" not in parts else {}
        missing = [m for m in previous["missing"]] if "encode" not in parts else []

    def add(**record):
        files.append(record)

    for entry in manifest["static"] if "static" in parts and not named else []:
        add(id=entry["name"], group=entry["group"], kind="static",
            url=entry["url"], dest=entry["dest"])

    roadmap = manifest["roadmap"]
    for cell, info in manifest["cells"].items():
        if not info.get("group") or "roadmap" not in parts or (named and cell not in named):
            continue
        for mark in roadmap["marks"]:
            name = roadmap["pattern"].format(eid=info["eid"], mark=mark)
            add(id=f"{cell} {mark}", group=info["group"], kind="track",
                cell=cell, eid=info["eid"], mark=mark,
                url=roadmap["base_url"] + name, dest=f"tracks/roadmap/{name}")

    encode = manifest["encode"]

    def lookup(item):
        cell, info = item
        entry: Dict = {"role": info["role"], "eid": info["eid"]}
        if info.get("group"):
            entry["ctcf"] = choose_ctcf(cell, info, encode["ctcf"])
        if info["role"] != "pretrain":
            entry["ccre"] = choose_ccre(cell, info)
            entry["loops"] = choose_loops(cell, info, encode["loops"])
        return cell, info, entry

    cells = [(c, i) for c, i in manifest["cells"].items()
             if "encode" in parts and (not named or c in named)]
    with ThreadPoolExecutor(4) as pool:
        looked_up = list(pool.map(lookup, cells))
    for cell, info, entry in looked_up:
        needs_inputs = bool(info.get("group"))
        print(f"  {cell:22s}", end=" ")
        if needs_inputs:
            ctcf = entry["ctcf"]
            if ctcf:
                add(id=f"{cell} CTCF", group=info["group"], kind="ctcf", cell=cell,
                    url=ctcf["url"], size=ctcf["size"], md5=ctcf["md5"],
                    assembly=ctcf["assembly"],
                    dest=f"tracks/ctcf/{cell}_{ctcf['accession']}.{ctcf['assembly']}.bed.gz")
            else:
                missing.append(f"{cell}: no CTCF peaks")
        if info["role"] != "pretrain":
            ccre = entry["ccre"]
            if ccre:
                add(id=f"{cell} cCREs", group="labels", kind="ccre", cell=cell,
                    url=ccre["url"], size=ccre["size"], md5=ccre["md5"],
                    dest=f"labels/ccre/{cell}_{ccre['accession']}.hg19.bed.gz")
            else:
                missing.append(f"{cell}: no hg19 cell-type cCREs")
            for loop in entry["loops"]:
                assay = "intactHiC" if loop["assay_title"] == "intact Hi-C" else "HiC"
                add(id=f"{cell} {assay} loops {loop['accession']}", group="labels",
                    kind="loops", cell=cell, assay=assay, url=loop["url"],
                    size=loop["size"], md5=loop["md5"], assembly=loop["assembly"],
                    dest=(f"labels/encode_loops/{cell}.{assay}.{loop['accession']}."
                          f"{loop['assembly']}.{loop['file_format']}.gz"))
        choices[cell] = entry
        print(f"ctcf={'yes' if entry.get('ctcf') else '-'} "
              f"ccre={'yes' if entry.get('ccre') else '-'} "
              f"loops={len(entry.get('loops', []))}")

    hct = hct116_tracks(encode["hct116"]) if "encode" in parts and not named else []
    if hct:
        print(f"  HCT116 tracks: {len([r for r in hct if not r.get('missing')])} files")
    for record in hct:
        if record.get("missing"):
            missing.append(f"HCT116: no {record['mark']} track")
            continue
        ext = "bed.gz" if record["file_format"] == "bed" else "bigWig"
        add(id=f"HCT116 {record['mark']}", group="eval", kind="hct116",
            cell="HCT116", mark=record["mark"], url=record["url"],
            size=record["size"], md5=record["md5"], assembly=record["assembly"],
            output_type=record["output_type"],
            dest=f"tracks/hct116/HCT116_{record['mark']}_{record['accession']}.{record['assembly']}.{ext}")
    if hct:
        choices["HCT116"] = {"role": "test (optional)", "tracks": hct}

    eqtl = manifest["eqtl"]
    for d in eqtl["datasets"] if "eqtl" in parts and not named else []:
        name = f"{d['dataset']}.credible_sets.tsv.gz"
        add(id=f"eQTL {d['label']}", group="eval", kind="eqtl", label=d["label"],
            url=f"{eqtl['base_url']}{d['study']}/{d['dataset']}/{name}",
            dest=f"eval/eqtl/{name}")

    for entry in manifest["github_folders"] if "github" in parts and not named else []:
        for item in github_files(entry):
            add(id=f"{entry['name']}: {item['path']}", group=entry["group"],
                kind="github", url=item["download_url"], size=item["size"],
                dest=f"{entry['dest']}/{item['path']}")

    if not args.skip_sizes:
        todo = [f for f in files if not f.get("size")]
        print(f"  asking servers for {len(todo)} file sizes ...")
        with ThreadPoolExecutor(8) as pool:
            for record, size in zip(todo, pool.map(lambda f: head_size(f["url"]), todo)):
                record["size"] = size
        for record in todo:
            if record["size"] is None:
                missing.append(f"{record['id']}: server gave no size ({record['url']})")

    totals: Dict[str, float] = {}
    for record in files:
        totals[record["group"]] = totals.get(record["group"], 0) + (record.get("size") or 0)
    payload = {
        "resolved_on": dt.date.today().isoformat(),
        "group_gb": {k: round(v / 2 ** 30, 2) for k, v in sorted(totals.items())},
        "missing": missing,
        "choices": choices,
        "files": files,
    }
    RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(payload, indent=1))
    print(f"\n  {len(files)} files, by group (GB): {payload['group_gb']}")
    print(f"  {len(missing)} gaps:")
    for line in missing:
        print("   -", line)
    print(f"  written to {RESOLVED}")


if __name__ == "__main__":
    main()

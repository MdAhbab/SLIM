"""
Download the files listed in datasets/resolved.json.

Groups, smallest first:
  tools            liftOver chain, chromosome sizes, blacklist, BENGI annotations
  labels           loop lists, capture Hi-C, cCRE sets
  eval             CRISPR, ABC, eQTL, TargetFinder, CBOEP2, HCT116
  tracks-check     HMEC tracks, to check the processing against the
                   published HMEC track files
  tracks-bengi     tracks of the other five BENGI cell lines, so that every
                   cell shares one processing
  tracks-core      tracks of the new test, validation and training cells
  tracks-pretrain  tracks of the remaining epigenomes

Every file is resumed if a partial copy exists and skipped if it is complete,
so the command can be repeated after any interruption. Each outcome is
appended to <raw>/download_log.jsonl.

Usage:
    python scripts/data/download.py --list
    python scripts/data/download.py --groups tools labels eval
    python scripts/data/download.py --groups all --workers 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import download, load_resolved, raw_root  # noqa: E402

ORDER = ["tools", "labels", "eval", "tracks-check", "tracks-bengi", "tracks-core",
         "tracks-pretrain"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", default=["tools", "labels", "eval"],
                        help=f"any of {ORDER}, or 'all'")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel downloads (the Roadmap server is slow "
                             "per connection, so a few help)")
    parser.add_argument("--list", action="store_true",
                        help="print sizes per group and exit")
    args = parser.parse_args()

    resolved = load_resolved()
    files = resolved["files"]
    root = raw_root()
    groups = ORDER if "all" in args.groups else args.groups
    unknown = set(groups) - set(ORDER)
    if unknown:
        parser.error(f"unknown groups: {sorted(unknown)}")

    if args.list:
        for group in ORDER:
            chosen = [f for f in files if f["group"] == group]
            have = sum(1 for f in chosen if (root / f["dest"]).exists())
            size = sum(f.get("size") or 0 for f in chosen) / 2 ** 30
            print(f"  {group:16s} {len(chosen):4d} files {size:8.1f} GB   "
                  f"{have} present")
        print(f"  raw root: {root}")
        return

    todo = [f for g in groups for f in files if f["group"] == g]
    total = sum(f.get("size") or 0 for f in todo)
    print(f"  {len(todo)} files, {total / 2 ** 30:.1f} GB, into {root}")
    root.mkdir(parents=True, exist_ok=True)
    log = root / "download_log.jsonl"
    done_bytes, failures = 0, []
    start = time.time()

    def fetch(record):
        return record, download(record["url"], root / record["dest"],
                                size=record.get("size"), md5=record.get("md5"))

    with ThreadPoolExecutor(max(1, args.workers)) as pool, open(log, "a") as out:
        futures = [pool.submit(fetch, r) for r in todo]
        for n, future in enumerate(as_completed(futures), 1):
            record, result = future.result()
            result.update(id=record["id"], group=record["group"],
                          time=time.strftime("%Y-%m-%d %H:%M:%S"))
            out.write(json.dumps(result) + "\n")
            out.flush()
            done_bytes += record.get("size") or result.get("bytes") or 0
            if result["status"] == "failed":
                failures.append(result)
            rate = done_bytes / max(1.0, time.time() - start) / 2 ** 20
            print(f"  [{n}/{len(todo)}] {result['status']:10s} {record['id']}"
                  f"  ({done_bytes / 2 ** 30:.1f} GB, {rate:.1f} MB/s)", flush=True)

    print(f"\n  finished: {len(todo) - len(failures)} ok, {len(failures)} failed")
    for result in failures:
        print(f"   - {result['id']}: {result.get('error')}")
    if failures:
        print("  Run the same command again to retry the failed files.")
        sys.exit(1)


if __name__ == "__main__":
    main()

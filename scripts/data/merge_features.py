"""
Merge the feature list of new cells into an existing feature configuration.

The training code reads one JSON file mapping each cell and mark to a track
file. The existing file names the six BENGI cell lines; prepare_tracks.py
writes features_new_cells.json for the new ones. This writes a third file with
both, leaving the existing one untouched, and checks that every named track
file is present in the track directory.

Usage (on the training PC, from the repository root):
    python scripts/data/merge_features.py \
        --existing "D:/E/AI Models/FYDP-Model/data/genomic_data/CTCF_DNase_6histone_local.500.json" \
        --new      "D:/E/AI Models/FYDP-Model/data/genomic_data/features_new_cells.json" \
        --out      "D:/E/AI Models/FYDP-Model/data/genomic_data/CTCF_DNase_6histone_extended.500.json"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import resolve_track_location  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    existing = json.loads(Path(args.existing).read_text())
    new = json.loads(Path(args.new).read_text())
    location = existing.get("_location")
    new.pop("_location", None)

    clashes = sorted(set(existing) & set(new))
    if clashes:
        sys.exit(f"cells named in both files: {clashes}; nothing written")

    merged = dict(existing)
    merged.update(new)
    track_dir = resolve_track_location(args.out, location)
    missing = [f"{cell} {mark}: {name}" for cell, marks in new.items()
               for mark, name in marks.items()
               if not os.path.exists(os.path.join(track_dir, name))]
    if missing:
        print(f"  {len(missing)} track files are not in {track_dir}:")
        for line in missing[:20]:
            print("   -", line)
        sys.exit("copy the new .pt files there first; nothing written")

    if os.path.abspath(args.out) == os.path.abspath(args.existing):
        sys.exit("--out must differ from --existing")
    Path(args.out).write_text(json.dumps(merged, indent=2))
    cells = [c for c in merged if not c.startswith("_")]
    print(f"  {len(cells)} cells written to {args.out} (tracks in {track_dir})")


if __name__ == "__main__":
    main()

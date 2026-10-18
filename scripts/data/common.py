"""
Shared pieces of the dataset scripts: where files live, the source list,
the ENCODE portal, and a resumable download.

Two roots hold everything, both under the untracked datasets/ directory by
default. RAW keeps the downloads. OUT keeps what the training machine needs:
the binned track files, the new benchmark files and the feature configuration.
The defaults come from configs/data_sources.yaml; the SLIM_RAW and SLIM_OUT
environment variables move them, for example to a notebook's scratch disk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "data_sources.yaml"
DATASETS = ROOT / "datasets"
RESOLVED = DATASETS / "resolved.json"
ENCODE = "https://www.encodeproject.org"


def load_manifest() -> Dict:
    with open(MANIFEST) as handle:
        return yaml.safe_load(handle)


def load_resolved() -> Dict:
    if not RESOLVED.exists():
        raise FileNotFoundError(
            f"{RESOLVED} is missing; run scripts/data/resolve_sources.py first")
    with open(RESOLVED) as handle:
        return json.load(handle)


def _root(env: str, key: str) -> Path:
    path = Path(os.path.expanduser(
        os.environ.get(env, load_manifest()["roots"][key])))
    return path if path.is_absolute() else ROOT / path


def raw_root() -> Path:
    return _root("SLIM_RAW", "raw")


def out_root() -> Path:
    return _root("SLIM_OUT", "out")


def open_text(path) -> Iterable[str]:
    """Open a plain or gzipped text file for reading. The content decides, not
    the name: some ENCODE files named .gz are plain text."""
    with open(path, "rb") as handle:
        gzipped = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gzipped else open(path, "rt")


# ---------------------------------------------------------------------------
# ENCODE portal
# ---------------------------------------------------------------------------
def _session():
    import requests
    session = requests.Session()
    session.headers.update({"Accept": "application/json",
                            "User-Agent": "SLIM-dataset-resolver"})
    return session


_SESSION = None


def encode_json(path: str, params: Optional[Dict] = None, retries: int = 4) -> Dict:
    """GET a portal object or search as JSON, retrying transient failures."""
    global _SESSION
    if _SESSION is None:
        _SESSION = _session()
    url = path if path.startswith("http") else ENCODE + path
    for attempt in range(retries):
        try:
            response = _SESSION.get(url, params=params, timeout=120)
            if response.status_code == 404:
                return {"@graph": []}
            response.raise_for_status()
            time.sleep(0.1)          # the portal asks for at most 10 per second
            return response.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def encode_search(**params) -> List[Dict]:
    """Every object matching an ENCODE search, as a list."""
    query = {"format": "json", "limit": "all", "status": "released"}
    query.update(params)
    return encode_json("/search/", query).get("@graph", [])


def encode_file(accession: str) -> Dict:
    """The download record of one ENCODE file."""
    record = encode_json(f"/files/{accession}/", {"format": "json"})
    if not record or "accession" not in record:
        raise KeyError(f"ENCODE file {accession} not found")
    url = (record.get("cloud_metadata") or {}).get("url") or (ENCODE + record["href"])
    return {
        "accession": record["accession"],
        "url": url,
        "md5": record.get("md5sum"),
        "size": record.get("file_size"),
        "assembly": record.get("assembly"),
        "file_format": record.get("file_format"),
        "file_type": record.get("file_type"),
        "output_type": record.get("output_type"),
        "dataset": record.get("dataset"),
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, size: Optional[int] = None,
             md5: Optional[str] = None, retries: int = 8) -> Dict:
    """Fetch `url` into `dest`, resuming a partial `.part` file.

    A finished file is recognised by its size (when known) and kept, so the
    call is safe to repeat. The md5 is checked when the source publishes one.
    Returns a record of what happened.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return {"url": url, "dest": str(dest), "status": "present",
                "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120,
                              allow_redirects=True) as response:
                if response.status_code == 416:      # already complete
                    break
                if have and response.status_code == 200:
                    have = 0                          # server ignored Range
                response.raise_for_status()
                total = response.headers.get("Content-Length")
                if size is None and total is not None:
                    size = int(total) + have
                with open(part, "ab" if have else "wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
            if size is None or part.stat().st_size >= size:
                break
        except Exception as error:  # network hiccup: back off and resume
            if attempt == retries - 1:
                return {"url": url, "dest": str(dest), "status": "failed",
                        "error": repr(error)}
            time.sleep(min(60, 2 ** attempt))
    if size is not None and part.stat().st_size != size:
        return {"url": url, "dest": str(dest), "status": "failed",
                "error": f"size {part.stat().st_size} != expected {size}"}
    if md5 and md5_of(part) != md5:
        part.unlink()
        return {"url": url, "dest": str(dest), "status": "failed",
                "error": "md5 mismatch; partial file removed"}
    part.replace(dest)
    return {"url": url, "dest": str(dest), "status": "downloaded",
            "bytes": dest.stat().st_size}

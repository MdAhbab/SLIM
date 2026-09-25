"""
Configuration loading with single-level inheritance.

Every variant shares one base configuration and overrides only the few keys
that define it. This is the configuration counterpart of the single model
class: a variant cannot accidentally differ in the data pipeline, the
optimizer or the training budget, because it does not restate them.

A configuration file opts in by naming its parent:

    extends: base.yaml

Paths are resolved relative to the child file's directory. Merging is
recursive for mappings; any other value replaces the parent's outright.

The three dataset paths under `paths:` are made absolute on load. A relative
path is taken relative to the project root, never the working directory, so
training, testing and evaluation read the same files whichever directory the
command is issued from. An absolute path is left exactly as written, which is
what lets the tests point a configuration at a temporary fixture directory.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

# The repository root: the parent of the directory holding this file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keys under `paths:` that name a file or directory on disk.
PATH_KEYS = ("bengi_dir", "feats_config", "ref_genome")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested mappings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (key in merged and isinstance(merged[key], dict)
                and isinstance(value, dict)):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_path(value: str, root: str = PROJECT_ROOT) -> str:
    """Expand `~` and environment variables, then absolutise against `root`.

    An empty value is returned unchanged: `ref_genome: ""` is the documented
    way to ask for placeholder sequences instead of hg19.
    """
    if not value:
        return value
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if not os.path.isabs(expanded):
        expanded = os.path.join(root, expanded)
    return os.path.normpath(expanded)


def resolve_paths(config: Dict[str, Any],
                  root: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Absolutise every dataset path in `config['paths']`, in place."""
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return config
    for key in PATH_KEYS:
        if key in paths and isinstance(paths[key], str):
            paths[key] = resolve_path(paths[key], root)
    return config


def resolve_track_location(feats_config_path: str,
                           declared: Any = None) -> str:
    """Directory holding the binned .pt tracks a feature configuration names.

    A feature configuration may carry a `_location` key recorded on the
    machine that produced it. When that directory is absent here, because the
    dataset has been copied to another drive, fall back to the configuration's
    own directory, preferring a `processed/` subdirectory. Without this the
    tracks would simply fail to load and every chromatin channel would be
    zeros, which trains and scores without ever looking wrong.
    """
    config_dir = os.path.dirname(os.path.abspath(feats_config_path))
    candidates = []
    if declared:
        candidates.append(resolve_path(str(declared), config_dir))
    candidates.append(os.path.join(config_dir, "processed"))
    candidates.append(config_dir)
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _load_merged(path: str) -> Dict[str, Any]:
    """Load a configuration file, applying its parent first if it names one."""
    path = os.path.abspath(path)
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    parent_name = config.pop("extends", None)
    if parent_name is None:
        return config

    parent_path = os.path.join(os.path.dirname(path), parent_name)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(
            f"{path} extends {parent_name}, which does not exist at "
            f"{parent_path}")
    return deep_merge(_load_merged(parent_path), config)


def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration, inherit from its parent, and absolutise paths."""
    return resolve_paths(_load_merged(path))

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
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


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


def load_config(path: str) -> Dict[str, Any]:
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
    return deep_merge(load_config(parent_path), config)

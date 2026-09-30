"""
Tests for the long-training suite (python run.py --suite long).

They check that the training-fraction subsets are stratified, nested and
fixed by the seed; that the long configurations of SLIM-GA and the baseline
share one recipe; and that every stage of the suite writes to its own
directory under the chosen protocol, so the main results are never touched.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bengi import fraction_of_rows  # noqa: E402
from src.config import load_config  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location("run", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


def test_fraction_keeps_every_row_at_one():
    labels = [i % 7 == 0 for i in range(70)]
    assert fraction_of_rows(range(70), labels, 1.0, seed=3) == list(range(70))


def test_fraction_is_stratified_nested_and_fixed_by_the_seed():
    labels = [1 if i % 10 == 0 else 0 for i in range(1000)]
    idx = list(range(1000))
    half = fraction_of_rows(idx, labels, 0.5, seed=0)
    quarter = fraction_of_rows(idx, labels, 0.25, seed=0)
    assert len(half) == 500 and sum(labels[i] for i in half) == 50
    assert len(quarter) == 250 and sum(labels[i] for i in quarter) == 25
    assert set(quarter) <= set(half)
    assert quarter == fraction_of_rows(idx, labels, 0.25, seed=0)
    assert quarter != fraction_of_rows(idx, labels, 0.25, seed=1)


def test_fraction_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        fraction_of_rows([0, 1], [0, 1], 0.0, seed=0)
    with pytest.raises(ValueError):
        fraction_of_rows([0, 1], [0, 1], 1.5, seed=0)


def test_long_configs_share_one_recipe():
    ga = load_config(str(ROOT / "configs" / "slim_ga_long.yaml"))
    base = load_config(str(ROOT / "configs" / "baseline_long.yaml"))
    assert ga["model"]["variant"] == "GA"
    assert ga["training"]["epochs"] == 30
    for key in ("epochs", "patience", "lr", "scheduler", "weight_decay",
                "ema_decay", "warmup_fraction", "min_lr_ratio"):
        assert ga["training"][key] == base["training"][key], key
    assert ga["data"] == base["data"]
    for percent in (25, 50):
        cfg = load_config(str(ROOT / "configs" / f"slim_ga_long_frac{percent}.yaml"))
        assert cfg["data"]["train_fraction"] == percent / 100
        assert cfg["model"] == ga["model"]
        assert cfg["training"] == ga["training"]


def test_long_suite_writes_apart_from_the_main_results():
    run = load_runner()
    plan = run.build_long_plan([0, 1, 2])
    main = {stage["name"] for stage in run.build_plan([0, 1, 2])}
    names = [stage["name"] for stage in plan]
    assert len(names) == len(set(names))
    assert not set(names) & main
    for stage in plan:
        if not stage["name"].startswith("train_"):
            continue
        command = stage["command"]
        config = command[command.index("--config") + 1]
        assert (ROOT / config).exists(), config
        assert "--protocol" in command
        out = command[command.index("--output-dir") + 1]
        assert out.split("/")[1] not in {"baseline", "a", "ka", "ga"}

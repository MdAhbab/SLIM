"""
Run every experiment in order, and survive being interrupted.

This is the single command to run on the training machine. It checks the
environment, checks the data, runs the tests, then works through the
experiments one at a time, recording what finished. If the power fails, or the
machine is switched off, or the process is stopped with Ctrl-C, run the same
command again: finished stages are skipped, and a training stage that was cut
off resumes from its last completed epoch rather than starting over.

    python run.py

That is the whole workflow. Other useful forms:

    python run.py --check          only verify the environment and the data
    python run.py --list           show what is done, pending or failed
    python run.py --only leakage   run one stage
    python run.py --from train_ka_seed0
                                   run that stage and everything after it
    python run.py --redo train_a_seed0
                                   mark a stage unfinished so it runs again
    python run.py --dry-run        print the plan without running anything
    python run.py --archive before_pipeline_fixes
                                   move every existing result into
                                   results/archive/<name>/ and start afresh

Progress lives in `results/run_state.json`. Every stage also writes its console
output to `results/logs/<stage>.log`, so a failure can be read afterwards.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
STATE_PATH = RESULTS / "run_state.json"
PY = sys.executable

# Packages the pipeline needs. pyfaidx is listed as required rather than
# optional on purpose: without it the sequence branch silently receives
# placeholder sequences and every result is meaningless.
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pyfaidx", "pyfaidx"),
    ("pytest", "pytest"),
]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

PROTOCOL = "results/protocol.json"


def training_stage(name, config, seed, extra=(), out=None, minutes=80,
                   why="training run", protocol=True):
    """One call of scripts/train.py.

    With `protocol`, the run applies the training assays and inputs that the
    pilot runs chose (results/protocol.json), so every model and ablation
    trains under the same protocol.
    """
    command = [PY, "scripts/train.py", "--config", f"configs/{config}.yaml",
               "--seed", str(seed), *extra]
    if protocol:
        command += ["--protocol", PROTOCOL]
    if out:
        command += ["--output-dir", out]
        produces = f"{out}/eval_results.npz"
    else:
        variant = {"baseline": "baseline", "slim_a": "a",
                   "slim_ka": "ka", "slim_ga": "ga"}[config]
        produces = f"results/{variant}/seed{seed}/eval_results.npz"
    return {"name": name, "command": command, "produces": produces,
            "minutes": minutes, "resumable": True, "why": why}


# The four pilot runs: variant KA at seed 0, crossing the two protocol
# choices. Each is validated on the same Hi-C rows.
PILOTS = [
    ("pilot_hic_dna", ["--train-assays", "HiC", "--modalities", "all"], 75,
     "Hi-C training rows, DNA branch on"),
    ("pilot_hic_nodna", ["--train-assays", "HiC", "--modalities", "epi"], 65,
     "Hi-C training rows, DNA branch off"),
    ("pilot_all_dna", ["--train-assays", "all", "--modalities", "all"], 130,
     "every assay, repeats merged, DNA branch on"),
    ("pilot_all_nodna", ["--train-assays", "all", "--modalities", "epi"], 110,
     "every assay, repeats merged, DNA branch off"),
]


def writelog_stage(run):
    name = run.replace("results/", "").replace("/seed0", "")
    return {"name": f"writelog_{name}",
            "command": [PY, "scripts/memory_writelog.py", "--run", run],
            "produces": f"{run}/writelog.json", "minutes": 30,
            "why": "do the selected positions carry regulatory signal"}


def build_plan(seeds, extra_seeds=(3, 4)):
    plan = [
        {"name": "tests",
         "command": [PY, "-m", "pytest", "-q"],
         "produces": None, "minutes": 2,
         "why": "catch a broken checkout before spending hours on training"},
        {"name": "audit_dataset",
         "command": [PY, "scripts/audit_dataset.py", "--out", "results/audit"],
         "produces": "results/audit/dataset_audit.json", "minutes": 2,
         "why": "repeats, anchor reuse, distance-only and locus baselines"},
        {"name": "audit_encoder",
         "command": [PY, "scripts/audit_encoder_leakage.py",
                     "--out", "results/audit/encoder_leakage.json"],
         "produces": "results/audit/encoder_leakage.json", "minutes": 20,
         "why": "does the sequence encoding leak training labels"},
    ]

    # Protocol pilots, then the choice, made on validation only.
    for name, extra, minutes, why in PILOTS:
        plan.append(training_stage(name, "slim_ka", 0, extra=extra,
                                   out=f"results/{name}", minutes=minutes,
                                   why=f"pilot: {why}", protocol=False))
    plan.append(
        {"name": "choose_protocol",
         "command": [PY, "scripts/choose_protocol.py", "--pilots",
                     *[f"results/{name}" for name, *_ in PILOTS],
                     "--out", PROTOCOL],
         "produces": PROTOCOL, "minutes": 1,
         "why": "pick the protocol on Hi-C validation; test is not read"})
    plan.append(
        {"name": "leakage",
         "command": [PY, "scripts/audit_leakage.py",
                     "--config", "configs/slim_ka.yaml",
                     "--protocol", PROTOCOL, "--out", "results/leakage"],
         "produces": "results/leakage/leakage_index.npz", "minutes": 2,
         "why": "test loci shared with the training territory"})

    # Every model at the first seeds, under the chosen protocol.
    for seed in seeds:
        plan.append(training_stage(
            f"train_baseline_seed{seed}", "baseline", seed,
            why="global-attention control, the comparison point"))
        plan.append(training_stage(
            f"train_ka_seed{seed}", "slim_ka", seed,
            why="matched to the baseline; carries the central claim"))
        plan.append(training_stage(
            f"train_ga_seed{seed}", "slim_ga", seed, minutes=40,
            why="gated feed-forward variant"))
        plan.append(training_stage(
            f"train_a_seed{seed}", "slim_a", seed, minutes=40,
            why="rectified feed-forward variant"))

    # Ablations, all variant KA at seed 0 under the chosen protocol.
    plan.append(training_stage(
        "train_ka_old_recipe_seed0", "slim_ka_old_recipe", 0,
        out="results/ka_old_recipe/seed0", minutes=70,
        why="fixed data, original training recipe: what the recipe changed"))
    plan.append(training_stage(
        "train_ka_legacy_seed0", "slim_ka_legacy", 0,
        out="results/ka_legacy/seed0",
        why="original memory rules: what the encoder fixes changed"))
    plan.append(training_stage(
        "train_ka_dna_geometry", "slim_ka", 0,
        extra=["--modalities", "seq"], out="results/ka_dna_geometry/seed0",
        why="no chromatin tracks: DNA plus pair geometry only"))
    for config, why in (
            ("slim_ka_random_select", "survivors chosen at random"),
            ("slim_ka_no_readback", "memory never read back: local attention only"),
            ("slim_ka_learned_only", "no novelty or prediction-error term")):
        short = config.replace("slim_", "")
        plan.append(training_stage(
            f"train_{short}_seed0", config, 0, out=f"results/{short}/seed0",
            why=f"memory ablation: {why}"))

    # More seeds for the central comparison.
    for seed in extra_seeds:
        plan.append(training_stage(
            f"train_baseline_seed{seed}", "baseline", seed,
            why="extra seed for the central comparison"))
        plan.append(training_stage(
            f"train_ka_seed{seed}", "slim_ka", seed,
            why="extra seed for the central comparison"))

    plan += [
        {"name": "efficiency",
         "command": [PY, "scripts/benchmark_efficiency.py",
                     "--out", "results/efficiency.json"],
         "produces": "results/efficiency.json", "minutes": 10,
         "why": "measured latency, throughput and peak memory"},
        {"name": "evaluate",
         "command": [PY, "scripts/evaluate.py",
                     "--run", "AUTO_RUNS",
                     "--leakage", "results/leakage",
                     "--bootstrap", "2000",
                     "--out", "results/evaluation.json"],
         "produces": "results/evaluation.json", "minutes": 5,
         "why": "thresholds, calibration, intervals, leakage-disjoint subset"},
        {"name": "overfitting",
         "command": [PY, "scripts/check_overfitting.py",
                     "--run", "AUTO_RUNS",
                     "--out", "results/overfitting.json"],
         "produces": "results/overfitting.json", "minutes": 1,
         "why": "separates memorisation from loss of transfer"},
        {"name": "seeds",
         "command": [PY, "scripts/aggregate_seeds.py",
                     "--reference", "baseline",
                     "--out", "results/seed_summary.json"],
         "produces": "results/seed_summary.json", "minutes": 1,
         "why": "per-seed values, means and standard deviations"},
        writelog_stage("results/ka/seed0"),
        writelog_stage("results/ka_legacy/seed0"),
        writelog_stage("results/ga/seed0"),
        {"name": "figures",
         "command": [PY, "scripts/build_results.py", "--seed", "0",
                     "--figdir", "figures"],
         "produces": "results/results.json", "minutes": 2,
         "why": "tables and figures, regenerated from the saved predictions"},
    ]
    return plan


def archive_results(name):
    """Move every current result into results/archive/<name>/.

    Tracked and untracked files alike (predictions, histories, reports, logs,
    checkpoints and the progress record), so the next run starts from a
    clean results directory and no stage is skipped because an old output
    already exists.
    """
    target = RESULTS / "archive" / name
    if target.exists():
        print(f"  {target.relative_to(ROOT)} already exists; choose another name.")
        return False
    entries = [e for e in RESULTS.iterdir() if e.name != "archive"] \
        if RESULTS.exists() else []
    if not entries:
        print("  Nothing to archive.")
        return True
    target.mkdir(parents=True)
    for entry in sorted(entries):
        shutil.move(str(entry), str(target / entry.name))
        print(f"  moved {entry.name}")
    print(f"  {len(entries)} entries are now in {target.relative_to(ROOT)}")
    return True


def expand_auto_runs(command):
    """Replace the AUTO_RUNS placeholder with the run directories that exist."""
    if "AUTO_RUNS" not in command:
        return command
    found = sorted(str(p.parent.relative_to(ROOT))
                   for p in RESULTS.glob("*/seed*/eval_results.npz"))
    found += sorted(str(p.parent.relative_to(ROOT))
                    for p in RESULTS.glob("*/eval_results.npz")
                    if not p.parent.name.startswith("seed"))
    if not found:
        return None
    index = command.index("AUTO_RUNS")
    return command[:index] + found + command[index + 1:]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            print(f"  {STATE_PATH} is unreadable, starting a fresh record.")
    return {"stages": {}, "started": datetime.now().isoformat()}


def save_state(state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, STATE_PATH)


def is_done(stage, state):
    """A stage counts as finished only if it was recorded AND its output exists."""
    record = state["stages"].get(stage["name"], {})
    if record.get("status") != "done":
        return False
    produces = stage.get("produces")
    if produces and not (ROOT / produces).exists():
        return False
    return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_packages():
    print("Packages")
    missing = []
    for module, package in REQUIRED_PACKAGES:
        try:
            loaded = importlib.import_module(module)
            version = getattr(loaded, "__version__", "")
            print(f"  {package:<16} {version}")
        except ImportError:
            print(f"  {package:<16} MISSING")
            missing.append(package)
    if missing:
        print("\n  Install what is missing:")
        print(f"    pip install {' '.join(missing)}")
        if "torch" in missing:
            print("  Install torch with the CUDA build matching your driver,")
            print("  from https://pytorch.org/get-started/locally/")
    return not missing


def check_gpu():
    print("\nHardware")
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        print("  No CUDA device visible. Training on the processor would take")
        print("  days rather than hours. Check the driver and the torch build:")
        print("    python -c \"import torch; print(torch.version.cuda)\"")
        return False
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU            {name}")
    print(f"  Memory         {total:.1f} GB")
    print(f"  torch CUDA     {torch.version.cuda}")
    if total < 15:
        print("  Under 16 GB. Lower data.batch_size in configs/base.yaml from")
        print("  64 to 32 if training runs out of memory.")
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"  Free disk      {free:.1f} GB")
    if free < 20:
        print("  Under 20 GB free. Checkpoints and predictions need room.")
    return True


def check_data():
    """Confirm the three inputs are present before a long job starts."""
    print("\nData")
    sys.path.insert(0, str(ROOT))
    from src.config import load_config, resolve_track_location
    config = load_config(str(ROOT / "configs" / "base.yaml"))
    paths = config["paths"]
    ok = True

    bengi = Path(paths["bengi_dir"])
    wanted = ["GM12878", "HeLa", "K562", "IMR90", "HMEC", "NHEK"]
    if not bengi.exists():
        print(f"  BENGI directory missing: {bengi}")
        ok = False
    else:
        present = {p.name.split(".")[0] for p in bengi.glob("*.tsv*")}
        for cell in wanted:
            mark = "found" if cell in present else "MISSING"
            print(f"  {cell:<10} {mark}")
            if cell not in present:
                ok = False

    feats = Path(paths["feats_config"])
    if not feats.exists():
        print(f"  Track configuration missing: {feats}")
        ok = False
    else:
        import json as _json
        mapping = _json.loads(feats.read_text())
        location = resolve_track_location(str(feats), mapping.get("_location"))
        absent = 0
        for cell, assays in mapping.items():
            if cell.startswith("_") or not isinstance(assays, dict):
                continue
            for mark, filename in assays.items():
                candidate = Path(filename)
                if not candidate.is_absolute():
                    candidate = Path(location) / filename
                if not candidate.exists():
                    absent += 1
        print(f"  Track directory  {location}")
        print(f"  Track config found, {absent} track file(s) missing")
        if absent:
            ok = False

    genome = Path(paths["ref_genome"])
    if not genome.exists():
        print(f"  Reference genome missing: {genome}")
        print("  Without it the sequence branch reads placeholder sequences")
        print("  and every result is meaningless. Download hg19 first.")
        ok = False
    else:
        index = genome.with_suffix(genome.suffix + ".fai")
        print(f"  hg19 found ({genome.stat().st_size / 1e9:.1f} GB), "
              f"index {'present' if index.exists() else 'MISSING'}")
        if not index.exists():
            print("  Build it once:")
            print(f"    python -c \"import pyfaidx; pyfaidx.Fasta('{genome}')\"")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_stage(stage, state):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{stage['name']}.log"
    command = expand_auto_runs(stage["command"])
    if command is None:
        print(f"  Skipping {stage['name']}: no finished runs to read yet.")
        return True

    state["stages"][stage["name"]] = {
        "status": "running",
        "started": datetime.now().isoformat(),
        "command": " ".join(command),
    }
    save_state(state)

    print(f"  $ {' '.join(command)}")
    print(f"  log: {log_path.relative_to(ROOT)}")
    began = time.time()

    # Stream to the console and the log at once, so a stage that is cut off
    # still leaves everything it printed.
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {' '.join(command)}\n# started {datetime.now()}\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace")
        for line in process.stdout:
            sys.stdout.write("    " + line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = process.wait()

    elapsed = time.time() - began
    produces = stage.get("produces")
    produced = (ROOT / produces).exists() if produces else True

    if code == 0 and produced:
        state["stages"][stage["name"]] = {
            "status": "done",
            "finished": datetime.now().isoformat(),
            "seconds": round(elapsed, 1),
            "command": " ".join(command),
        }
        save_state(state)
        print(f"  done in {timedelta(seconds=int(elapsed))}")
        return True

    reason = (f"exit code {code}" if code != 0
              else f"expected output {produces} was not written")
    state["stages"][stage["name"]] = {
        "status": "failed",
        "finished": datetime.now().isoformat(),
        "seconds": round(elapsed, 1),
        "reason": reason,
        "command": " ".join(command),
    }
    save_state(state)
    print(f"  FAILED after {timedelta(seconds=int(elapsed))}: {reason}")
    print(f"  The last lines of {log_path.relative_to(ROOT)}:")
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in tail[-15:]:
        print(f"    {line}")
    return False


def show_plan(plan, state):
    print(f"\n{'stage':<24}{'status':<10}{'estimate':>10}   why")
    print("-" * 78)
    total = 0
    for stage in plan:
        record = state["stages"].get(stage["name"], {})
        status = "done" if is_done(stage, state) else record.get("status", "pending")
        if status != "done":
            total += stage["minutes"]
        why = stage.get("why", "")
        print(f"{stage['name']:<32}{status:<10}{stage['minutes']:>7} min   {why}")
    print("-" * 86)
    print(f"{'remaining':<32}{'':<10}{total:>7} min   "
          f"about {total / 60:.1f} hours")


def main():
    parser = argparse.ArgumentParser(
        description="Run every experiment in order, resuming after interruptions.")
    parser.add_argument("--check", action="store_true",
                        help="verify the environment and the data, then stop")
    parser.add_argument("--list", action="store_true",
                        help="show the plan and what is already finished")
    parser.add_argument("--only", metavar="STAGE",
                        help="run one stage and stop")
    parser.add_argument("--from", dest="start_at", metavar="STAGE",
                        help="run this stage and everything after it")
    parser.add_argument("--redo", metavar="STAGE", action="append",
                        help="mark a stage unfinished so it runs again")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="seeds for every model")
    parser.add_argument("--extra-seeds", nargs="*", type=int, default=[3, 4],
                        help="further seeds for the baseline and KA only")
    parser.add_argument("--archive", metavar="NAME",
                        help="move all existing results to "
                             "results/archive/NAME/ and stop")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, without running it")
    parser.add_argument("--skip-checks", action="store_true",
                        help="do not verify the environment first")
    args = parser.parse_args()

    print("=" * 78)
    print("  SLIM: full experiment run")
    print("=" * 78)
    print(f"  Directory  {ROOT}")
    print(f"  Python     {platform.python_version()} on {platform.system()}")
    print(f"  Started    {datetime.now():%Y-%m-%d %H:%M}")
    print()

    if args.archive:
        return 0 if archive_results(args.archive) else 1

    plan = build_plan(args.seeds, args.extra_seeds)
    state = load_state()

    if args.redo:
        for name in args.redo:
            if name in state["stages"]:
                del state["stages"][name]
                print(f"  {name} marked unfinished; it will run again.")
            else:
                print(f"  {name} was not recorded as finished.")
        save_state(state)

    if args.list:
        show_plan(plan, state)
        return 0

    if not args.skip_checks:
        packages_ok = check_packages()
        gpu_ok = check_gpu()
        data_ok = check_data()
        print()
        if not packages_ok:
            print("Install the missing packages, then run this again.")
            return 1
        if not data_ok:
            print("Prepare the data as the guide describes, then run this "
                  "again.")
            print("To check only the environment: python run.py --check")
            return 1
        if not gpu_ok:
            print("No usable GPU was found. Continuing would take days.")
            print("Fix the driver or the torch build, then run this again.")
            return 1
        print("Environment and data look correct.\n")

    if args.check:
        show_plan(plan, state)
        return 0

    if args.only:
        chosen = [s for s in plan if s["name"] == args.only]
        if not chosen:
            print(f"No stage named {args.only}. Known stages:")
            for stage in plan:
                print(f"  {stage['name']}")
            return 1
        plan = chosen
    elif args.start_at:
        names = [s["name"] for s in plan]
        if args.start_at not in names:
            print(f"No stage named {args.start_at}.")
            return 1
        plan = plan[names.index(args.start_at):]

    show_plan(plan, state)
    if args.dry_run:
        print("\nDry run, nothing was executed.")
        return 0

    print("\nSafe to interrupt at any time. Run this command again to "
          "continue:")
    print("  python run.py\n")

    started = time.time()
    failures = []
    for index, stage in enumerate(plan, 1):
        header = f"[{index}/{len(plan)}] {stage['name']}"
        if is_done(stage, state) and not args.only:
            print(f"{header}: already finished, skipping")
            continue
        print(f"\n{header}  (about {stage['minutes']} min)")
        if stage.get("resumable"):
            print("  Resumable: an interrupted run continues from its last "
                  "completed epoch.")
        try:
            if not run_stage(stage, state):
                failures.append(stage["name"])
                if stage["name"] == "tests":
                    print("\nThe test suite failed, so the checkout is broken. "
                          "Stopping before")
                    print("any training starts. Send the log to the authors.")
                    return 1
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {stage['name']}.")
            print("Progress is saved. Continue with:  python run.py")
            return 130

    elapsed = time.time() - started
    print("\n" + "=" * 78)
    print(f"  Finished in {timedelta(seconds=int(elapsed))}")
    print("=" * 78)
    show_plan(build_plan(args.seeds, args.extra_seeds), load_state())

    if failures:
        print(f"\n  Stages that failed: {', '.join(failures)}")
        print(f"  Read the logs in {LOGS.relative_to(ROOT)}, then re-run "
              f"'python run.py' to retry them.")
        return 1

    print("\n  Everything finished. Send these back to the authors:")
    print("    results/            every prediction file, report and log")
    print("    figures/            the regenerated figures")
    print("  Zip the whole results directory; it is small, because model "
          "weights")
    print("  are not included and are not needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Isolated retry for the two unfinished official Batch-06 bugs.

Never edits Batch 06. Setup and preflight use zero Qwen calls. The new
independent run keeps the original cohort, source evidence and policy settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.graft_100_run import check_batch, check_preflight, graph_readiness, write_new
from scripts.audit_graft_phase3_multibug import inspect
from rave_fl.graft.runner import validate_config

BUGS = ["Math-69", "Time-22"]
ORIGINAL = Path("results/graft_100_bm25_recovery_v1")
TARGET = Path("results/graft_100_bm25_retry_2_v1")


def prepare():
    original = ORIGINAL.resolve()
    target = TARGET.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Retry workspace already exists; refusing to overwrite it")
    old_exp = original / "batches" / "batch_06" / "experiment"
    source_manifest = original / "batches" / "batch_06" / "prepared" / "manifest.json"
    source = json.loads(source_manifest.read_text())
    original_config = json.loads((old_exp / "config.json").read_text())
    if [c["bug"] for c in source["cases"]] != [
        "Chart-8", "Closure-106", "Lang-36", "Math-69", "Time-22"
    ]:
        raise ValueError("Unexpected original Batch-06 case identity or ordering")
    cohort = json.loads((original / "cohort.json").read_text())
    if source["cohort_selection_sha256"] != cohort["selection_sha256"]:
        raise ValueError("Frozen cohort differs from source evidence")
    retry_dataset = {
        **source,
        "cases": [c for c in source["cases"] if c["bug"] in BUGS],
    }
    if [c["bug"] for c in retry_dataset["cases"]] != BUGS:
        raise ValueError("Selected cases do not match declared retry IDs")
    config = json.loads(json.dumps(original_config))
    benchmark = config["benchmark"]
    if benchmark.get("backend") != "ollama" or benchmark.get("model") != "qwen2.5:7b":
        raise ValueError("Unexpected original model")
    if benchmark.get("bugs") != ["Chart-8", "Closure-106", "Lang-36", "Math-69", "Time-22"]:
        raise ValueError("Unexpected original benchmark bug list")
    benchmark.update({
        "dataset_file": str((target / "prepared_manifest.json").resolve()),
        "bugs": BUGS,
        "cache_dir": str((target / "experiment" / "cache").resolve()),
        "run_id": "graft_100_batch_06_retry_two_v1",
    })
    config = validate_config(config)
    # Create only fresh files. All source checkouts and observed evidence
    # remain the same, and we do not reuse the original run's cache/journal.
    target.mkdir(parents=True)
    write_new(target / "prepared_manifest.json", retry_dataset)
    write_new(target / "experiment" / "config.json", config)
    write_new(target / "retry_protocol.json", {
        "schema": "graft-100-batch06-independent-retry-v1",
        "original_batch": 6,
        "bugs": BUGS,
        "selection_sha256": cohort["selection_sha256"],
        "reason": "Batch06 provider response incomplete, usage unknown",
        "original_run": str((old_exp / "run").resolve()),
        "original_run_untouched": True,
        "isolated_cache_and_accounting": True,
        "no_bugs_replaced": True,
        "not_an_original_five_bug_batch_gate": True,
        "setup_model_calls": 0,
    })
    print("Retry workspace:", target)
    print("Only:", BUGS)
    print("Setup performed zero Qwen calls.")


def execute():
    target = TARGET.resolve()
    cfg = target / "experiment" / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError("First execute: python -m scripts.graft_100_retry_two setup")
    benchmark = json.loads(cfg.read_text())["benchmark"]
    if benchmark["bugs"] != BUGS:
        raise ValueError("Retry bug selection changed")
    check_preflight(cfg)
    graph_readiness(cfg)
    exp = cfg.parent
    run_dir = exp / "run"
    resume = (run_dir / "manifest.json").exists()
    command = (
        [sys.executable, "-m", "rave_fl.graft.cli", "resume",
         "--run-dir", str(run_dir), "--allow-inference"]
        if resume else
        [sys.executable, "-m", "rave_fl.graft.cli", "run", "--config",
         str(cfg), "--output", str(run_dir), "--allow-inference"]
    )
    with (exp / "model_run.log").open("a" if resume else "w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    print("MODEL RUN exit:", result.returncode, "log:", exp / "model_run.log")
    if not (run_dir / "results.json").exists():
        raise RuntimeError("Missing run results; preserve model_run.log")
    audit = inspect(run_dir)
    (exp / "paired_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    gate = check_batch(audit, BUGS)
    (exp / "completion_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print("TWO-BUG COMPLETION GATE:", gate["completed"])
    print("Missing:", gate["missing"], "Incomplete:", gate["incomplete"])
    if result.returncode or not gate["completed"]:
        raise SystemExit(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "preflight", "run"))
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args()
    if args.command == "setup":
        prepare()
    elif args.command == "preflight":
        target = TARGET.resolve()
        cfg = target / "experiment" / "config.json"
        check_preflight(cfg)
        graph_readiness(cfg)
    else:
        if not args.allow_inference:
            parser.error("Real Qwen inference requires --allow-inference")
        execute()


if __name__ == "__main__":
    main()

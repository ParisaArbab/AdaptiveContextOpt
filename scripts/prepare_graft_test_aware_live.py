"""Prepare opt-in, paired Phase-3 Time-4 test-aware graph ablations.

No inference. Same real buggy dataset, candidate pool, L1 Qwen, model digest,
and 4096-unit graph budget as the saved Phase-2 Time-4 run. Time-4 was already
inspected during algorithm development and MUST NOT be called held-out.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

FUSIONS = ["rrf", "suspicion_rank_blend"]
POLICIES = [
    "l1_anchor",
    "l1_graph_focused",
    "l1_graph_test_aware_nostop",
    "l1_graph_test_aware",
]


def prepare(source: Path, output: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Source configuration missing: {source}")
    raw = source.read_bytes()
    config = json.loads(raw)
    benchmark = config["benchmark"]
    graft = config["graft"]
    if benchmark.get("bugs") != ["Time-4"]:
        raise ValueError("Phase-3 development run is restricted to Time-4")
    if benchmark.get("model") != "qwen2.5:7b":
        raise ValueError("Keep the original Qwen2.5 7B model")
    if benchmark.get("candidate_provider") != "flexfl":
        raise ValueError("Keep original FlexFL method candidates")
    if benchmark.get("batch_size") != 1:
        raise ValueError("Keep original batch size")
    if benchmark.get("representation") != "l1" or benchmark.get("granularity") != "method":
        raise ValueError("Keep the original L1 method granularity")
    if graft.get("input_budget") != 4096 or not graft.get("guard_unsupported_graph_downgrade"):
        raise ValueError("Keep graph budget 4096 and historical guard")
    if not graft.get("failure_name_anchor"):
        raise ValueError("Keep the historical anchor setting")
    dataset = Path(benchmark.get("dataset_file", ""))
    if not dataset.is_file():
        raise FileNotFoundError(f"Time-4 dataset unavailable: {dataset}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    benchmark["policies"] = ["fixed_l1"]
    benchmark["fusions"] = FUSIONS
    benchmark["max_new_calls"] = 45
    benchmark["physical_budget"] = 60000
    benchmark["per_bug_physical_budget"] = 45000
    benchmark["run_id"] = "graft_phase3_time4_" + output.name
    config["graft_policies"] = POLICIES

    from rave_fl.graft.runner import validate_config
    validate_config(config)

    output.mkdir(parents=True)
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "schema": "graft-test-aware-time4-dev-v1",
        "source_config": str(source.resolve()),
        "source_sha256": sha256(raw).hexdigest(),
        "cohort": "Time-4, already inspected development instance; NOT held-out",
        "same_initial_evidence_and_model": True,
        "gold_in_online_decisions": False,
        "historical_focused_policy_unchanged": True,
        "tested_treatments": POLICIES,
        "fusions": FUSIONS,
        "selector_budget_proxy": 4096,
        "new_calls_during_preparation": 0,
        "max_new_provider_calls": 45,
        "max_physical_provider_tokens": 60000,
        "stopping_ablation": (
            "Same test-aware selector, same candidate order and evidence; "
            "only early stop rule differs between the two test-aware arms"
        ),
    }, indent=2) + "\n")
    print("Prepared:", path)
    print("Policies:", POLICIES)
    print("Fusions:", FUSIONS)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.source, args.output)

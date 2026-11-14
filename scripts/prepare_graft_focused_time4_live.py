"""Prepare a strictly paired Phase-2 Time-4 focused retrieval development run.

No downloads, no source modifications, no model calls.
Inputs are the exact verified Time-4 Phase-1 source and existing model settings.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path

FUSIONS = ["rrf", "suspicion_rank_blend"]
POLICIES = ["l1_anchor", "l1_graph_suspicion", "l1_graph_focused"]


def prepare(source: Path, output: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Missing original config: {source}")
    raw = source.read_bytes()
    conf = json.loads(raw)
    benchmark = conf["benchmark"]
    graft = conf["graft"]
    if benchmark.get("bugs") != ["Time-4"]:
        raise ValueError("Only already studied development bug Time-4 allowed")
    if benchmark.get("model") != "qwen2.5:7b" or benchmark.get("batch_size") != 1:
        raise ValueError("Original Qwen2.5:7b model and batch size must be unchanged")
    if benchmark.get("candidate_provider") != "flexfl":
        raise ValueError("Original FlexFL candidate provider required")
    if benchmark.get("representation") != "l1" or benchmark.get("granularity") != "method":
        raise ValueError("Original method-level L1 representation required")
    if graft.get("input_budget") != 4096:
        raise ValueError("Use the same evidence budget as the historical graph method")
    if not graft.get("guard_unsupported_graph_downgrade", True):
        raise ValueError("Keep graph downgrade guard enabled")
    if not graft.get("failure_name_anchor"):
        raise ValueError("Original failure-name anchor setting must remain unchanged")
    dataset = Path(benchmark.get("dataset_file", ""))
    if not dataset.is_file():
        raise FileNotFoundError(f"Validated Time-4 dataset missing: {dataset}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    # The old treatment and the new treatment share the same L1 inputs and
    # ranking controls, with only the graph selection/packing differing.
    benchmark["policies"] = ["fixed_l1"]
    benchmark["fusions"] = FUSIONS
    benchmark["run_id"] = f"graft_phase2_time4_{output.name}"
    benchmark["max_new_calls"] = 40
    benchmark["physical_budget"] = 55000
    benchmark["per_bug_physical_budget"] = 45000
    conf["graft_policies"] = POLICIES

    # Validate before creating experiment files.
    from rave_fl.graft.runner import validate_config
    validate_config(conf)

    output.mkdir(parents=True)
    target = output / "config.json"
    target.write_text(json.dumps(conf, indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps({
            "schema": "graft-phase2-time4-development-v1",
            "source_config": str(source.resolve()),
            "source_config_sha256": hashlib.sha256(raw).hexdigest(),
            "cohort": "Time-4, investigated development bug, NOT held out",
            "new_provider_calls_during_preparation": 0,
            "same_initial_l1_evidence": True,
            "same_candidate_provider_and_model": True,
            "ranking_fusions": FUSIONS,
            "graph_policies": POLICIES,
            "original_graph_baseline_unchanged": True,
            "gold_used_for_online_decisions": False,
            "graph_input_budget": 4096,
            "max_new_calls": 40,
            "physical_token_cap": 55000,
            "per_bug_physical_token_cap": 45000,
        }, indent=2) + "\n"
    )
    print("Prepared:", target)
    print("Policies:", POLICIES)
    print("Fusions:", FUSIONS)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)

"""Prepare an explicitly separate Phase-3 paired BM25/observed-test experiment.

Precondition: prepare_graft_phase3_real_bugs --collect succeeded for *all*
three frozen IDs. This does not retroactively replace the missing FlexFL data.
Method-level labels are unavailable here; only official modified-file proxies
are scored. No Qwen call occurs during preparation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

BUGS = ("Time-1", "Lang-1", "Math-1")
POLICIES = (
    "l1_anchor", "l1_graph_focused",
    "l1_graph_test_aware_nostop", "l1_graph_test_aware",
)
FUSIONS = ("rrf", "suspicion_rank_blend")


def prepare(source_config: Path, dataset: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config = json.loads(source_config.read_text())
    record = json.loads(dataset.read_text())
    cases = record.get("cases", [])
    if [r.get("bug") for r in cases] != list(BUGS):
        raise ValueError("All three predeclared bugs must be present, in fixed order")
    if not record.get("not_original_flexfl_condition"):
        raise ValueError("Must explicitly declare new BM25 condition")
    if any(r.get("evidence_kind") != "observed_failure" or
           not Path(r.get("evidence_file", "")).is_file()
           or not Path(r.get("checkout", "")).is_dir()
           for r in cases):
        raise ValueError("Authentic observed test files and checkouts required")
    if any(not r.get("gold", {}).get("files") or r["gold"].get("functions") for r in cases):
        raise ValueError("This cohort must have only real file-level proxy labels")
    b = config["benchmark"]
    g = config["graft"]
    if b["model"] != "qwen2.5:7b" or b["batch_size"] != 1 or g["input_budget"] != 4096:
        raise ValueError("Frozen Qwen2.5, batch 1 and graph budget 4096 required")
    if b["granularity"] != "method" or b["representation"] != "l1":
        raise ValueError("Keep method candidates with L1 representation")
    b["dataset_file"] = str(dataset.resolve())
    b["candidate_provider"] = "bm25"
    b["bugs"] = list(BUGS)
    b["policies"] = ["fixed_l1"]
    b["fusions"] = list(FUSIONS)
    b["require_observed_failure"] = True
    b["max_new_calls"] = 135
    b["physical_budget"] = 240000
    b["per_bug_physical_budget"] = 80000
    b["run_id"] = "graft_phase3_real_bugs_" + output.name
    config["graft_policies"] = list(POLICIES)
    from rave_fl.graft.runner import validate_config
    config = validate_config(config)
    output.mkdir(parents=True)
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "schema": "graft-phase3-3bug-paired-v1",
        "cohort": list(BUGS),
        "source_config_sha256": sha256(source_config.read_bytes()).hexdigest(),
        "prepared_manifest_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "candidate_condition": "bm25_not_original_flexfl",
        "evidence_condition": "full_test_source_plus_observed_failing_tests",
        "gold_condition": "file_only_official_modified_classes_proxy",
        "method_localization_accuracy_available": False,
        "fusions": list(FUSIONS),
        "policies": list(POLICIES),
        "graph_evidence_budget": 4096,
        "new_calls_during_preparation": 0,
        "known_physical_provider_token_cap": 240000,
        "provider_calls_cap": 135,
        "selects_gold_online": False,
        "development_or_exploratory_not_heldout": True,
    }, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("configs/graft_independent_qwen7b_control.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Prepared:", prepare(args.source, args.dataset, args.output))

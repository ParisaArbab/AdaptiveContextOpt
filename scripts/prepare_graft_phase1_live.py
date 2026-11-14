"""Prepare frozen-baseline vs opt-in GRAFT Phase-1 evaluation configuration.

Uses already prepared local Defects4J and FlexFL files, no downloads and no
provider calls. This experiment is a development-bug smoke, not held-out data.
"""
from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path


FUSIONS = ["rrf", "suspicion_rank_blend"]
POLICIES = ["l1_anchor", "l1_graph", "l1_graph_suspicion", "graft_adaptive"]


def prepare(source: Path, output: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Missing prior configuration: {source}")
    raw = source.read_bytes()
    conf = json.loads(raw)
    benchmark = conf["benchmark"]
    graft = conf["graft"]
    if benchmark.get("model") != "qwen2.5:7b" or benchmark.get("batch_size") != 1:
        raise ValueError("Verified Qwen2.5:7b with batch_size=1 is required")
    if benchmark.get("bugs") != ["Time-4"]:
        raise ValueError("The initial Phase-1 smoke must use predeclared development bug Time-4")
    if benchmark.get("candidate_provider") != "flexfl":
        raise ValueError("Use real FlexFL candidates, not synthetic retrieval")
    if not graft.get("failure_name_anchor", False):
        raise ValueError("Original and new arms must use identical failure-name setting")
    if graft.get("max_graph", 0) < 1 or graft.get("max_full", 0) < 1:
        raise ValueError("Graph and Full inspection caps must allow the ablation")
    if benchmark.get("max_full") != graft.get("max_full"):
        raise ValueError("Full-stage limits differ")
    if not benchmark.get("dataset_file"):
        raise ValueError("Existing dataset file is required")
    if not Path(benchmark["dataset_file"]).is_file():
        raise FileNotFoundError("Original dataset file is missing: " + benchmark["dataset_file"])
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    # Make a single immutable shared-journal evaluation. Prevent cost explosion.
    benchmark["policies"] = ["fixed_l1"]
    benchmark["fusions"] = FUSIONS
    benchmark["max_new_calls"] = 45
    benchmark["physical_budget"] = 60000
    benchmark["per_bug_physical_budget"] = 45000
    benchmark["run_id"] = "graft_phase1_compare_" + output.name
    conf["graft_policies"] = POLICIES
    output.mkdir(parents=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(conf, indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_config": str(source.resolve()),
                "source_config_sha256": sha256(raw).hexdigest(),
                "cohort": "Time-4 development only, NOT held out",
                "fusions": FUSIONS,
                "graph_policies": POLICIES,
                "legacy_controls_unchanged": True,
                "initial_sketches_shared_across_policies": True,
                "gold_used_for_online_decisions": False,
                "physical_call_limit": 45,
                "physical_token_limit": 60000,
                "per_bug_token_limit": 45000,
                "notes": "Frozen RRF and legacy graph controls; new optional fusion and graph-action treatments.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared:", config_path)
    print("Fusions:", FUSIONS)
    print("Policies:", ["fixed_l1"] + POLICIES)
    return config_path


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.source, args.output)

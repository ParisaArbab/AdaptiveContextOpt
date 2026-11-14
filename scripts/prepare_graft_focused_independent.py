"""Prepare an explicit, paired retrieval ablation on previously explored bugs.

Inputs Time-1, Lang-1, Math-1 were investigated before. This is an
independent-of-Time-4 exploratory cohort, NOT a clean held-out set.
No inference is performed by this preparation script.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

BUGS = ["Time-1", "Lang-1", "Math-1"]
FUSIONS = ["rrf", "suspicion_rank_blend"]
GRAPH_POLICIES = ["l1_anchor", "l1_graph_suspicion", "l1_graph_focused"]


def prepare(source: Path, output: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    value = json.loads(raw)
    bench = value["benchmark"]
    if bench.get("bugs") != BUGS:
        raise ValueError("Use the original predeclared Time-1/Lang-1/Math-1 order")
    if bench.get("model") != "qwen2.5:7b" or bench.get("batch_size") != 1:
        raise ValueError("Paired Qwen2.5 7B, batch size 1 required")
    if bench.get("candidate_provider") != "flexfl":
        raise ValueError("Real FlexFL candidates required")
    if not Path(bench["dataset_file"]).is_file():
        raise FileNotFoundError("Source manifest not found: " + str(bench["dataset_file"]))
    if value["graft"].get("input_budget") != 4096:
        raise ValueError("Preserve the 4096-unit evidence budget")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    value["benchmark"]["policies"] = ["fixed_l1"]
    value["benchmark"]["fusions"] = FUSIONS
    value["benchmark"]["run_id"] = "graft_focused_independent_" + output.name
    value["benchmark"]["max_new_calls"] = 115
    value["benchmark"]["physical_budget"] = 180000
    value["benchmark"]["per_bug_physical_budget"] = 60000
    value["graft_policies"] = GRAPH_POLICIES
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(value, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "source_config": str(source.resolve()),
        "source_config_sha256": sha256(raw).hexdigest(),
        "bugs": BUGS,
        "cohort": "previously explored, independent-of-Time-4, NOT held-out",
        "fixed_budgets_before_inference": True,
        "same_model_candidate_evidence_pool": True,
        "fusions": FUSIONS,
        "policies": GRAPH_POLICIES,
        "gold_used_for_online_selection": False,
        "physical_token_cap": 180000,
        "max_new_calls": 115,
        "new_calls_during_preparation": 0,
    }, indent=2) + "\n")
    return output / "config.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print("Prepared:", prepare(args.source, args.output))

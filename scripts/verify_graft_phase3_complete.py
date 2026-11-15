"""Evaluate Phase-3 completion without model calls, labels leakage, or estimates.

This gate reads only the archived paired audit. It does not change any saved
rankings or infer missing calls, and cannot mark incomplete experiments complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BUGS = ("Lang-1", "Math-1", "Time-1")
FUSIONS = ("rrf", "suspicion_rank_blend")
PAIRED_POLICIES = (
    "l1_anchor",
    "l1_graph_focused",
    "l1_graph_test_aware_nostop",
    "l1_graph_test_aware",
)


def check(audit: dict) -> dict:
    records = audit.get("records", [])
    index = {
        (r.get("bug"), r.get("fusion"), r.get("policy")): r
        for r in records
    }
    missing = []
    incomplete = []
    graph_missing = []
    for bug in BUGS:
        for fusion in FUSIONS:
            for policy in PAIRED_POLICIES:
                key = (bug, fusion, policy)
                item = index.get(key)
                if item is None:
                    missing.append(list(key))
                elif item.get("status") != "completed":
                    incomplete.append({
                        "bug": bug,
                        "fusion": fusion,
                        "policy": policy,
                        "status": item.get("status"),
                    })
            # Focused and no-stop policies should actually inspect graph
            # evidence. Test-aware stop policy may correctly make zero calls.
            for policy in ("l1_graph_focused", "l1_graph_test_aware_nostop"):
                item = index.get((bug, fusion, policy))
                if item and not item.get("graph_call_count", 0):
                    graph_missing.append([bug, fusion, policy])

    pair_rows = audit.get("paired_comparisons", [])
    paired = {
        (p.get("bug"), p.get("fusion"))
        for p in pair_rows if p.get("status") == "paired"
    }
    expected = {(b, f) for b in BUGS for f in FUSIONS}
    accounting = audit.get("physical_provider_accounting", {})
    trustworthy_usage = (
        accounting.get("synthetic") is False
        and accounting.get("usage_unknown") is False
        and accounting.get("pending") is False
        and type(accounting.get("new_calls")) is int
        and type(accounting.get("known_provider_tokens")) is int
    )
    all_paired = bool(audit.get("all_pairs_completed")) and paired == expected
    completed = (
        not missing and not incomplete and not graph_missing and all_paired
        and trustworthy_usage
    )
    return {
        "schema": "graft-phase3-real-graph-completion-gate-v1",
        "all_six_bug_fusion_pairs_completed": all_paired,
        "genuine_graph_evidence_queried": not graph_missing,
        "completed_real_graph_comparison": completed,
        "missing_policy_rows": missing,
        "incomplete_policy_rows": incomplete,
        "graph_focused_or_nostop_without_graph_calls": graph_missing,
        "physical_provider_accounting": accounting,
        "new_model_calls_for_gate": 0,
        "interpretation": (
            "Real, file-level modified-class-proxy development comparison only. "
            "No method-level accuracy, causal graph gains, or held-out result "
            "is asserted by this gate."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing audit: {args.output}")
    audit = json.loads(args.audit.read_text())
    if audit.get("schema") != "graft-phase3-multibug-audit-v1":
        parser.error("Unexpected audit schema")
    result = check(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Completed real graph comparison:",
        result["completed_real_graph_comparison"],
    )
    print("Completed paired bug/fusion sets:", result["all_six_bug_fusion_pairs_completed"])
    print("Graph-focused/no-stop actually queried graph:", result["genuine_graph_evidence_queried"])
    if result["incomplete_policy_rows"]:
        print("Incomplete policy arms:", len(result["incomplete_policy_rows"]))
    if result["graph_focused_or_nostop_without_graph_calls"]:
        print("Missing graph treatments:", result["graph_focused_or_nostop_without_graph_calls"])
    return 0 if result["completed_real_graph_comparison"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

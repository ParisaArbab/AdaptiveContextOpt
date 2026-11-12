"""Time-1: matched-token-proxy experiment with one truly retrieved helper.

The candidate pool and retrieved method are frozen. Both are charged using
the SAME regex token proxy, unlike earlier mixed L1/source cost comparisons.
A source declaration has been verified at the exact buggy Git revision,
but the caller-to-callee semantic binding remains unverified.

Conditions: anchor-first alone; forced source addition; confidence-gated
source addition (abstain with unverified binding). No gold patch or LLM.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .anchor_budget import anchor_first_selection

SNAPSHOT = Path("results/rave_fl_iteration_d/independent_live/Time-1/input_snapshot.json")
SOURCE_RECORD = Path("data/aitech/time1_verified_dependency.json")
BUDGETS = (300, 450, 600, 800, 1000)


def token_proxy(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def select_with_optional(
    costs: list[int], anchors: list[int], budget: int,
    helper_cost: int, *, force_helper: bool,
) -> tuple[set[int], int, bool]:
    selected = set()
    spent = 0
    for i in anchors:
        if i not in selected and spent + costs[i] <= budget:
            selected.add(i)
            spent += costs[i]
    included = False
    if force_helper and spent + helper_cost <= budget:
        spent += helper_cost
        included = True
    for i, cost in enumerate(costs):
        if i not in selected and spent + cost <= budget:
            selected.add(i)
            spent += cost
    return selected, spent, included


def run(
    snapshot: dict, verified: dict,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict:
    if not verified.get("source_declaration_verified"):
        raise ValueError("Retrieved source declaration is not verified")
    if verified.get("compiler_binding_verified") is not False:
        raise ValueError("Binding status must remain explicitly unverified")
    if verified.get("bug") != "Time-1":
        raise ValueError("Historical source belongs to another bug")
    if verified.get("patch_gold_used_for_request") is not False:
        raise ValueError("Unknown gold use in retrieval")
    candidates = snapshot["candidates"]
    ids = [c["function_id"] for c in candidates]
    costs = [token_proxy(c["levels"][1]) for c in candidates]
    helper_cost = token_proxy(verified["method_source"])
    helper_id = verified["target"]
    context = snapshot["context"]
    anchors = [
        i for i, identifier in enumerate(ids)
        if identifier.split("(", 1)[0].rsplit(".", 1)[-1] + "(" in context
    ]
    rows = []
    for budget in budgets:
        for condition in ("anchor_first", "force_verified_source", "gate_unverified_binding"):
            forced = condition == "force_verified_source"
            selected, used, included = select_with_optional(
                costs, anchors, budget, helper_cost, force_helper=forced)
            rows.append({
                "condition": condition,
                "budget_regex_proxy": budget,
                "selected_regex_proxy": used,
                "failure_name_anchors_kept": len(selected.intersection(anchors)),
                "failure_name_anchors_total": len(anchors),
                "retrieved_helper_included": included,
                "helper_proxy_tokens": helper_cost,
                "source_declaration_verified": True,
                "semantic_binding_verified": False,
                "selected_ids": [ids[i] for i in sorted(selected)] +
                                ([helper_id] if included else []),
            })
    return {
        "bug": "Time-1",
        "scope": "Archived real buggy-source snippets and one pinned historical helper; selector-only",
        "buggy_commit": verified["buggy_revision"],
        "retrieved_source_path": verified["source_path"],
        "retrieved_source_blob_sha": verified["source_blob_sha"],
        "request_target": helper_id,
        "source_declaration_verified": True,
        "semantic_binding_verified": False,
        "gold_patch_used": False,
        "llm_executed": False,
        "token_proxy": "regex words and punctuation applied equally to all candidates and helper",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/time1_verified_retrieval.json"))
    args = parser.parse_args()
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    verified = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    result = run(snapshot, verified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(f"{row['budget_regex_proxy']:4} {row['condition']:25} "
              f"anchors={row['failure_name_anchors_kept']}/{row['failure_name_anchors_total']} "
              f"helper={row['retrieved_helper_included']} "
              f"cost={row['selected_regex_proxy']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

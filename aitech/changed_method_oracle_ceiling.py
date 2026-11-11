"""Posthoc oracle ceiling for changed-method L1 source availability.

The oracle sees official patch changed-method labels. It is a SCORING
DIAGNOSTIC ONLY and must NEVER select evidence for an LLM or tune a
policy on a held-out set. It answers whether an existing candidate's
full L1 sketch *could* fit under a given stored proxy-token budget.

This is not fault localization or LLM accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .failure_conditioned_study import BUG_INPUTS, BUDGETS


def oracle_ceiling(candidates: list[dict], changed_methods: set[str],
                   budget: int) -> dict:
    if budget < 0:
        raise ValueError("negative budget")
    pool = [
        (int(item["features"]["l1_tokens"]), item["function_id"])
        for item in candidates if item["function_id"] in changed_methods
    ]
    if any(cost <= 0 for cost, _ in pool):
        raise ValueError("invalid cost")
    # Every positive-cost changed method contributes one covered gold entity.
    # Taking the cheapest first maximizes cardinality at fixed budget.
    pool.sort(key=lambda x: (x[0], x[1]))
    selected = []
    remaining = budget
    for cost, identifier in pool:
        if cost <= remaining:
            selected.append(identifier)
            remaining -= cost
    return {
        "budget_proxy": budget,
        "gold_total": len(changed_methods),
        "gold_present_in_pool": len(pool),
        "oracle_gold_l1_hits": len(selected),
        "oracle_min_tokens": budget - remaining,
        "oracle_gold_ids": selected,
        "missing_from_pool": sorted(changed_methods - {name for _, name in pool}),
    }


def compare(frozen: dict, labels: dict, sources: dict[str, Path] = BUG_INPUTS) -> dict:
    if labels.get("scoring_only") is not True:
        raise ValueError("labels must be scoring only")
    if frozen.get("gold_used") is not False:
        raise ValueError("frozen selection accessed labels")
    per_case = {x["bug"]: x for x in frozen["cases"]}
    rows = []
    for bug, path in sources.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        gold = set(labels["bugs"][bug]["changed_methods"])
        row_by_budget = {x["budget_proxy"]: x for x in per_case[bug]["rows"]}
        for budget in BUDGETS:
            bound = oracle_ceiling(data["candidates"], gold, budget)
            row = row_by_budget[budget]
            baseline_hits = len(gold & set(row["single_selected_ids"]))
            tiered_hits = len(gold & set(row["tiered_l1_ids"]))
            rows.append({
                "bug": bug,
                "budget_proxy": budget,
                **bound,
                "baseline_l1_hits": baseline_hits,
                "tiered_l1_hits": tiered_hits,
                "baseline_headroom": bound["oracle_gold_l1_hits"] - baseline_hits,
                "tiered_headroom": bound["oracle_gold_l1_hits"] - tiered_hits,
            })
    return {
        "scope": "Posthoc oracle upper bound on three previously studied Defects4J bugs",
        "oracle_uses_gold": True,
        "oracle_not_a_deployable_selector": True,
        "llm_ran": False,
        "budget_unit": "stored L1 sketch proxy estimates",
        "rows": rows,
        "limitations": [
            "Gold labels used intentionally, cannot be an inference method",
            "Changed-method labels are imperfect fault location proxies",
            "Only three previously seen cases, not a clean test set",
            "Oracle is a ceiling on availability of L1 evidence, not localization accuracy",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path(
        "data/aitech/three_bug_posthoc_patch_labels.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    result = compare(frozen, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in result["rows"]:
        print(r["bug"], r["budget_proxy"],
              "oracle", r["oracle_gold_l1_hits"],
              "baseline", r["baseline_l1_hits"],
              "tiered", r["tiered_l1_hits"])
    print("Saved", args.output)


if __name__ == "__main__":
    main()

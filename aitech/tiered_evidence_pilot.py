"""Tiered Java evidence under token budgets, no gold labels at selection.

Candidate methods have pre-existing L0 (identity/signature) and L1
(lossy source sketch) representations. Preserve failing-test anchors
at L0, add rank-ordered L0 candidates, then optionally upgrade selected
methods to L1 when affordable. The policy never sees patched locations.

WARNING: a gold method present only at L0 is NOT equivalent to having
its detailed body or successfully localizing it. Report tiers separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .risk_gated_pilot import SOURCES, BUDGETS


def select_tiers(snapshot: dict, budget: int, mode: str) -> dict:
    items = snapshot["candidates"]
    ids = [x["function_id"] for x in items]
    if mode not in ("l1_only", "tiered_l0", "tiered_upgrade", "anchor_detail_first"):
        raise ValueError("invalid representation policy")
    if budget < 0:
        raise ValueError("negative budget")
    anchors = [
        i for i, identifier in enumerate(ids)
        if identifier.split("(", 1)[0].rsplit(".", 1)[-1] + "("
        in snapshot["context"]
    ]
    l0 = [int(item["features"]["l0_tokens"]) for item in items]
    l1 = [int(item["features"]["l1_tokens"]) for item in items]
    if any(a <= 0 or b < a for a, b in zip(l0, l1)):
        raise ValueError("invalid or inconsistent tier costs")
    choices: dict[int, str] = {}
    used = 0
    order = list(dict.fromkeys(anchors + list(range(len(items)))))
    if mode == "anchor_detail_first":
        # Keep failure anchor identity coverage before paying for detail.
        for i in anchors:
            if used + l0[i] <= budget:
                choices[i] = "L0"
                used += l0[i]
        # Upgrade only anchored methods before admitting optional L0 names.
        for i in anchors:
            if choices.get(i) == "L0" and used + l1[i] - l0[i] <= budget:
                choices[i] = "L1"
                used += l1[i] - l0[i]
        for i in range(len(items)):
            if i not in choices and used + l0[i] <= budget:
                choices[i] = "L0"
                used += l0[i]
        # Later affordable upgrades are optional.
        for i in range(len(items)):
            if choices.get(i) == "L0" and used + l1[i] - l0[i] <= budget:
                choices[i] = "L1"
                used += l1[i] - l0[i]
    else:
        for i in order:
            cost = l1[i] if mode == "l1_only" else l0[i]
            if used + cost <= budget:
                choices[i] = "L1" if mode == "l1_only" else "L0"
                used += cost
        if mode == "tiered_upgrade":
            for i in order:
                if choices.get(i) != "L0":
                    continue
                incremental = l1[i] - l0[i]
                if used + incremental <= budget:
                    choices[i] = "L1"
                    used += incremental
    return {
        "method": mode,
        "budget_proxy": budget,
        "used_proxy_tokens": used,
        "anchors_total": len(anchors),
        "anchors_at_any_level": sum(i in choices for i in anchors),
        "anchors_with_l1_details": sum(choices.get(i) == "L1" for i in anchors),
        "candidate_id_tier": {ids[i]: choices[i] for i in sorted(choices)},
        "L0_candidate_count": sum(t == "L0" for t in choices.values()),
        "L1_candidate_count": sum(t == "L1" for t in choices.values()),
    }


def experiment(sources: dict[str, Path] = SOURCES) -> dict:
    cases = []
    for bug, path in sources.items():
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        cases.append({
            "bug": bug,
            "source_path": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": [
                {"bug": bug, **select_tiers(snapshot, budget, mode)}
                for budget in BUDGETS
                for mode in ("l1_only", "tiered_l0", "tiered_upgrade", "anchor_detail_first")
            ],
        })
    return {
        "scope": "Three previously examined Defects4J archived source pools, tier-selection only",
        "methods": ["l1_only", "tiered_l0", "tiered_upgrade", "anchor_detail_first"],
        "gold_labels_read": False,
        "llm_executed": False,
        "budget_unit": "precomputed L0/L1 token proxy, NOT model tokenizer",
        "critical_caveat": "L0 presence is method-name/signature coverage, NOT code-body evidence or localization accuracy",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/tiered_evidence.json"))
    args = parser.parse_args()
    result = experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        for row in case["rows"]:
            print(f"{case['bug']} {row['budget_proxy']:3} {row['method']:14} "
                  f"anchors={row['anchors_at_any_level']}/{row['anchors_total']} "
                  f"detailed={row['anchors_with_l1_details']} "
                  f"L0={row['L0_candidate_count']} L1={row['L1_candidate_count']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

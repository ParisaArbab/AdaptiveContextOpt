"""Paired counterfactual analysis of the real archived risk-gated pilot.

Do not mistake retained failure-name anchors or changed context for actual
fault-localization accuracy. This study uses only frozen source ranks,
L1 sketch proxy costs and candidate call edges, without gold patches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIR = ("anchor_first", "risk_gated")
BASELINES = ("rank_fill", "anchor_first", "always_expand", "risk_gated")


def analyze(result: dict) -> dict:
    out = []
    for case in result["cases"]:
        groups: dict[int, dict[str, dict]] = {}
        for row in case["rows"]:
            budget = row["budget_proxy"]
            entries = groups.setdefault(budget, {})
            if row["method"] in entries:
                raise ValueError("duplicate policy at budget")
            entries[row["method"]] = row
        for budget, arms in sorted(groups.items()):
            if set(arms) != set(BASELINES):
                raise ValueError("missing policy at budget")
            a, b = (arms[x] for x in PAIR)
            if a["used_proxy_tokens"] > budget or b["used_proxy_tokens"] > budget:
                raise ValueError("selected evidence exceeded matched budget")
            selected_a, selected_b = set(a["selected_ids"]), set(b["selected_ids"])
            out.append({
                "bug": case["bug"],
                "budget_proxy": budget,
                "arity_edges_available": case["arity_sketch_edges"],
                "graph_promoted_choices": b["graph_promoted_choices"],
                "risk_changes_context": selected_a != selected_b,
                "risk_only_ids": sorted(selected_b - selected_a),
                "anchor_only_ids": sorted(selected_a - selected_b),
                "risk_anchor_delta": (b["failure_name_anchors_kept"] -
                                      a["failure_name_anchors_kept"]),
                "risk_minus_anchor_missing_edges": (
                    b["missing_arity_sketch_edges"] -
                    a["missing_arity_sketch_edges"]
                ),
                "risk_minus_anchor_proxy_tokens": (
                    b["used_proxy_tokens"] - a["used_proxy_tokens"]
                ),
                "risk_anchors_vs_rank_fill": (
                    b["failure_name_anchors_kept"] -
                    arms["rank_fill"]["failure_name_anchors_kept"]
                ),
            })
    return {
        "scope": "paired selector-only analysis; no downstream model run",
        "comparison": "risk_gated minus anchor_first at same stored L1 proxy budget",
        "rows": out,
        "paired_budget_conditions": len(out),
        "context_changed_conditions": sum(x["risk_changes_context"] for x in out),
        "graph_promoted_choices": sum(x["graph_promoted_choices"] for x in out),
        "anchor_gains_vs_anchor_first": sum(x["risk_anchor_delta"] > 0 for x in out),
        "anchor_losses_vs_anchor_first": sum(x["risk_anchor_delta"] < 0 for x in out),
        "conditions_without_any_arity_edges": sum(
            x["arity_edges_available"] == 0 for x in out
        ),
        "interpretation": (
            "A changed context does not imply better fault localization. "
            "Graph confidence is uncalibrated, no gold patch or LLM scored."
        ),
    }


def report(summary: dict) -> str:
    lines = [
        "# Risk-gated dependency retrieval: paired diagnostic",
        "",
        "This report compares risk-gated and anchor-first selection on frozen "
        "archived Defects4J candidates. No LLM localization was performed.",
        "",
        f"Conditions compared: {summary['paired_budget_conditions']}",
        f"Contexts changed: {summary['context_changed_conditions']}",
        f"Graph-promoted choices: {summary['graph_promoted_choices']}",
        f"Anchor gains vs anchor-first: {summary['anchor_gains_vs_anchor_first']}",
        f"Anchor losses vs anchor-first: {summary['anchor_losses_vs_anchor_first']}",
        f"Conditions with zero arity-sketch edges: "
        f"{summary['conditions_without_any_arity_edges']}",
        "",
        "| Bug | Proxy budget | Changed context? | Graph promotions | Anchor delta | Missing edge delta | Proxy token delta |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for x in summary["rows"]:
        lines.append(
            f"| {x['bug']} | {x['budget_proxy']} | "
            f"{'Yes' if x['risk_changes_context'] else 'No'} | "
            f"{x['graph_promoted_choices']} | {x['risk_anchor_delta']:+d} | "
            f"{x['risk_minus_anchor_missing_edges']:+d} | "
            f"{x['risk_minus_anchor_proxy_tokens']:+d} |"
        )
    lines.extend([
        "",
        "## Scientific limitations",
        "",
        "- Existing bugs have been examined before; Time-25 is not untouched test data.",
        "- Graph edges are L1 sketch arity candidates, not compiler-verified.",
        "- The 0.8 edge confidence is not calibrated against real edge precision.",
        "- Cost figures are sketch proxies, not model token counts.",
        "- Anchor-name recall is only a debugging signal, not true fault location.",
        "- No gold label, LLM localization, or patch success measured.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "paired_report.md").write_text(
        report(summary), encoding="utf-8")
    print(report(summary))


if __name__ == "__main__":
    main()

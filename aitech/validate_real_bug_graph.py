"""Check reproducibility and interpretation of a real Defects4J compiler study.

Separate two questions:
1. Is the experimental output structurally valid and reproducible?
2. Is there any evidence that the *graph* changed the selected context?

Neither implies improved fault localization. No gold labels are read.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REQUIRED_METHODS = {"rank_fill", "anchor_first", "compiler_graph_first"}


def validate(record: dict) -> dict:
    failures: list[str] = []
    if record.get("status") != "compiler_attributed_static_targets":
        failures.append("compiler attribution did not complete")
    if record.get("gold_patch_used") is not False:
        failures.append("gold-label use is unknown or enabled")

    hashes = record.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        failures.append("source file hashes missing")
    elif not all(isinstance(h, str) and len(h) == 64 and
                 all(c in "0123456789abcdef" for c in h.lower())
                 for h in hashes.values()):
        failures.append("invalid source SHA-256")

    resolved = record.get("resolved_static_calls")
    unresolved = record.get("unresolved_calls")
    if type(resolved) is not int or resolved < 0:
        failures.append("resolved call count missing or invalid")
    if type(unresolved) is not int or unresolved < 0:
        failures.append("unresolved call count missing or invalid")

    mapping = record.get("candidate_mapping")
    if not isinstance(mapping, dict):
        failures.append("frozen candidate evidence mapping missing")
        mapping = {}

    candidate_hash = mapping.get("candidate_source_sha256")
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
        failures.append("candidate snapshot SHA-256 missing")

    count = mapping.get("candidate_count")
    if type(count) is not int or count < 1:
        failures.append("candidate count missing or invalid")

    edges = mapping.get("edges")
    if not isinstance(edges, list):
        failures.append("candidate graph edges missing")
        edges = []
    for index, edge in enumerate(edges):
        if not (isinstance(edge, list) and len(edge) == 2
                and all(type(v) is int and 0 <= v < count for v in edge)
                if type(count) is int and count > 0 else False):
            failures.append(f"edge {index}: invalid candidate index")

    mapped_calls = mapping.get("mapped_calls")
    if type(mapped_calls) is not int or mapped_calls < 0:
        failures.append("mapped call count missing or invalid")

    comparisons = mapping.get("selection_comparison")
    if not isinstance(comparisons, list) or not comparisons:
        failures.append("matched-budget baseline comparison missing")
        comparisons = []
    by_budget: dict[int, dict[str, dict]] = defaultdict(dict)
    for index, row in enumerate(comparisons):
        if not isinstance(row, dict):
            failures.append(f"row {index}: not a record")
            continue
        method, budget = row.get("method"), row.get("budget_proxy")
        tokens = row.get("selected_proxy_tokens")
        selected = row.get("selected_ids")
        retained = row.get("failure_name_anchors_kept")
        total = row.get("failure_name_anchors_total")
        if (method not in REQUIRED_METHODS or type(budget) is not int
                or budget < 0):
            failures.append(f"row {index}: invalid method or budget")
            continue
        if type(tokens) is not int or not (0 <= tokens <= budget):
            failures.append(f"row {index}: budget exceeded or malformed")
        if (type(retained) is not int or type(total) is not int
                or not (0 <= retained <= total)):
            failures.append(f"row {index}: impossible anchor count")
        if not isinstance(selected, list) or any(not isinstance(s, str)
                                                  for s in selected):
            failures.append(f"row {index}: selected evidence IDs missing")
        if method in by_budget[budget]:
            failures.append(f"budget {budget}: duplicate method {method}")
        by_budget[budget][method] = row

    if not by_budget:
        failures.append("no valid matched-budget rows")
    for budget, methods in sorted(by_budget.items()):
        if set(methods) != REQUIRED_METHODS:
            failures.append(f"budget {budget}: missing method comparison")

    valid = not failures
    # A correct run can report zero mapped edges. That is a negative
    # result about graph coverage, not evidence that a graph improved selection.
    graph_available = valid and bool(edges) and mapped_calls > 0
    graph_changed_context = (
        graph_available and any(
            set(rows["compiler_graph_first"]["selected_ids"])
            != set(rows["anchor_first"]["selected_ids"])
            for rows in by_budget.values()
        )
    )
    if not graph_available:
        reason = "no compiler-resolved edges mapped into frozen candidates"
    elif not graph_changed_context:
        reason = "compiler graph did not change selected evidence at any budget"
    else:
        reason = "graph changed selected evidence; downstream quality still unmeasured"

    return {
        "valid_for_downstream_evidence_analysis": valid,
        "graph_available_for_comparison": graph_available,
        "graph_changed_selected_evidence": graph_changed_context,
        "graph_effect_interpretation": reason,
        "provenance_file_count": len(hashes) if isinstance(hashes, dict) else 0,
        "resolved_static_calls": resolved,
        "unresolved_calls": unresolved,
        "candidate_graph_edges": len(edges),
        "mapped_call_count": mapped_calls,
        "mapping_ambiguous_type_count": mapping.get(
            "ambiguous_short_type_identity_count", 0),
        "comparison_budgets": sorted(by_budget),
        "comparison_rows": len(comparisons),
        "reasons": failures,
        "not_validated": [
            "true Java call graph completeness",
            "fault localization accuracy",
            "LLM token usage",
            "patch success",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    result = validate(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["valid_for_downstream_evidence_analysis"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

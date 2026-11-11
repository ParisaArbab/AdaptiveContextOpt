"""Validate a real Defects4J compiler graph result before scientific use.

This checks provenance, data completeness and fair evidence budgets.
It does NOT claim edge correctness, localization improvement or LLM success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(record: dict) -> dict:
    failures: list[str] = []
    if record.get("status") != "compiler_attributed_static_targets":
        failures.append("compiler attribution did not complete")
    if record.get("gold_patch_used") is not False:
        failures.append("gold-label use is unknown or enabled")
    files = record.get("source_hashes")
    if not isinstance(files, dict) or not files:
        failures.append("source file hashes missing")
    elif not all(isinstance(h, str) and len(h) == 64 for h in files.values()):
        failures.append("invalid source SHA-256")
    if not isinstance(record.get("resolved_static_calls"), int):
        failures.append("resolved call count missing")
    if not isinstance(record.get("unresolved_calls"), int):
        failures.append("unresolved call count missing")
    mapping = record.get("candidate_mapping")
    if not isinstance(mapping, dict):
        failures.append("frozen candidate evidence mapping missing")
        mapping = {}
    if not isinstance(mapping.get("candidate_source_sha256"), str) or len(mapping.get("candidate_source_sha256", "")) != 64:
        failures.append("candidate snapshot SHA-256 missing")
    comparisons = mapping.get("selection_comparison")
    if not isinstance(comparisons, list) or not comparisons:
        failures.append("matched-budget baseline comparison missing")
        comparisons = []
    for index, row in enumerate(comparisons):
        try:
            if not (0 <= row["selected_proxy_tokens"] <= row["budget_proxy"]):
                failures.append(f"row {index}: budget exceeded")
            if row["failure_name_anchors_kept"] > row["failure_name_anchors_total"]:
                failures.append(f"row {index}: impossible anchor count")
        except (TypeError, KeyError):
            failures.append(f"row {index}: malformed budget row")
    methods = {r.get("method") for r in comparisons}
    expected = {"rank_fill", "anchor_first", "compiler_graph_first"}
    if methods != expected:
        failures.append("missing baseline policy: " + str(sorted(expected - methods)))
    summary = {
        "valid_for_downstream_evidence_analysis": not failures,
        "provenance_file_count": len(files) if isinstance(files, dict) else 0,
        "resolved_static_calls": record.get("resolved_static_calls"),
        "unresolved_calls": record.get("unresolved_calls"),
        "mapped_call_count": mapping.get("mapped_calls"),
        "mapping_ambiguous_type_count": mapping.get("ambiguous_short_type_identity_count", 0),
        "comparison_rows": len(comparisons),
        "reasons": failures,
        "not_validated": [
            "true Java call graph completeness",
            "fault localization accuracy",
            "LLM token usage",
            "patch success",
        ],
    }
    return summary


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

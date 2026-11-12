"""Readable, non-overclaiming report from a frozen real-bug compiler record.

No gold labels, no LLM, and no new dependency extraction. The report
explicitly distinguishes graph availability, changed context and actual
fault-localization improvement, which has NOT been measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate_real_bug_graph import validate

METHODS = ("rank_fill", "anchor_first", "compiler_graph_first")


def render(record: dict) -> str:
    status = validate(record)
    mapping = record.get("candidate_mapping") or {}
    rows = mapping.get("selection_comparison") or []
    lines = [
        "# AITECH real-bug compiler evidence diagnostic",
        "",
        f"**Result structurally valid:** {status['valid_for_downstream_evidence_analysis']}",
        f"**Mapped graph available:** {status['graph_available_for_comparison']}",
        f"**Graph changed selected evidence:** {status['graph_changed_selected_evidence']}",
        "",
        f"Interpretation: {status['graph_effect_interpretation']}",
        "",
        "## Provenance and source coverage",
        "",
        f"- Source files hashed: {status['provenance_file_count']}",
        f"- Compiler-attributed calls: {status['resolved_static_calls']}",
        f"- Calls without static binding: {status['unresolved_calls']}",
        f"- Calls mapped to ranked candidates: {status['mapped_call_count']}",
        f"- Unique candidate graph edges: {status['candidate_graph_edges']}",
        f"- Ambiguous short-type identities: {status['mapping_ambiguous_type_count']}",
        "",
        "## Same-budget evidence selection",
        "",
        "| Budget proxy | Selector | Selected proxy tokens | Failure-name anchors | Unresolved mapped links |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x.get("budget_proxy", -1),
                                           METHODS.index(x["method"]) if x.get("method") in METHODS else 99)):
        lines.append(
            f"| {row.get('budget_proxy', 'NA')} | {row.get('method', 'NA')} "
            f"| {row.get('selected_proxy_tokens', 'NA')} "
            f"| {row.get('failure_name_anchors_kept', 'NA')}/"
            f"{row.get('failure_name_anchors_total', 'NA')} "
            f"| {row.get('unresolved_mapped_support', 'NA')} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "Graph-aware context selection is not a fault-localization result.",
        "A different selected evidence set does not prove it contains the bug.",
        "The budget counts are stored L1 sketch proxies, not model prompt tokens.",
        "A complete source/compiler analysis may still miss dynamic dispatch and reflection.",
        "No gold-patch evaluation or LLM inference was performed by this workflow.",
    ])
    if status["reasons"]:
        lines.extend(["", "## Acceptance failures", ""])
        lines.extend("- " + reason for reason in status["reasons"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote diagnostic report: {args.output}")


if __name__ == "__main__":
    main()

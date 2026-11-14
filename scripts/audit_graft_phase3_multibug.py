"""Summarize real paired Phase-3 results without any model calls.

Gold is used ONLY for offline evaluation already stored in results.json.
File-level labels here are official modified-class proxies, not method gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

PAIRS = ("l1_anchor", "l1_graph_focused",
         "l1_graph_test_aware_nostop", "l1_graph_test_aware")
FUSIONS = ("rrf", "suspicion_rank_blend")


def inspect(root: Path) -> dict:
    rows = json.loads((root / "results.json").read_text())
    accounting = json.loads((root / "accounting.json").read_text())
    output = []
    for row in rows:
        if row.get("policy") not in PAIRS or row.get("fusion") not in FUSIONS:
            continue
        metrics = (row.get("metrics") or {}).get("java_strict_v2") or {}
        file_metrics = metrics.get("file")
        method_metrics = metrics.get("method")
        calls = row.get("arm", {}).get("calls", [])
        graph_calls = [c for c in calls if c.get("stage") == "graph"]
        trace = row.get("arm", {}).get("trace", [])
        stops = [
            option.get("candidate")
            for event in trace if event.get("phase") == "route"
            for option in event.get("options", [])
            if option.get("reason") == "high_suspicion_no_open_evidence_question"
        ]
        output.append({
            "bug": row["instance_id"],
            "fusion": row["fusion"],
            "policy": row["policy"],
            "status": row.get("status"),
            "first_changed_file_rank": (
                file_metrics.get("first_relevant_rank") if file_metrics else None
            ),
            "file_hit_at_1": file_metrics.get("Top-1") if file_metrics else None,
            "file_hit_at_5": file_metrics.get("Top-5") if file_metrics else None,
            "file_mrr": file_metrics.get("MRR") if file_metrics else None,
            "gold_files_in_candidate_pool": (
                file_metrics.get("pool_gold_count") if file_metrics else None
            ),
            "gold_file_coverage": (
                file_metrics.get("candidate_pool_coverage") if file_metrics else None
            ),
            "method_mrr": method_metrics.get("MRR") if method_metrics else None,
            "gold_semantics": metrics.get("gold_semantics"),
            "graph_call_count": len(graph_calls),
            "graph_candidate_ids": [c["candidate_ids"][0] for c in graph_calls],
            "stop_gate_candidates": sorted(set(stops)),
            "logical_tokens": row.get("arm", {}).get("actual_tokens"),
            "provider_usage_is_shared": True,
            "graph_missing_callee_counts": sum(
                len((c.get("selection") or {}).get("diagnostics", {}).get("missing_callee_bodies", []))
                for c in graph_calls
            ),
            "graph_downgrade_guard_count": sum(
                h.get("guard_reason") == "unsupported_graph_downgrade"
                for entries in (row.get("arm", {}).get("stage_histories") or {}).values()
                for h in entries if h.get("stage") == "graph"
            ),
        })
    output.sort(key=lambda r: (r["bug"], r["fusion"], PAIRS.index(r["policy"])))
    keyrows = {(r["bug"], r["fusion"], r["policy"]): r for r in output}
    comparisons = []
    for bug, fusion in sorted({(r["bug"], r["fusion"]) for r in output}):
        items = [keyrows.get((bug, fusion, policy)) for policy in PAIRS]
        if any(v is None for v in items):
            comparisons.append({"bug": bug, "fusion": fusion, "status": "incomplete_arm_set"})
            continue
        base, focused, no_stop, with_stop = items
        if any(v["status"] != "completed" for v in items):
            comparisons.append({"bug": bug, "fusion": fusion, "status": "incomplete_execution"})
            continue
        comparisons.append({
            "bug": bug,
            "fusion": fusion,
            "status": "paired",
            "file_mrr_test_aware_minus_focused": (
                no_stop["file_mrr"] - focused["file_mrr"]
                if no_stop["file_mrr"] is not None and focused["file_mrr"] is not None
                else None
            ),
            "file_mrr_stop_minus_nostop": (
                with_stop["file_mrr"] - no_stop["file_mrr"]
                if with_stop["file_mrr"] is not None and no_stop["file_mrr"] is not None
                else None
            ),
            "graph_calls_saved_by_stop": (
                no_stop["graph_call_count"] - with_stop["graph_call_count"]
            ),
            "logical_tokens_saved_by_stop": (
                no_stop["logical_tokens"] - with_stop["logical_tokens"]
            ),
            "stop_gate_activated": bool(with_stop["stop_gate_candidates"]),
            "stopped_candidate_ids": with_stop["stop_gate_candidates"],
            "file_gold_in_pool": (base["gold_files_in_candidate_pool"] or 0) > 0,
        })
    return {
        "schema": "graft-phase3-multibug-audit-v1",
        "new_model_calls_by_this_audit": 0,
        "physical_provider_accounting": accounting,
        "gold_is_file_level_proxy_not_method_truth": True,
        "records": output,
        "paired_comparisons": comparisons,
        "all_pairs_completed": bool(comparisons) and all(
            r["status"] == "paired" for r in comparisons
        ),
        "any_real_stop_gate_activated": any(
            r.get("stop_gate_activated") for r in comparisons
        ),
        "note": (
            "Do not claim method-level localization accuracy: only official "
            "modified-file proxy labels are available. Per-policy logical "
            "tokens reuse shared provider responses and cannot be summed "
            "as independent physical token costs."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    data = inspect(args.run_dir)
    acct = data["physical_provider_accounting"]
    print("Physical new calls:", acct.get("new_calls"),
          "provider tokens:", acct.get("known_provider_tokens"))
    print("All paired arms complete:", data["all_pairs_completed"])
    for row in data["records"]:
        print(row["bug"], row["fusion"], row["policy"],
              "status", row["status"], "file_rank", row["first_changed_file_rank"],
              "file_MRR", row["file_mrr"], "file_Top5", row["file_hit_at_5"],
              "graph_calls", row["graph_call_count"], "logical_tokens", row["logical_tokens"])
    for comparison in data["paired_comparisons"]:
        print("PAIR", comparison)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print("Report saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

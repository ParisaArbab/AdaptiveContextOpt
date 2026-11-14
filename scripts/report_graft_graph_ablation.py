"""Read saved GRAFT ablation results without making model calls."""
import json
import sys
from collections import Counter
from pathlib import Path

POLICIES = ("fixed_l1", "l1_anchor", "l1_graph", "l1_graph_suspicion", "l1_graph_focused", "graft_adaptive")

def main(root):
    rows = json.loads((root / "results.json").read_text())
    accounting = json.loads((root / "accounting.json").read_text())
    print("PHYSICAL", "new_calls", accounting.get("new_calls"),
          "provider_tokens", accounting.get("known_provider_tokens"),
          "usage_unknown", accounting.get("usage_unknown"))
    rows = sorted((r for r in rows if r.get("policy") in POLICIES),
                  key=lambda r: (r.get("instance_id", ""), r.get("fusion", ""), POLICIES.index(r["policy"])))
    for row in rows:
        arm = row.get("arm", {})
        metrics = row.get("metrics", {}).get("java_strict_v2", {}).get("method", {})
        calls = arm.get("calls", [])
        stages = Counter(call.get("stage") for call in calls)
        paid = sum((c.get("input_tokens", 0) + c.get("output_tokens", 0)) for c in calls)
        print(row.get("instance_id"), "fusion", row.get("fusion"), row["policy"],
              "status",row.get("status"), "rank",metrics.get("first_relevant_rank"),
              "Top5",metrics.get("Top-5"),"MRR",metrics.get("MRR"),
              "logical_tokens",arm.get("actual_tokens"),
              "calls",dict(stages),"stop",arm.get("stop_reason"),
              "label_note",row.get("gold_unavailable_reason"))
        if row["policy"] == "l1_anchor" and any(s != "sketch" for s in stages):
            raise RuntimeError("L1-only ablation violated: graph/Full call detected")
        if row["policy"] in {"l1_graph", "l1_graph_suspicion", "l1_graph_focused"} and stages.get("method_full", 0):
            raise RuntimeError("Graph-only ablation violated: Full call detected")
    print("IMPORTANT: physical provider totals count shared calls once; per-policy logical totals can count shared cache requests repeatedly.")
    print("Do not claim graph-specific benefit unless rank changes between l1_anchor and l1_graph with the same evidence and LLM model.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/report_graft_graph_ablation.py RUN_DIR")
    main(Path(sys.argv[1]))

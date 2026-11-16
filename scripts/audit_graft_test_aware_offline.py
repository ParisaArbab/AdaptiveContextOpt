"""Zero-provider-call Phase-3 comparison using saved Time-4 graph and L1 responses.

Do not open evaluation_only.json and do not use gold labels in selection.
This is a prompt-packaging audit, not an accuracy experiment.

Example:
python -m scripts.audit_graft_test_aware_offline \
  results/graft_phase2_time4_live_20261113_202949/run \
  --output results/graft_test_aware_offline_20261113.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rave_fl.graft.focused import focus_select
from rave_fl.graft.graph import Graph
from rave_fl.graft.selection import ByteCounter
from rave_fl.graft.test_aware import test_aware_select, test_signal, stop_decision


def inspect_instance(folder: Path, budget: int = 4096) -> dict:
    graph = Graph.from_dict(json.loads((folder / "graph.json").read_text()))
    arms = list(folder.glob("l1_anchor__rrf_*.json"))
    if len(arms) != 1:
        raise ValueError("Exactly one saved RRF L1 control is required")
    arm = json.loads(arms[0].read_text())["arm"]
    states = arm["states"]
    observed = [
        (ident, state["observation"]["suspicion"])
        for ident, state in states.items()
        if state.get("observation") and state["observation"].get("valid")
    ]
    if not observed:
        raise ValueError("No valid L1 observations")
    # Both selectors use exactly the same online highest-suspicion target.
    target = max(sorted(observed), key=lambda row: row[1])[0]
    prior = states[target]["observation"]
    missing = prior.get("missing_evidence", "")
    before = focus_select(graph, target, budget, ByteCounter(), missing)
    after = test_aware_select(graph, target, budget, ByteCounter(), missing)
    ranked = arm["ranked_functions"]
    rank = ranked.index(target) + 1
    stop = stop_decision(
        type("Obs", (), prior)(), rank, top_k=arm["graft_config"]["top_k"]
    )
    def describe(selection):
        parsed = json.loads(selection.text)
        sources = parsed.get("sources", [])
        return {
            "selector": selection.selector,
            "status": selection.status,
            "reason": selection.reason,
            "input_units": selection.input_units,
            "counter": selection.counter,
            "exact_template_tokens": selection.exact_template_tokens,
            "target_source_lines": [
                [start, end] for section in sources if section["fn"] == target
                for start, end, _ in section["units"]
            ],
            "selected_methods": [section["fn"] for section in sources],
            "selected_fragments": [
                {"method": section["fn"], "path": section["path"],
                 "start": start, "end": end, "chars": len(src),
                 "preview": src.strip().splitlines()[0][:180]}
                for section in sources for start, end, src in section["units"]
            ],
            "diagnostics": selection.diagnostics,
        }
    return {
        "instance": folder.name,
        "selection_basis": "highest valid saved L1 suspicion, without gold",
        "selected_target": target,
        "initial_l1_suspicion": prior["suspicion"],
        "initial_l1_uncertainty": prior["uncertainty"],
        "initial_l1_missing_evidence": missing,
        "test_signal": test_signal(graph),
        "stop_decision": stop,
        "old_focused": describe(before),
        "new_test_aware": describe(after),
        "new_provider_calls": 0,
        "gold_labels_read": False,
        "developer_data_only": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--budget", type=int, default=4096)
    args = ap.parse_args(argv)
    files = sorted((args.run_dir / "instances").glob("*/graph.json"))
    if not files:
        raise SystemExit("No saved graphs")
    rows = [inspect_instance(path.parent, args.budget) for path in files]
    report = {
        "schema": "graft-phase3-test-aware-offline-v1",
        "new_provider_calls": 0,
        "uses_gold_during_routing": False,
        "development_only": True,
        "budget": args.budget,
        "results": rows,
    }
    for row in rows:
        print(f"\n{row['instance']}")
        print("Target from L1:", row["selected_target"])
        print("Test signal:", row["test_signal"])
        print("Conservative stop:", row["stop_decision"])
        for name in ("old_focused", "new_test_aware"):
            details = row[name]
            print(f"{name:16} {details['status']} {details['input_units']}/{args.budget} units")
            print("  Methods:", details["selected_methods"])
            print("  Target line groups:", details["target_source_lines"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Saved report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit new compact source retrieval on saved real graphs, with ZERO provider calls.

Usage:
 python -m scripts.audit_graft_focused_offline \
    results/graft_phase1_live_20261113_174425/run \
    --output results/graft_focused_offline_20261113.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rave_fl.graft.focused import focus_select
from rave_fl.graft.graph import Graph
from rave_fl.graft.selection import ByteCounter, package


def inspect_instance(folder: Path, budget: int = 4096) -> dict:
    graph = Graph.from_dict(json.loads((folder / "graph.json").read_text()))
    arms = list(folder.glob("l1_anchor__rrf_*.json"))
    if len(arms) != 1:
        raise ValueError(f"Expected exactly one saved RRF L1 arm in {folder}")
    arm = json.loads(arms[0].read_text())["arm"]
    states = arm["states"]
    # Use online model observations only. Never open evaluation_only.json here.
    eligible = sorted(
        (name, state["observation"]["suspicion"])
        for name, state in states.items()
        if state.get("observation") and state["observation"].get("valid")
    )
    if not eligible:
        raise ValueError("No valid L1 observations for focused retrieval")
    target = max(eligible, key=lambda row: row[1])[0]
    missing = states[target]["observation"].get("missing_evidence", "")
    focused = focus_select(graph, target, budget, ByteCounter(), missing)
    prior = next(folder.glob("l1_graph_suspicion__rrf_*.json"), None)
    old_call = None
    if prior is not None:
        previous = json.loads(prior.read_text())["arm"]["calls"]
        old_call = next(
            (c for c in previous if c.get("stage") == "graph" and target in c.get("candidate_ids", [])),
            None
        )
    compact = json.loads(focused.text)
    code = [
        {"method": section["fn"], "path": section["path"],
         "lines": [[start, end] for start, end, _ in section["units"]],
         "length": sum(len(text) for _, _, text in section["units"])}
        for section in compact.get("sources", [])
    ]
    round_trip = package(
        graph, target, focused.ids, budget, ByteCounter(), selector=focused.selector
    )
    if focused.text != round_trip.text or focused.input_units != round_trip.input_units:
        raise RuntimeError("Focused package is not reproducible")
    return {
        "instance": folder.name,
        "selected_target_from_l1_only": target,
        "missing_evidence_l1": missing,
        "counter": focused.counter,
        "budget": budget,
        "new_llm_calls": 0,
        "focused_status": focused.status,
        "focused_reason": focused.reason,
        "focused_input_units": focused.input_units,
        "focused_diagnostics": focused.diagnostics,
        "selected_sources": code,
        "old_graph_input_units_for_same_target": (
            old_call["selection"]["input_units"] if old_call else None
        ),
        "old_graph_selector_for_same_target": (
            old_call["selection"]["selector"] if old_call else None
        ),
        "note": (
            "Offline input-packaging check only. No new Qwen verdict or accuracy "
            "claim. This selector uses L1 text and exact buggy source; not gold labels."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=int, default=4096)
    args = parser.parse_args(argv)
    dirs = sorted((args.run_dir / "instances").glob("*/graph.json"))
    if not dirs:
        raise SystemExit("No saved graph.json found")
    results = [inspect_instance(p.parent, args.budget) for p in dirs]
    out = {
        "schema": "graft-focused-offline-audit-v1",
        "new_provider_calls": 0,
        "uses_evaluation_only_labels": False,
        "budget": args.budget,
        "results": results,
    }
    for row in results:
        print("Instance:", row["instance"])
        print("Target selected from saved L1:", row["selected_target_from_l1_only"])
        print("Focused:", row["focused_status"], row["focused_input_units"], "/", row["budget"])
        print("Source methods:", [r["method"] for r in row["selected_sources"]])
        print("Required missing:", row["focused_diagnostics"].get("required_missing"))
        print("Unresolved:", row["focused_diagnostics"].get("unresolved_callees"))
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

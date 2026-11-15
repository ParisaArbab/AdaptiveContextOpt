"""Read-only offline feasibility check for actual Phase-3 graph requests.

Uses a saved *online* L1-only ranking and the matching immutable graph, never
evaluation_only, test gold, or another model call. Refuses to start the new
physical experiment if both graph treatments remain impossible for any bug.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rave_fl.graft.graph import Graph
from rave_fl.graft.focused import focus_select
from rave_fl.graft.test_aware import test_aware_select
from rave_fl.graft.selection import ByteCounter


BUGS = ("Lang-1", "Math-1", "Time-1")
BUDGET = 4096


def inspect(source_run: Path, config: dict) -> dict:
    g = config["graft"]
    if g.get("failure_digest_mode") != "bounded_excerpt" or g["input_budget"] != BUDGET:
        raise ValueError("Only the declared 4096-byte bounded-excerpt graph condition is supported")
    rows = []
    for bug in BUGS:
        directories = list((source_run / "instances").glob(f"{bug}_*"))
        if len(directories) != 1:
            raise ValueError(f"Expected one previous genuine run for {bug}, found {len(directories)}")
        folder = directories[0]
        graph = Graph.from_dict(json.loads((folder / "graph.json").read_text()))
        anchors = list(folder.glob("l1_anchor__rrf_*.json"))
        if len(anchors) != 1:
            raise ValueError(f"Missing exact saved online L1 anchor for {bug}")
        record = json.loads(anchors[0].read_text())
        if (
            record["instance_id"] != bug
            or record["status"] != "completed"
            or record["synthetic"]
            or record["arm"]["stop_reason"] != "l1_anchor_only"
            or len(record["arm"]["ranked_functions"]) != 24
        ):
            raise ValueError(f"Old L1 observations not a completed 24-method real run: {bug}")
        states = record["arm"]["states"]
        for policy, function in (
            ("focused", focus_select),
            ("test_aware_nostop", test_aware_select),
        ):
            checks = []
            for rank, target in enumerate(record["arm"]["ranked_functions"], 1):
                if target not in graph.targets:
                    raise ValueError("Archived graph and L1 ranking disagree")
                obs = states[target]["observation"]
                if not obs or not obs["valid"]:
                    raise ValueError(f"Archived L1 candidate observation invalid: {bug} rank {rank}")
                selected = function(
                    graph,
                    target,
                    BUDGET,
                    ByteCounter(),
                    missing_evidence=obs.get("missing_evidence", ""),
                    failure_digest_mode="bounded_excerpt",
                )
                checks.append({
                    "rank": rank,
                    "target": target,
                    "status": selected.status,
                    "reason": selected.reason,
                    "input_units": selected.input_units,
                    "source_sha256": (
                        json.loads(selected.text).get("failure_excerpt", {}).get("source_sha256")
                    ),
                })
            ready = sum(c["status"] == "ready" for c in checks)
            top5 = sum(c["status"] == "ready" for c in checks[:5])
            rows.append({
                "bug": bug,
                "policy": policy,
                "ready_count_of_24": ready,
                "ready_in_previous_online_l1_top5": top5,
                "all_selections": checks,
            })
    return {
        "schema": "graft-phase3-graph-readiness-v1",
        "source_run": str(source_run.resolve()),
        "graph_budget_utf8_bytes_plus_template": BUDGET,
        "uses_saved_online_l1_not_gold": True,
        "new_model_calls": 0,
        "ready_for_real_graph_comparison": (
            len(rows) == 6
            and all(row["ready_count_of_24"] >= 1 and row["ready_in_previous_online_l1_top5"] >= 1 for row in rows)
        ),
        "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("Refusing to replace existing readiness results")
    data = inspect(a.source_run, json.loads(a.config.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2) + "\n")
    for row in data["rows"]:
        print(row["bug"], row["policy"], "ready", row["ready_count_of_24"],
              "of 24, top5 ready", row["ready_in_previous_online_l1_top5"])
    print("Real graph queries feasible under 4096-unit limit:", data["ready_for_real_graph_comparison"])
    print("New Qwen calls: 0")
    return 0 if data["ready_for_real_graph_comparison"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

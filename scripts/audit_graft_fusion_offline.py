"""Offline, zero-provider-call comparison of frozen GRAFT-FL fusion methods.

Usage:
  python -m scripts.audit_graft_fusion_offline \
    results/graft_time4_graph_ablation_20261113_143016/run \
    --output results/graft_time4_fusion_offline_20261113.json

Gold function IDs are read only AFTER rankings are computed, strictly for metrics.
Time-4 is development data, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rave_fl.eval.fusion import Fusion
from rave_fl.pilot_types import CandidateState, Observation, PilotCandidate

METHODS = ("rrf", "prior_aware", "consistent_score", "suspicion_rank_blend")


def inspect_instance(folder: Path, policy: str = "l1_anchor") -> dict:
    input_file = folder / "input.json"
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    arms = sorted(folder.glob(policy + "__*.json"))
    if len(arms) != 1:
        raise ValueError(f"Expected exactly one {policy} arm, found {len(arms)} in {folder}")
    data = json.loads(arms[0].read_text())
    if data.get("status") != "completed":
        return {"instance_id": data.get("instance_id", folder.name), "status": data.get("status"), "skipped": True}
    original = json.loads(input_file.read_text())
    candidates = [PilotCandidate(**row) for row in original["candidates"]]
    if len({c.function_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate IDs")
    saved = data["arm"]
    states = {
        key: CandidateState(
            level=value["level"],
            observation=Observation(**value["observation"]) if value["observation"] else None,
            history=value.get("history", []),
        )
        for key, value in saved["states"].items()
    }
    if set(states) != {c.function_id for c in candidates}:
        raise ValueError("Saved states differ from original candidate pool")
    ranker = saved["ranker"]
    config = {key: ranker[key] for key in ("tie", "rounding", "rrf_k")}
    rankings = {}
    for name in METHODS:
        fusion = Fusion(name, **config)
        scores = fusion.scores(candidates, states)
        rankings[name] = [c.function_id for c in fusion.order(candidates, scores)]

    # Reconstruct the historical arm before looking at gold IDs. In particular,
    # preserve the documented failure-name bonus exception.
    anchors = saved.get("failure_name_anchors") or {}
    matched = bool(any(anchors.values()) and saved.get("graft_config", {}).get("failure_name_anchor"))
    if not matched and rankings["rrf"] != saved["ranked_functions"]:
        raise ValueError("Frozen RRF reproduction failed; refuse misleading analysis")

    # Evaluation-only labels never enter Fusion, scores or rank construction.
    evaluation = folder / "evaluation_only.json"
    labels = json.loads(evaluation.read_text()).get("functions", []) if evaluation.exists() else []
    result = {
        "instance_id": data["instance_id"],
        "status": "completed",
        "candidate_count": len(candidates),
        "gold_used_for_scoring": False,
        "rrf_reproduced": not matched,
        "rankings": {},
    }
    for name, order in rankings.items():
        hits = [order.index(gold) + 1 for gold in labels if gold in order]
        rank = min(hits) if hits else None
        result["rankings"][name] = {
            "ranked_functions": order,
            "first_relevant_rank": rank,
            "Top-1": int(rank == 1) if rank is not None else None,
            "Top-5": int(rank <= 5) if rank is not None else None,
            "MRR": 1 / rank if rank is not None else None,
        }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--policy", default="l1_anchor")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    files = sorted((args.run_dir / "instances").glob("*/input.json"))
    if not files:
        raise SystemExit("No saved instance inputs found; no inference performed")
    results = [inspect_instance(f.parent, args.policy) for f in files]
    summary = {
        "analysis": "offline_frozen_llm_fusion",
        "policy": args.policy,
        "run_dir": str(args.run_dir),
        "new_model_calls": 0,
        "development_not_heldout": True,
        "gold_used_for_scoring": False,
        "fixed_blend_weights": {"rank": 0.30, "suspicion": 0.70},
        "results": results,
    }
    for row in results:
        if row.get("skipped"):
            print(row["instance_id"], "skipped:", row["status"])
            continue
        print(f"\n{row['instance_id']}: {row['candidate_count']} candidates, RRF reproduced={row['rrf_reproduced']}")
        for method in METHODS:
            metrics = row["rankings"][method]
            print(f"  {method:24s} rank={str(metrics['first_relevant_rank']):>4s} "
                  f"Top5={metrics['Top-5']} MRR={metrics['MRR']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite saved audit: {args.output}")
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print("Report saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

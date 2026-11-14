"""Predeclared, gold-aware OFFLINE difficulty screen, never inference.

All twelve IDs come from configs/rave_v3_independent_protocol.json, recorded before
this screen. Gold is read only for difficulty stratification, NOT model prompts.
Every defect stays in the output, including missing and easy ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from rave_fl.expanded_pool import union_pool
from wp1.flexfl_data import read_ground_truth
from wp1.evaluation import first_relevant_rank

METHODS = ("SBIR", "Ochiai", "BoostN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-flexfl", action="store_true")
    args = parser.parse_args()
    plan_path = Path("configs/rave_v3_independent_protocol.json")
    plan = json.loads(plan_path.read_text())
    bugs = plan["exploratory_bugs"]
    assert len(bugs) == 12 and len(set(bugs)) == 12
    assert set(plan["smoke_bugs"]) <= set(bugs)
    ref = Path("references/FlexFL_Time25").resolve()
    if not (ref / ".git").is_dir():
        raise RuntimeError("Missing real FlexFL Git checkout " + str(ref))

    relative = [
        f"FlexFL/data/FL_results/{method}/Defects4J/{bug}_method-susps.csv"
        for bug in bugs for method in METHODS
    ]
    relative += ["FlexFL/data/input/ground_truth/Defects4J/gt.json"]
    if args.fetch_flexfl:
        # Existing sparse checkout remains intact. No bugs are substituted.
        subprocess.run(
            ["git", "-C", str(ref), "sparse-checkout", "add", *relative],
            check=True,
        )
    rows = []
    for bug in bugs:
        try:
            candidates, _ = union_pool(ref, bug, 10)
            ranked = [c.function_id for c in candidates][:24]
            truth = read_ground_truth(ref, bug)
            if not truth:
                status = "unlabeled"
                rank = None
            else:
                rank = first_relevant_rank(ranked, truth)
                status = "hard_in_pool" if rank is not None and rank > 5 else (
                    "easy_in_pool" if rank is not None else "gold_absent_from_top24"
                )
            rows.append({
                "bug": bug, "status": status,
                "candidate_count": len(ranked), "gold_rank_in_pool": rank,
                "candidate_recall": rank is not None,
                "gold_label_available": bool(truth),
                "error": None,
            })
        except (FileNotFoundError, ValueError, KeyError) as exc:
            rows.append({
                "bug": bug, "status": "unavailable", "candidate_count": None,
                "gold_rank_in_pool": None, "candidate_recall": None,
                "gold_label_available": None, "error": str(exc),
            })
    result = {
        "schema": "graft-prior-difficulty-screen-v1",
        "selection_origin": str(plan_path),
        "selection_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "candidate_methods": list(METHODS),
        "top_per_method": 10, "candidate_count_limit": 24,
        "all_predeclared_bugs": bugs,
        "difficulty_rule": "gold_rank 6..24 in original candidate union",
        "exploratory_gold_used_for_stratification": True,
        "not_held_out": True,
        "no_inference": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(f'{r["bug"]:8} {r["status"]:23} rank={r["gold_rank_in_pool"]} pool={r["candidate_count"]}', flush=True)
    print("Hard and present:", [r["bug"] for r in rows if r["status"] == "hard_in_pool"])
    print("Unreachable by retrieval:", [r["bug"] for r in rows if r["status"] == "gold_absent_from_top24"])
    print("Output:", args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preparation failed without model inference: {exc}", file=sys.stderr)
        sys.exit(2)

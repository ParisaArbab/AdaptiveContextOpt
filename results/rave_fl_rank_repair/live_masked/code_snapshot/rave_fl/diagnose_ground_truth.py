"""Inspect ground-truth mismatch without changing any experimental labels."""
import argparse
import json
from pathlib import Path
from rave_fl.flexfl_adapter import load_candidates
from wp1.flexfl_data import read_ground_truth
from wp1.evaluation import method_matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flexfl-repo", type=Path, required=True)
    p.add_argument("--bug", required=True)
    p.add_argument("--method", default="Ochiai", choices=["Ochiai","SBIR","BoostN"])
    args = p.parse_args()
    truths = read_ground_truth(args.flexfl_repo, args.bug)
    candidates = load_candidates(args.flexfl_repo, args.bug, args.method)
    print(json.dumps({
        "bug": args.bug, "ground_truth": truths,
        "candidate_matches": [
            {"candidate": c.function_id,
             "matches_ground_truth": [t for t in truths if method_matches(c.function_id, t)]}
            for c in candidates
        ]
    }, indent=2))


if __name__ == "__main__":
    main()

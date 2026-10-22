"""Run a reproducible real-input cost-profile and ranking evaluation.

Uses FlexFL's existing Defects4J CSVs and evaluation metrics. IMPORTANT:
without a real verifier, RAVE-FL cannot produce evidence-based reranking.
We report only unchanged ranking quality and planned token costs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from rave_fl.core import RaveFL, approx_tokens, sketch
from rave_fl.flexfl_adapter import load_candidates
from wp1.evaluation import evaluate_top5
from wp1.flexfl_data import read_ground_truth


def profile(repo: Path, bug: str, method: str, budget: int) -> dict:
    candidates = load_candidates(repo, bug, method)
    baseline = [c.function_id for c in candidates]
    # No external verifier: do not claim that any model checked any function.
    baseline_cost = sum(sketch(c, 2).approx_tokens for c in candidates)
    engine = RaveFL(candidates, token_budget=budget)
    plan = engine.execute()
    # Default verifier returns zero-token noninformative responses.
    # The selected action plan counts only sketch input estimation.
    result = {
        "bug": bug,
        "method": method,
        "candidate_count": len(candidates),
        "candidate_source": "FlexFL traditional FL results",
        "real_verifier": False,
        "risk_model": "heuristic, untrained",
        "baseline_full_sketch_estimated_tokens": baseline_cost,
        "rave_sketch_estimated_tokens": plan["tokens_used"],
        "rave_actions": plan["actions"],
        "ranked_functions": baseline,
        "ranking_changed_by_verified_evidence": False,
        "notes": "Sketch-only cost profile, not a fair end-to-end LLM token comparison",
    }
    truths = read_ground_truth(repo, bug)
    result["baseline_accuracy"] = evaluate_top5(baseline, truths)
    result["rave_accuracy"] = evaluate_top5(baseline, truths)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexfl-repo", type=Path, required=True)
    parser.add_argument("--bug", required=True, help="e.g., Time-25")
    parser.add_argument("--method", choices=["SBIR", "Ochiai", "BoostN"], default="Ochiai")
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = profile(args.flexfl_repo, args.bug, args.method, args.token_budget)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()

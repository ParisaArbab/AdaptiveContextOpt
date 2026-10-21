"""Paired comparison: full verification versus risk-prioritized verification.

Usage is reported from Ollama responses, not estimated sketches.
Output token cap and preflight estimates reduce, but do not guarantee,
compliance with a strict provider-side total-token budget.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rave_fl.flexfl_adapter import load_candidates
from rave_fl.ollama_verify import admission_estimate, verify
from wp1.flexfl_data import read_bug_report, read_trigger_test, read_ground_truth
from wp1.evaluation import evaluate_top5


def run_arm(candidates, sources, context, model, url, budget, selective, max_output):
    consumed = 0
    records = []
    order = sorted(candidates, key=lambda c: c.semantic_score, reverse=True)
    for c in order:
        source = sources.get(c.function_id)
        if not source:
            raise ValueError(f"No real source supplied for {c.function_id}")
        # Policy controls depth of evidence; the same model and context are used.
        # For now 'selective' only verifies high-ranked methods. Other methods
        # retain their base ranking, never acquire fake zero-verdict evidence.
        if selective and c.semantic_score < .60:
            continue
        estimate = admission_estimate(context, c.function_id, source, max_output)
        if consumed + estimate > budget:
            continue
        r = verify(url, model, context, c.function_id, source, max_output)
        consumed += r.input_tokens + r.output_tokens
        records.append(asdict(r))
        if consumed > budget:
            break
    scores = {r["method"]: r["suspicion"] for r in records}
    # Only re-rank verified candidates; keep unverified relative ordering.
    checked = sorted([c for c in order if c.function_id in scores],
                     key=lambda c: scores[c.function_id], reverse=True)
    unchecked = [c for c in order if c.function_id not in scores]
    ranking = [c.function_id for c in checked + unchecked]
    return {
        "ranked_functions": ranking,
        "calls": records, "actual_tokens": consumed,
        "budget": budget, "budget_exceeded": consumed > budget,
        "verification_count": len(records),
    }


def run(args):
    candidates = load_candidates(args.flexfl_repo, args.bug, args.method)
    sources = json.loads(args.sources.read_text())
    if not isinstance(sources, dict):
        raise ValueError("--sources must be a JSON object from candidate ID to actual Java code")
    missing = [c.function_id for c in candidates if not sources.get(c.function_id)]
    if missing:
        raise ValueError(f"Missing source for {len(missing)} candidates: {missing}")
    context = (read_bug_report(args.flexfl_repo, args.bug) + "\n" +
               read_trigger_test(args.flexfl_repo, args.bug))
    if not context.strip():
        raise ValueError("No real bug report or trigger test available")
    full = run_arm(candidates, sources, context, args.model, args.base_url,
                   args.token_budget, False, args.max_output_tokens)
    selective = run_arm(candidates, sources, context, args.model, args.base_url,
                        args.token_budget, True, args.max_output_tokens)
    truth = read_ground_truth(args.flexfl_repo, args.bug)  # offline evaluation only
    for arm in [full, selective]:
        arm["accuracy"] = evaluate_top5(arm["ranked_functions"], truth)
    return {"bug": args.bug, "model": args.model, "method": args.method,
            "full": full, "selective": selective,
            "limitations": [
                "Candidate pool is limited to FlexFL's five traditional-method results.",
                "Selective policy uses an ordinal rank threshold, not learned risk.",
                "Token counts are real Ollama counters; admission estimates are approximate.",
                "Arms may exceed budget by one call because provider totals are known afterward.",
                "This comparison does not establish statistical significance.",
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flexfl-repo", type=Path, required=True)
    p.add_argument("--bug", required=True)
    p.add_argument("--method", choices=["SBIR","Ochiai","BoostN"], default="Ochiai")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--token-budget", type=int, default=12000)
    p.add_argument("--max-output-tokens", type=int, default=96)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

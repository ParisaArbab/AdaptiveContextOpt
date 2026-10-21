"""Run Phase-3 candidate-pool experiments with automatically extracted Java sources.

First use --dry-run to validate source spans without an LLM. A real run
uses two paired Ollama arms and reports measured model token usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rave_fl.compare_llm import run_arm
from rave_fl.expanded_pool import union_pool
from rave_fl.ollama_verify import admission_estimate
from rave_fl.source_spans import extract_pool_sources
from wp1.evaluation import evaluate_top5
from wp1.flexfl_data import read_bug_report, read_trigger_test, read_ground_truth


def load_context(flexfl_repo: Path, bug: str, evidence_file: Path | None = None) -> tuple[str, dict]:
    """Load observed failure evidence only; never read gold labels into the prompt.

    A supplied evidence file is authoritative. Missing or empty files fail rather
    than silently switching to the original short FlexFL input.
    """
    if evidence_file is not None:
        context = evidence_file.read_text(encoding="utf-8")
        source = "structured_failure_file"
    else:
        context = read_bug_report(flexfl_repo, bug) + "\n" + read_trigger_test(flexfl_repo, bug)
        source = "flexfl_original"
    if not context.strip():
        raise ValueError(f"Empty debugging evidence from {evidence_file or source}")
    meta = {
        "source": source,
        "file": str(evidence_file) if evidence_file is not None else None,
        "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "characters": len(context),
    }
    return context, meta


def experiment(args) -> dict:
    candidates, metadata = union_pool(args.flexfl_repo, args.bug, args.top_per_method)
    sources, paths = extract_pool_sources(args.checkout, metadata)
    context, evidence = load_context(args.flexfl_repo, args.bug, args.evidence_file)
    result = {
        "bug": args.bug,
        "candidate_sources": ["SBIR", "Ochiai", "BoostN"],
        "top_per_method": args.top_per_method,
        "unique_candidates": len(candidates),
        "source_files": paths,
        "source_checkout": str(args.checkout),
        "model": args.model,
        "dry_run": args.dry_run,
        "evidence": evidence,
    }
    if args.dry_run:
        result["candidate_ids"] = [c.function_id for c in candidates]
        result["approx_full_admission_total"] = sum(
            admission_estimate(context, c.function_id, sources[c.function_id],
                               args.max_output_tokens) for c in candidates
        )
        result["note"] = "Source validation and planning only: zero LLM calls."
        return result
    if not args.model:
        raise ValueError("--model is required unless --dry-run is provided")
    if args.token_budget <= 0 or args.max_output_tokens <= 0:
        raise ValueError("Token budget and max output tokens must be positive")
    full = run_arm(candidates, sources, context, args.model, args.base_url,
                   args.token_budget, False, args.max_output_tokens)
    selective = run_arm(candidates, sources, context, args.model, args.base_url,
                        args.token_budget, True, args.max_output_tokens)
    gold = read_ground_truth(args.flexfl_repo, args.bug)  # Offline only, after both LLM runs
    full["accuracy"] = evaluate_top5(full["ranked_functions"], gold)
    selective["accuracy"] = evaluate_top5(selective["ranked_functions"], gold)
    result.update({
        "full": full, "selective": selective,
        "actual_token_difference": full["actual_tokens"] - selective["actual_tokens"],
        "ground_truth_evaluation_only": True,
        "limitations": [
            "The policy is a simple rank-proxy threshold, not a learned risk model.",
            "The source-line spans come from FlexFL CSV files and are checked for signature.",
            "Total token budget uses approximate preflight admission, not a hard cap.",
            "Time-25 is a single bug; do not claim statistical accuracy improvement.",
            "Changing failure evidence may affect both model scores and token usage; do not equate this with a new policy.",
        ],
    })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flexfl-repo", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True, help="Actual buggy Defects4J checkout")
    p.add_argument("--bug", required=True)
    p.add_argument("--top-per-method", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--evidence-file", type=Path, help="Observed failure evidence; replaces the short FlexFL prompt context")
    p.add_argument("--model")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--token-budget", type=int, default=16000)
    p.add_argument("--max-output-tokens", type=int, default=96)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    data = experiment(args)
    text = json.dumps(data, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

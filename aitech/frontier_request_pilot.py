"""Frozen missing-method retrieval manifest for three archived real bugs.

This intentionally stops *before* code retrieval: method references are
unverified class-name+arity hints. Never invent definitions. This manifest
enables a second-stage source lookup against a pinned buggy checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .frontier_request_planner import plan_missing_definitions
from .risk_gated_pilot import SOURCES


def experiment(sources: dict[str, Path] = SOURCES, limit: int = 5) -> dict:
    cases = []
    for bug, path in sources.items():
        raw = path.read_bytes()
        snapshot = json.loads(raw)
        requests = plan_missing_definitions(snapshot, max_requests=limit)
        cases.append({
            "bug": bug,
            "candidate_source_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_source_path": str(path),
            "ranked_candidates": len(snapshot["candidates"]),
            "requests": [r.__dict__ for r in requests],
            "anchored_request_count": sum(r.failure_anchor_caller for r in requests),
        })
    return {
        "scope": "Exploratory missing-code acquisition planning on three previously studied buggy-source snapshots",
        "method": "failure-anchor priority + callsite rank; no gold labels",
        "request_limit_per_bug": limit,
        "actual_source_retrieved": False,
        "verified_method_bindings": False,
        "llm_executed": False,
        "cases": cases,
        "next_step": "Lookup each requested owner/method in a pinned buggy checkout, verify method declaration/type binding and freeze selected source snippets before testing real localization",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/frontier_requests.json"))
    args = parser.parse_args()
    result = experiment(limit=args.max_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['bug']}: requested={len(case['requests'])} "
              f"failure-anchor-related={case['anchored_request_count']}")
        for req in case["requests"]:
            print(f"  {req['owner']}.{req['method']}/{req['arity']} "
                  f"score={req['priority']} anchored={req['failure_anchor_caller']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

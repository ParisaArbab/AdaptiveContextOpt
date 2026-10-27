#!/usr/bin/env python3
"""Read-only audit of stored RAVE-FL batch response cardinality and output caps.

This inspects saved results.json files only, never sends requests to Ollama.
Malformed responses are *not* silently repaired or credited as valid.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect(root: Path, output_per_candidate: int) -> None:
    bad_total = 0
    total = 0
    for path in sorted(root.glob("*/results.json")):
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path.parent.name}: unreadable results: {exc}")
            continue
        for row in rows:
            arm = row.get("arm") or {}
            calls = arm.get("calls", [])
            malformed = []
            for index, call in enumerate(calls, 1):
                if call.get("action") != "BATCH_SCREEN":
                    continue
                total += 1
                err = call.get("error")
                if not err:
                    continue
                bad_total += 1
                candidates = len(call.get("candidate_ids", []))
                response = call.get("raw_response", "")
                count = None
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                        count = len(parsed["results"])
                except (ValueError, TypeError):
                    pass
                outputs = call.get("output_tokens", 0)
                cap = output_per_candidate * candidates
                malformed.append({
                    "call": index, "expected": candidates, "returned": count,
                    "output_tokens": outputs, "output_cap": cap,
                    "near_cap": cap > 0 and outputs >= cap - 5,
                    "error": err,
                })
            invalid = [
                k for k, s in (arm.get("states") or {}).items()
                if s.get("level", -1) >= 0 and not (s.get("observation") or {}).get("valid", False)
            ]
            print(f"{path.parent.name:18} {row.get('policy', '?'):14} "
                  f"status={row.get('status')} calls={len(calls)} "
                  f"bad_batches={len(malformed)} invalid_candidates={len(invalid)}")
            for entry in malformed:
                print("  call={call} expected={expected} returned={returned} "
                      "output={output_tokens}/{output_cap} near_cap={near_cap} "
                      "error={error}".format(**entry))
    print(f"\nTotal screening batches: {total}; malformed batches: {bad_total}")
    if total:
        print(f"Malformed batch rate: {bad_total / total:.1%}")
    print("NOTE: Near-cap suggests possible truncation; otherwise inspect prompting/model compliance.")
    print("NOTE: Each malformed call consumed real tokens, counted in the original run.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-tokens-per-candidate", type=int, default=160)
    args = p.parse_args()
    if not args.run_root.is_dir():
        p.error(f"Run directory missing: {args.run_root}")
    if args.output_tokens_per_candidate < 1:
        p.error("Output token cap must be positive")
    inspect(args.run_root, args.output_tokens_per_candidate)


if __name__ == "__main__":
    main()

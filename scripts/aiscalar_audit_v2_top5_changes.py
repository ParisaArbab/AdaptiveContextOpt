#!/usr/bin/env python3
"""Read-only fault localization rank-repair audit on a completed benchmark.

Diagnose V2 Top-5 gains and losses using OFFLINE gold labels only.
Never forwards gold labels to model, policy, prompts or candidate selection.
Creates a new derived CSV and prints the top rank-regression cases.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

# Permit both `python -m scripts.aiscalar_audit_v2_top5_changes` and
# `python scripts/aiscalar_audit_v2_top5_changes.py` from any cwd.
# The latter normally puts scripts/, not the repository root, on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp1.evaluation import method_matches

POLICIES = ("full", "fixed_l1", "v2_utility")


def read_case(path):
    file = path / "results.json"
    if not file.exists():
        return None
    rows = json.loads(file.read_text())
    return {r["policy"]: r for r in rows if r.get("policy") in POLICIES}


def gold_functions(path):
    sources = list((path / "instances").glob("*/evaluation_only.json"))
    if len(sources) != 1:
        return []
    labels = json.loads(sources[0].read_text())
    return labels.get("functions", []) or []


def first_rank(candidates, gold):
    if not candidates or not gold:
        return None
    for rank, ident in enumerate(candidates, 1):
        if any(method_matches(ident, target) for target in gold):
            return rank
    return None


def matching_ids(candidates, gold):
    return [
        ident for ident in candidates
        if any(method_matches(ident, target) for target in gold)
    ]


def audit_case(bug, rows, gold):
    full, v2 = rows["full"], rows["v2_utility"]
    f = (full.get("metrics") or {}).get("method") or {}
    v = (v2.get("metrics") or {}).get("method") or {}
    if not all(isinstance(m.get("Top-5"), (int, float)) for m in (f, v)):
        return None
    if any(row.get("status") != "completed" for row in (full, v2)):
        return None

    if f["Top-5"] == 1 and v["Top-5"] == 0:
        outcome = "loss"
    elif f["Top-5"] == 0 and v["Top-5"] == 1:
        outcome = "gain"
    else:
        return None

    arm = v2.get("arm") or {}
    states = arm.get("states") or {}
    initial = arm.get("initial_l1") or {}
    initial_order = initial.get("ranked_functions") or []
    final_order = arm.get("ranked_functions") or []
    good_ids = matching_ids(final_order, gold)
    inspected_good = [i for i in good_ids if (states.get(i) or {}).get("level") == 2]
    # Diagnostics give the last known utility estimate for each candidate.
    proxies = {}
    for frame in arm.get("repair_diagnostics") or []:
        for option in frame.get("options") or []:
            fid = option.get("function_id")
            if fid in good_ids:
                proxies[fid] = option.get("utility_per_token")

    first_gold = min(good_ids, key=lambda i: final_order.index(i)) if good_ids else None
    obs = ((states.get(first_gold) or {}).get("observation") or {}) if first_gold else {}
    decision_trace = arm.get("trace") or []
    full_verify_decisions = [
        t for t in decision_trace if t.get("action") == "FULL_VERIFY"
    ]

    f_arm = full.get("arm") or {}
    return {
        "bug": bug,
        "outcome": outcome,
        "full_top5": f.get("Top-5"),
        "v2_top5": v.get("Top-5"),
        "full_fault_rank_reported": f.get("first_relevant_rank"),
        "v2_fault_rank_reported": v.get("first_relevant_rank"),
        "full_fault_rank_from_ids": first_rank(
            f_arm.get("ranked_functions"), gold),
        "v2_fault_rank_from_ids": first_rank(final_order, gold),
        "v2_initial_l1_fault_rank": first_rank(initial_order, gold),
        "full_tokens": full.get("accounted_tokens"),
        "v2_tokens": v2.get("accounted_tokens"),
        "v2_token_delta_vs_full": (
            v2["accounted_tokens"] - full["accounted_tokens"]
            if isinstance(v2.get("accounted_tokens"), (int, float))
            and isinstance(full.get("accounted_tokens"), (int, float))
            else None
        ),
        "full_full_inspections": full.get("full_calls"),
        "v2_full_inspections": v2.get("full_calls"),
        "v2_full_gold_inspections": len(inspected_good),
        "v2_gold_count_in_pool": len(good_ids),
        "v2_last_gold_utility_proxy": max(
            (x for x in proxies.values() if isinstance(x, (int, float))),
            default=None,
        ),
        "v2_stop_reason": arm.get("stop_reason"),
        "v2_first_gold_l1_or_full_suspicion": obs.get("suspicion"),
        "v2_first_gold_l1_or_full_uncertainty": obs.get("uncertainty"),
        "v2_first_gold_level": (states.get(first_gold) or {}).get("level") if first_gold else None,
        "v2_inspected_functions": ";".join(
            ident for t in full_verify_decisions for ident in t.get("ids", [])
        ),
        "v2_gold_candidates": ";".join(good_ids),
    }


def analyze(root):
    results = []
    skipped_labels = []
    for bug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = read_case(bug_dir)
        if rows is None or "full" not in rows or "v2_utility" not in rows:
            continue
        gold = gold_functions(bug_dir)
        if not gold:
            skipped_labels.append(bug_dir.name)
            continue
        r = audit_case(bug_dir.name, rows, gold)
        if r is not None:
            results.append(r)
    return results, skipped_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Original 8K per-bug result root, not 16K recovery.")
    parser.add_argument("--out", required=True, type=Path,
                        help="New CSV output. Refuses overwrite.")
    args = parser.parse_args()
    root, dest = args.root.resolve(), args.out.resolve()
    if not root.is_dir():
        parser.error(f"Missing source directory: {root}")
    if dest.exists():
        parser.error("Output already exists; do not overwrite a previous diagnosis.")
    if root in dest.parents:
        parser.error("Output must be outside the original per-bug run.")
    results, skipped = analyze(root)
    if not results:
        parser.error("No paired method Top-5 gains/losses with offline gold found.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    losses = [r for r in results if r["outcome"] == "loss"]
    gains = [r for r in results if r["outcome"] == "gain"]
    print("Top-5 losses:", len(losses), "; Top-5 gains:", len(gains),
          "; skipped without offline method gold:", len(skipped))
    print("Losses with V2 higher token cost:", sum(
        r["v2_token_delta_vs_full"] is not None and r["v2_token_delta_vs_full"] > 0
        for r in losses
    ))
    print("Losses with no V2 full inspection of a gold method:",
          sum(r["v2_full_gold_inspections"] == 0 for r in losses))
    print("Losses where initial L1 placed gold in Top-5:",
          sum(r["v2_initial_l1_fault_rank"] is not None
              and r["v2_initial_l1_fault_rank"] <= 5 for r in losses))
    print("\nCases (initial L1 rank -> final V2 rank; full rank; gold full inspections):")
    for r in losses + gains:
        print(
            f"{r['outcome']:4} {r['bug']:23} "
            f"L1={r['v2_initial_l1_fault_rank']} "
            f"V2={r['v2_fault_rank_from_ids']} "
            f"Full={r['full_fault_rank_from_ids']} "
            f"gold_full={r['v2_full_gold_inspections']} "
            f"v2_extra_tokens={r['v2_token_delta_vs_full']} "
            f"stop={r['v2_stop_reason']}"
        )
    print("\nSaved:", dest)
    print("READ-ONLY: no LLM/GPU calls or original-run modification.")
    print("Reported rank may differ from ID-based rank if a predicted method "
          "maps to multiple gold signatures.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only ablation: initial L1 state versus V2 final ranking (same paid run).

The initial L1 ranking is saved *before* V2's full-code inspections. This
provides a useful within-run observational counterfactual, not a separately
re-executed L1-only policy, nor a causal estimate of added full inspections.

Gold labels are read only AFTER saved inference, exclusively for evaluation.
The original run is never modified. Do not mix 8K and 16K experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aiscalar_audit_v2_top5_changes import read_case, gold_functions, first_rank


def top5(rank):
    return int(rank is not None and rank <= 5)


def top1(rank):
    return int(rank == 1)


def mrr(rank):
    return (1.0 / rank) if rank is not None else 0.0


def collect(root):
    results = []
    exclusions = Counter()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = read_case(d)
        if not rows or "v2_utility" not in rows:
            exclusions["missing_v2"] += 1
            continue
        v2 = rows["v2_utility"]
        if v2.get("status") != "completed":
            exclusions["v2_not_completed"] += 1
            continue
        gold = gold_functions(d)
        if not gold:
            exclusions["missing_gold"] += 1
            continue
        arm = v2.get("arm") or {}
        initial = arm.get("initial_l1") or {}
        initial_order = initial.get("ranked_functions")
        final_order = arm.get("ranked_functions")
        if not isinstance(initial_order, list) or not initial_order:
            exclusions["missing_initial_l1"] += 1
            continue
        if not isinstance(final_order, list) or not final_order:
            exclusions["missing_v2_final"] += 1
            continue
        if set(initial_order) != set(final_order) or len(initial_order) != len(final_order):
            exclusions["candidate_pool_changed"] += 1
            continue

        l1_rank = first_rank(initial_order, gold)
        v2_rank = first_rank(final_order, gold)
        l1_cost = initial.get("tokens_used")
        v2_cost = v2.get("accounted_tokens")
        if not isinstance(l1_cost, (float, int)) or not isinstance(v2_cost, (float, int)):
            exclusions["missing_token_count"] += 1
            continue
        if l1_cost < 0 or v2_cost < l1_cost:
            exclusions["invalid_token_accounting"] += 1
            continue

        full = rows.get("full", {})
        fixed = rows.get("fixed_l1", {})
        m = (v2.get("metrics") or {}).get("method") or {}
        rr = m.get("first_relevant_rank")
        rank_consistent = (rr == v2_rank or rr is None and v2_rank is None)
        data = {
            "bug": d.name,
            "project": d.name.rpartition("-")[0],
            "l1_rank": l1_rank,
            "v2_rank": v2_rank,
            "l1_top1": top1(l1_rank),
            "v2_top1": top1(v2_rank),
            "l1_top5": top5(l1_rank),
            "v2_top5": top5(v2_rank),
            "l1_mrr": mrr(l1_rank),
            "v2_mrr": mrr(v2_rank),
            "l1_accounted_tokens": l1_cost,
            "v2_accounted_tokens": v2_cost,
            "v2_post_l1_tokens": v2_cost - l1_cost,
            "v2_full_calls": v2.get("full_calls"),
            "v2_stop_reason": arm.get("stop_reason"),
            "v2_rank_consistent_with_offline_metrics": rank_consistent,
            "full_status": full.get("status", "missing"),
            "full_top5_reported": ((full.get("metrics") or {}).get("method") or {}).get("Top-5"),
            "fixed_status": fixed.get("status", "missing"),
            "fixed_top5_reported": ((fixed.get("metrics") or {}).get("method") or {}).get("Top-5"),
        }
        data["l1_to_v2"] = ("gain" if data["v2_top5"] > data["l1_top5"]
                            else "loss" if data["v2_top5"] < data["l1_top5"]
                            else "unchanged")
        results.append(data)
    return results, exclusions


def summarize(rows, exclusions):
    if not rows:
        raise ValueError("No usable completed V2 runs with initial L1 ranking")
    result = {
        "included": len(rows),
        "exclusions": dict(exclusions),
        "top5_transitions": dict(Counter(r["l1_to_v2"] for r in rows)),
        "v2_rank_mismatches": sum(not r["v2_rank_consistent_with_offline_metrics"] for r in rows),
        "l1": {
            "top1": statistics.mean(r["l1_top1"] for r in rows),
            "top5": statistics.mean(r["l1_top5"] for r in rows),
            "mrr": statistics.mean(r["l1_mrr"] for r in rows),
            "mean_accounted_tokens_at_l1": statistics.mean(r["l1_accounted_tokens"] for r in rows),
            "total_accounted_tokens_at_l1": sum(r["l1_accounted_tokens"] for r in rows),
        },
        "v2": {
            "top1": statistics.mean(r["v2_top1"] for r in rows),
            "top5": statistics.mean(r["v2_top5"] for r in rows),
            "mrr": statistics.mean(r["v2_mrr"] for r in rows),
            "mean_accounted_tokens": statistics.mean(r["v2_accounted_tokens"] for r in rows),
            "total_accounted_tokens": sum(r["v2_accounted_tokens"] for r in rows),
            "total_incremental_tokens_after_l1": sum(r["v2_post_l1_tokens"] for r in rows),
        },
        "stop_reasons": dict(Counter(str(r["v2_stop_reason"]) for r in rows)),
        "per_project": {},
    }
    groups = defaultdict(list)
    for r in rows:
        groups[r["project"]].append(r)
    for project, group in sorted(groups.items()):
        result["per_project"][project] = {
            "n": len(group),
            "initial_l1_top5": statistics.mean(r["l1_top5"] for r in group),
            "v2_top5": statistics.mean(r["v2_top5"] for r in group),
            "gain": sum(r["l1_to_v2"] == "gain" for r in group),
            "loss": sum(r["l1_to_v2"] == "loss" for r in group),
        }
    return result


def save(rows, summary, out, with_plots):
    out.mkdir(parents=True)
    with (out / "per_bug_l1_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Initial L1 to V2 rank-repair ablation (same 8K run)",
        "",
        f"Joint usable completed V2 bugs with gold labels: **{summary['included']}**",
        "",
        "| Metric | Initial L1 | Final V2 |",
        "|---|---:|---:|",
    ]
    for name in ("top1", "top5", "mrr"):
        lines.append(f"| {name.upper()} | {summary['l1'][name]:.3f} | {summary['v2'][name]:.3f} |")
    lines += [
        f"| Total accounted tokens | {summary['l1']['total_accounted_tokens_at_l1']:,} | {summary['v2']['total_accounted_tokens']:,} |",
        "",
        "## Within-V2 Top-5 changes after full-code inspections",
        "",
        f"- Gains: {summary['top5_transitions'].get('gain', 0)}",
        f"- Losses: {summary['top5_transitions'].get('loss', 0)}",
        f"- Unchanged: {summary['top5_transitions'].get('unchanged', 0)}",
        f"- Added accounted tokens after L1: {summary['v2']['total_incremental_tokens_after_l1']:,}",
        f"- Ranking mismatches vs evaluator: {summary['v2_rank_mismatches']}",
        "",
        "## Controller stopping reasons",
    ]
    for k, v in sorted(summary["stop_reasons"].items()):
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Important limitations",
        "",
        "- No extra model inference. Initial L1 rank is an archived snapshot from each V2 arm, "
        "not an independently executed new policy.",
        "- Token counts are policy-accounted, and may include shared/cached observations; "
        "do not claim this stage represents fresh physical inference cost.",
        "- The cohort only includes V2-completed bugs with offline method labels and "
        "matching initial/final pools; missing and incomplete bugs remain excluded.",
        "- Ground truth is used only for these post-hoc metrics. Do not feed gold functions "
        "back into the selection policy.",
        "- Changes are descriptive, not proof of causality. Experimental randomized "
        "or separately executed controls are required for causal claims.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    if with_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(["Initial L1", "Final V2"], [
            100 * summary["l1"]["top5"], 100 * summary["v2"]["top5"]
        ])
        axes[0].set(ylabel="Top-5 accuracy (%)", ylim=(0, 100),
                    title=f"Same V2 runs, n={len(rows)}")
        axes[1].bar(["Initial L1", "Final V2"], [
            summary["l1"]["mean_accounted_tokens_at_l1"],
            summary["v2"]["mean_accounted_tokens"],
        ])
        axes[1].set(ylabel="Mean accounted tokens", title="Cost of adaptive stage")
        fig.tight_layout()
        fig.savefig(out / "l1_vs_v2.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir():
        ap.error(f"Missing source directory: {root}")
    if out.exists() or out == root or root in out.parents:
        ap.error("Use a NEW output directory OUTSIDE the source run.")
    rows, exclusions = collect(root)
    if not rows:
        ap.error("No eligible completed V2 arms with labels and initial L1 snapshots.")
    summary = summarize(rows, exclusions)
    save(rows, summary, out, not args.no_plots)
    print((out / "summary.md").read_text())
    print(f"\nSaved to {out}")
    print("Read-only: original 8K run unchanged; no LLM inference was performed.")


if __name__ == "__main__":
    main()

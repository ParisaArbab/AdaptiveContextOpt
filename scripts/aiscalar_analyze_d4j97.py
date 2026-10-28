#!/usr/bin/env python3
"""Read-only analysis of the archived 8K, 97-bug RAVE-FL benchmark.

Never merges the separate 16K context-recovery cohort into 8K numbers.
No model calls, no editing/replaying source runs. Writes new derived files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

POLICIES = ("full", "fixed_l1", "v2_utility")
KEYS = ("Top-1", "Top-5", "MRR")


def mean(values):
    values = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.mean(values) if values else None


def metrics(row, granularity, key):
    if row is None:
        return None
    return ((row.get("metrics") or {}).get(granularity) or {}).get(key)


def load_cases(root):
    if not root.is_dir():
        raise FileNotFoundError(root)
    cases = {}
    missing = []
    for bug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        path = bug_dir / "results.json"
        if not path.exists():
            missing.append(bug_dir.name)
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path} is not a list")
        selected = {}
        for row in rows:
            policy = row.get("policy")
            if policy not in POLICIES:
                continue
            if policy in selected:
                raise ValueError(f"Duplicate {policy} in {path}; check fusion/config")
            if row.get("synthetic"):
                raise ValueError(f"Synthetic row in live benchmark: {path}")
            selected[policy] = row
        cases[bug_dir.name] = selected
    return cases, missing


def compute(cases, granularity):
    n = len(cases)
    complete = {
        bug: by for bug, by in cases.items()
        if all(by.get(p, {}).get("status") == "completed" for p in POLICIES)
    }
    cost_matched = {
        bug: by for bug, by in complete.items()
        if all(isinstance(by[p].get("accounted_tokens"), (float, int))
               for p in POLICIES)
    }
    accuracy_matched = {
        bug: by for bug, by in complete.items()
        if all(all(isinstance(metrics(by[p], granularity, k), (int, float))
                   for k in KEYS) for p in POLICIES)
    }
    # A labeled bug has metric evidence in at least one arm, so an
    # incomplete arm gets zero credit. Missing labels remain excluded.
    labeled = {
        bug: by for bug, by in cases.items()
        if any(isinstance(metrics(by.get(p), granularity, "Top-1"), (int, float))
               for p in POLICIES)
    }
    summary = {
        "n_bug_directories": n,
        "n_all_three_completed": len(complete),
        "n_paired_cost": len(cost_matched),
        "n_paired_labeled_accuracy": len(accuracy_matched),
        "n_bugs_with_at_least_one_label": len(labeled),
        "granularity": granularity,
        "policies": {},
    }
    for policy in POLICIES:
        statuses = Counter(by.get(policy, {}).get("status", "missing") for by in cases.values())
        paired_cost_rows = [by[policy] for by in cost_matched.values()]
        paired_acc_rows = [by[policy] for by in accuracy_matched.values()]
        completed_rows = [
            by[policy] for by in cases.values()
            if by.get(policy, {}).get("status") == "completed"
        ]
        end2end = [
            by.get(policy) for by in labeled.values()
        ]
        summary["policies"][policy] = {
            "statuses": dict(statuses),
            "completion_fraction": statuses["completed"] / n if n else None,
            "paired_mean_accounted_tokens": mean(r.get("accounted_tokens") for r in paired_cost_rows),
            "paired_total_accounted_tokens": sum(r["accounted_tokens"] for r in paired_cost_rows),
            "completed_mean_accounted_tokens_unmatched": mean(
                r.get("accounted_tokens") for r in completed_rows
            ),
            "paired_mean_wall_seconds": mean(r.get("wall_seconds") for r in paired_cost_rows),
            "paired_metrics": {
                k: mean(metrics(r, granularity, k) for r in paired_acc_rows)
                for k in KEYS
            },
            "end_to_end_metrics": {
                k: mean(
                    metrics(r, granularity, k)
                    if r and r.get("status") == "completed"
                       and isinstance(metrics(r, granularity, k), (int, float))
                    else 0.0
                    for r in end2end
                ) if end2end else None
                for k in KEYS
            },
            "errors": dict(Counter(
                str(by[policy].get("error") or "none")
                for by in cases.values()
                if policy in by and by[policy].get("status") != "completed"
            )),
        }

    full_cost = summary["policies"]["full"]["paired_total_accounted_tokens"]
    for policy in ("fixed_l1", "v2_utility"):
        part = summary["policies"][policy]
        part["paired_cost_reduction_fraction_vs_full"] = (
            1 - part["paired_total_accounted_tokens"] / full_cost
            if full_cost > 0 else None
        )
    return summary, complete, cost_matched, accuracy_matched


def write_details(cases, dest, granularity):
    columns = ["bug", "project"]
    for p in POLICIES:
        columns += [f"{p}_{field}" for field in
                    ("status", "accounted_tokens", "calls", "full_calls",
                     "wall_seconds", "top1", "top5", "mrr", "fault_rank",
                     "pool_gold", "error")]
    path = dest / "per_bug.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for bug, by in sorted(cases.items()):
            data = {"bug": bug, "project": bug.split("-")[0]}
            for p in POLICIES:
                r = by.get(p, {})
                m = (r.get("metrics") or {}).get(granularity) or {}
                values = {
                    "status": r.get("status", "missing"),
                    "accounted_tokens": r.get("accounted_tokens"),
                    "calls": r.get("calls"),
                    "full_calls": r.get("full_calls"),
                    "wall_seconds": r.get("wall_seconds"),
                    "top1": m.get("Top-1"),
                    "top5": m.get("Top-5"),
                    "mrr": m.get("MRR"),
                    "fault_rank": m.get("first_relevant_rank"),
                    "pool_gold": m.get("candidate_pool_any_gold"),
                    "error": r.get("error"),
                }
                data.update({f"{p}_{key}": value for key, value in values.items()})
            writer.writerow(data)
    return path


def plots(summary, cost_matched, accuracy_matched, dest, granularity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Full", "Fixed L1", "V2 Utility"]
    scores = summary["policies"]
    # Each bar uses the SAME matched bug cohort within its own plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, [scores[p]["statuses"].get("completed", 0) for p in POLICIES])
    axes[0].set(ylabel="Completed bugs", title="Execution reliability")
    axes[0].set_ylim(0, max(1, summary["n_bug_directories"]))
    axes[1].bar(labels, [
        scores[p]["paired_mean_accounted_tokens"] or 0 for p in POLICIES
    ])
    axes[1].set(ylabel="Mean accounted input + output tokens",
                title=f"Matched completed cohort (n={len(cost_matched)})")
    fig.tight_layout()
    fig.savefig(dest / "completion_and_tokens.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key in zip(axes, KEYS):
        ax.bar(labels, [
            scores[p]["paired_metrics"][key] or 0 for p in POLICIES
        ])
        ax.set(ylabel=key, ylim=(0, 1),
               title=f"{granularity.title()} {key} (matched n={len(accuracy_matched)})")
    fig.tight_layout()
    fig.savefig(dest / "paired_accuracy.png", dpi=160)
    plt.close(fig)

    reductions = [
        100 * (1 - by["v2_utility"]["accounted_tokens"] / by["full"]["accounted_tokens"])
        for by in cost_matched.values()
        if by["full"]["accounted_tokens"] > 0
    ]
    if reductions:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(reductions, bins=min(18, max(5, len(reductions) // 5)))
        ax.axvline(0, linestyle="--")
        ax.set(xlabel="Per-bug V2 token reduction vs Full (%)",
               ylabel="Number of matched bugs",
               title=f"Distribution of V2 savings (n={len(reductions)})")
        fig.tight_layout()
        fig.savefig(dest / "v2_savings_distribution.png", dpi=160)
        plt.close(fig)


def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:,.{digits}f}"


def report(summary, cases, dest, root):
    policies = summary["policies"]
    n = summary["n_bug_directories"]
    lines = [
        "# RAVE-FL 8K Defects4J benchmark: read-only analysis",
        "",
        f"Source: {root}",
        "",
        f"Bugs with results.json: **{n}**. "
        f"All three completed: **{summary['n_all_three_completed']}**. "
        f"Matched token cohort: **{summary['n_paired_cost']}**. "
        f"Matched labeled {summary['granularity']} cohort: **{summary['n_paired_labeled_accuracy']}**.",
        "",
        "## Completion and matched token costs",
        "",
        "| Policy | Completed/available | Mean accounted tokens (matched) | "
        "Matched total tokens | Matched Top-1 | Matched Top-5 | Matched MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in POLICIES:
        s = policies[p]
        lines.append(
            f"| {p} | {s['statuses'].get('completed', 0)}/{n} | "
            f"{fmt(s['paired_mean_accounted_tokens'], 0)} | "
            f"{s['paired_total_accounted_tokens']:,} | "
            f"{fmt(s['paired_metrics']['Top-1'])} | "
            f"{fmt(s['paired_metrics']['Top-5'])} | "
            f"{fmt(s['paired_metrics']['MRR'])} |"
        )
    lines += [
        "",
        f"V2 **paired aggregate accounted-token reduction vs Full:** "
        f"{fmt(100 * policies['v2_utility']['paired_cost_reduction_fraction_vs_full'], 1) if policies['v2_utility']['paired_cost_reduction_fraction_vs_full'] is not None else 'N/A'}%.",
        "",
        "Matched means and totals include ONLY bugs where all three policies completed. "
        "Do not compare unmatched means or include the separate 16K recovery cohort.",
        "",
        f"## End-to-end {summary['granularity']} metrics (label-grounded attempts)",
        "",
        f"Denominator: **{summary['n_bugs_with_at_least_one_label']}** bugs with metric "
        "evidence from at least one policy. Incomplete arms receive zero credit, "
        "not token savings. Bugs with no available labels are excluded.",
        "",
        "| Policy | Top-1 | Top-5 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for p in POLICIES:
        s = policies[p]["end_to_end_metrics"]
        lines.append(f"| {p} | {fmt(s['Top-1'])} | {fmt(s['Top-5'])} | {fmt(s['MRR'])} |")
    lines += ["", "## Non-completed policies", ""]
    for bug, by in cases.items():
        for p in POLICIES:
            r = by.get(p)
            if not r or r.get("status") != "completed":
                err = str((r or {}).get("error") or "no error recorded")
                lines.append(f"- {bug} / {p}: {(r or {}).get('status', 'missing')}; {err}")
    lines += [
        "",
        "## Accounting and scientific limitations",
        "",
        "- Policy tokens are **accounted verification tokens**, potentially including "
        "cached observations. Do not sum across policies and call them fresh physical inference.",
        "- This run is 8K context only. The four-bug 16K recovery was a separate intervention.",
        "- Exit 0 means command success, not localization success.",
        "- Method ranking is evaluated using offline gold information; "
        "the model is not given ground truth as prompt evidence.",
        "- Top-1/Top-5/MRR are localization metrics, not patch correctness.",
        "- One collection of bugs from multiple repositories; confidence intervals, "
        "per-project stratification and robustness studies are needed before broad claims.",
        "",
        "## Figures",
        "",
        "![Completion and token costs](completion_and_tokens.png)",
        "![Paired accuracy](paired_accuracy.png)",
        "![V2 savings distribution](v2_savings_distribution.png)",
    ]
    (dest / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Original 8K per-bug batch results root")
    ap.add_argument("--out", type=Path, required=True, help="NEW derived output folder")
    ap.add_argument("--granularity", choices=["file", "method"], default="method")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    dest = args.out.resolve()
    if dest.exists():
        ap.error("Output already exists: select a fresh destination, never overwrite prior analysis.")
    cases, missing = load_cases(root)
    if not cases:
        ap.error("No readable result cases found.")
    if dest == root or root in dest.parents:
        ap.error("Write analysis OUTSIDE the source 8K run directory.")
    dest.mkdir(parents=True)
    summary, complete, cost, accuracy = compute(cases, args.granularity)
    summary["missing_result_files"] = missing
    summary["source_path"] = str(root)
    (dest / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_details(cases, dest, args.granularity)
    if not args.no_plots:
        plots(summary, cost, accuracy, dest, args.granularity)
    report(summary, cases, dest, root)
    print(f"Analyzed {len(cases)} bug result folders; all-three completed: {len(complete)}")
    print(f"Paired cost: {len(cost)}; paired labeled accuracy: {len(accuracy)}")
    print("V2 reduction vs Full (matched):",
          fmt(100 * (summary["policies"]["v2_utility"]
                     ["paired_cost_reduction_fraction_vs_full"]))
          if summary["policies"]["v2_utility"]["paired_cost_reduction_fraction_vs_full"] is not None
          else "N/A", "%")
    print("Derived outputs:", dest)
    print("NOTE: no model requests were made; original results were not changed.")


if __name__ == "__main__":
    main()

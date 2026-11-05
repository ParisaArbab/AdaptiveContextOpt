"""English run summaries with explicit operational and matched denominators."""
from collections import defaultdict, Counter
import json
from pathlib import Path
import statistics
from rave_fl.pilot import write_json
from rave_fl.eval.schema import file_hash
from rave_fl.graft.evaluation import PROFILES


def report(directory):
    directory = Path(directory)
    rows = json.loads((directory / "results.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {
        (p, f)
        for p in ["prior"]
        + manifest["config"]["benchmark"]["policies"]
        + manifest["config"]["graft_policies"]
        for f in manifest["config"]["benchmark"]["fusions"]
    }
    groups = defaultdict(list)
    for row in rows:
        if "policy" in row:
            groups[row["instance_id"]].append(row)
    summaries = []
    for profile in PROFILES:
        matched = {
            ident
            for ident, group in groups.items()
            if {(r["policy"], r["fusion"]) for r in group} == expected
            and all(
                r["status"] == "completed"
                and r["metrics"].get(profile, {}).get("method") is not None
                for r in group
            )
        }
        for policy, fusion in sorted(expected):
            selected = [
                r
                for r in rows
                if r.get("policy") == policy
                and r.get("fusion") == fusion
                and r.get("metrics", {}).get(profile, {}).get("method") is not None
            ]
            for cohort, cohort_rows in [
                ("known_labeled_operational", selected),
                ("matched_completed_labeled", [r for r in selected if r["instance_id"] in matched]),
            ]:
                if not cohort_rows:
                    continue
                entry = {
                    "profile": profile,
                    "policy": policy,
                    "fusion": fusion,
                    "cohort": cohort,
                    "n": len(cohort_rows),
                    "completed": sum(r["status"] == "completed" for r in cohort_rows),
                }
                for metric in ["Top-1", "Top-5", "MRR"]:
                    entry[metric] = statistics.mean(
                        r["metrics"][profile]["method"][metric] if r["status"] == "completed" else 0
                        for r in cohort_rows
                    )
                observed_costs = [
                    r["accounted_tokens"]
                    for r in cohort_rows
                    if r.get("accounted_tokens") is not None
                ]
                entry["mean_accounted_verification_tokens"] = (
                    statistics.mean(observed_costs) if observed_costs else None
                )
                summaries.append(entry)
    spending = json.loads((directory / "accounting.json").read_text())
    summary = {
        "schema": "graft-summary-v1",
        "rows": summaries,
        "selected_instances": len(manifest["instances"]),
        "arm_statuses": dict(Counter(r["status"] for r in rows)),
        "spending": spending,
        "results_sha256": file_hash(directory / "results.json"),
        "synthetic": spending["synthetic"],
    }
    write_json(directory / "summary.json", summary)
    lines = [
        "# GRAFT execution report",
        "",
        "**Synthetic fixture: software validation only; these scores are not research findings.**"
        if spending["synthetic"]
        else "**Development experiment; held-out status must be established from a preselected split manifest.**",
        "",
        f"Selected instances: {len(manifest['instances'])}. Arm statuses: {summary['arm_statuses']}.",
        "Verification cost is provider input plus output, including paid invalid responses. Retrieval, embedding, test execution and whole-pipeline cost are separate. Exact-cache reuse retains logical policy cost but creates no new provider request.",
        "",
        "| Profile | Cohort | Policy | Fusion | Labeled n | Completed | Top-1 | Top-5 | MRR | Mean verification tokens |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        cost = (
            f"{r['mean_accounted_verification_tokens']:.1f}"
            if r["mean_accounted_verification_tokens"] is not None
            else "Unavailable"
        )
        lines.append(
            f"| {r['profile']} | {r['cohort']} | {r['policy']} | {r['fusion']} | {r['n']} | {r['completed']} | {r['Top-1']:.4f} | {r['Top-5']:.4f} | {r['MRR']:.4f} | {cost} |"
        )
    lines += [
        "",
        "Operational scores count incomplete labeled arms as zero; matched scores require every configured comparison to complete with labels. Unlabeled instances are not zero-accuracy defects. Empty matched cohorts are omitted, not evidence of equivalence.",
        "",
        "Actual spending and unresolved reservations:",
        "",
        "```json",
        json.dumps(spending, indent=2),
        "```",
        "",
        "Source: results.json, accounting.json and manifest.json in this run. Per-arm files retain selected/rejected actions, paid observations and stage histories. raw_ledgers/requests.jsonl and journal.sqlite3 preserve exact provider requests and counters. No cache-based wall-time speedup is claimed.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    return {
        "report": str(directory / "REPORT.md"),
        "summary": str(directory / "summary.json"),
        "synthetic": spending["synthetic"],
    }

"""Report-only aggregation, compatible paired comparisons and English figures."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import random
import shlex
import textwrap

from rave_fl.eval.metrics import aggregate
from rave_fl.eval.schema import CompatibilityError, fingerprint, file_hash
from rave_fl.pilot import write_json


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in r.items()
                }
            )


def markdown(rows, keys):
    def value(v):
        return (
            "Unavailable"
            if v is None
            else str(round(v, 5))
            if isinstance(v, float)
            else str(v).replace("|", "/").replace("\n", " ")
        )

    return "\n".join(
        ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
        + ["| " + " | ".join(value(r.get(k)) for k in keys) + " |" for r in rows]
    )


def paired_uncertainty(rows, seed=0):
    """Offline paired deltas, clustered by declared defect/repository, never action rows."""
    if any(r.get("synthetic") for r in rows):
        return []
    base = {
        (r["instance_id"], r["fusion"]): r
        for r in rows
        if r["policy"] == "fixed_l1" and r["status"] == "completed"
    }
    groups = defaultdict(list)
    for r in rows:
        b = base.get((r["instance_id"], r["fusion"]))
        if not b or r["status"] != "completed" or r["policy"] == "fixed_l1":
            continue
        for gran in ["file", "method"]:
            if not r["metrics"].get(gran) or not b["metrics"].get(gran):
                continue
            key = (r["policy"], r["fusion"], r.get("language"), gran)
            groups[key].append(
                (
                    r.get("independent_bug_id", r["instance_id"]),
                    r["repository"],
                    r["metrics"][gran]["MRR"] - b["metrics"][gran]["MRR"],
                )
            )
    result = []
    for key, group in groups.items():
        bugs = defaultdict(list)
        repos = {}
        for defect, repo, delta in group:
            bugs[defect].append(delta)
            repos[defect] = repo
        if len(bugs) < 10:
            continue
        clusters = defaultdict(list)
        for defect, values in bugs.items():
            clusters[repos[defect]].append(statistics.mean(values))
        by_project = len(clusters) >= 3
        units = (
            list(clusters.values()) if by_project else [[statistics.mean(v)] for v in bugs.values()]
        )
        rng = random.Random(seed)
        values = []
        for _ in range(2000):
            sample = [v for unit in rng.choices(units, k=len(units)) for v in unit]
            values.append(statistics.mean(sample))
        values.sort()
        result.append(
            {
                "policy": key[0],
                "fusion": key[1],
                "language": key[2],
                "granularity": key[3],
                "reference": "fixed_l1",
                "independent_bugs": len(bugs),
                "delta_MRR": statistics.mean(statistics.mean(v) for v in bugs.values()),
                "lower_95": values[49],
                "upper_95": values[1949],
                "method": "paired repository-cluster bootstrap"
                if by_project
                else "paired defect-cluster bootstrap",
                "replicates": 2000,
                "seed": seed,
            }
        )
    return result


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    opts = manifest["config"]
    if fingerprint({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise CompatibilityError("Saved manifest checksum mismatch")
    audit = json.loads((directory / "audits/results.json").read_text())
    if file_hash(directory / "results.json") != audit["results_sha256"]:
        raise CompatibilityError("Saved aggregate result checksum mismatch")
    for status_file in (directory / "instances").glob("*/status.json"):
        for arm in json.loads(status_file.read_text()).get("arms", {}).values():
            if file_hash(directory / arm["file"]) != arm["sha256"]:
                raise CompatibilityError(
                    "Saved result checksum mismatch; preserve raw data before investigating"
                )
    rows = json.loads((directory / "results.json").read_text())
    summary = aggregate(rows)
    write_json(directory / "summary.json", summary)
    synthetic = any(r.get("synthetic", False) for r in rows)
    tables = directory / "tables"
    figures = directory / "figures"
    reports = directory / "reports"
    for p in [tables, figures, reports]:
        p.mkdir(parents=True, exist_ok=True)
    flat = []
    decisions = []
    stages = []
    trajectories = []
    fusion_rows = []
    for row in rows:
        arm = row.get("arm", {})
        basic = {
            k: row.get(k)
            for k in [
                "instance_id",
                "dataset",
                "policy",
                "fusion",
                "admission",
                "evaluation_mode",
                "status",
                "accounted_tokens",
                "input_tokens",
                "output_tokens",
                "full_calls",
                "calls",
                "wall_seconds",
                "budget_exceeded",
                "unused_budget",
                "incomplete_l1_coverage",
                "error",
            ]
        }
        for granularity in ["file", "method"]:
            metric = row.get("metrics", {}).get(granularity)
            flat.append(
                {
                    **basic,
                    "granularity": granularity,
                    **(metric or {}),
                    "metric_available": metric is not None,
                    "gold_semantics": row.get("metrics", {}).get("gold_semantics"),
                }
            )
        for trace in arm.get("trace", []):
            decisions.append({**basic, **trace})
        for level, label in [
            (0, "L0 discovery"),
            (1, "L1 batching or expansion"),
            (2, "Full verification"),
        ]:
            stages.append(
                {
                    **basic,
                    "stage": label,
                    "tokens": sum(
                        c["input_tokens"] + c["output_tokens"]
                        for c in arm.get("calls", [])
                        if c["level"] == level
                    ),
                }
            )
        for rank, ident in enumerate(arm.get("ranked_functions", []), 1):
            trajectories.append(
                {
                    **basic,
                    "candidate_id": ident,
                    "final_rank": rank,
                    "score": arm["ranking_scores"][ident],
                    "level": arm["states"][ident]["level"],
                }
            )
        scores = list(arm.get("ranking_scores", {}).values())
        rounded = [
            round(v, opts["rounding"]) if opts["rounding"] is not None else v for v in scores
        ]
        fusion_rows.append(
            {
                **basic,
                "tied_candidate_count": len(scores) - len(set(scores)),
                "effective_tied_candidate_count": len(rounded) - len(set(rounded)),
                "selected_Full_ids": [
                    o["function_id"]
                    for c in arm.get("calls", [])
                    if c["level"] == 2
                    for o in c["observations"]
                ],
                "fusion_config": row.get("fusion_config", {}),
                "interpretation": "Frozen observations isolate reranking; policy mode includes changed decisions",
            }
        )
    csv_file(
        tables / "score_trajectories.csv",
        [
            {"instance_id": r["instance_id"], "policy": r["policy"], "fusion": r["fusion"], **v}
            for r in rows
            for v in r.get("score_trajectories", [])
        ],
    )
    csv_file(tables / "per_bug.csv", flat)
    csv_file(tables / "aggregate.csv", summary)
    csv_file(tables / "decision_traces.csv", decisions)
    csv_file(tables / "stage_tokens.csv", stages)
    csv_file(tables / "candidate_rankings.csv", trajectories)
    csv_file(tables / "fusion_sensitivity.csv", fusion_rows)
    status = json.loads((directory / "execution_status.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    write_json(
        directory / "audits/report.json",
        {
            "status": "verified_saved_accounting",
            "rows": len(rows),
            "arm_accounting_matches_calls": all(
                r.get("arm") is None
                or r["accounted_tokens"]
                == sum(c["input_tokens"] + c["output_tokens"] for c in r["arm"]["calls"])
                for r in rows
            ),
            "usage_unknown": status["spending"]["usage_unknown"],
            "synthetic": synthetic,
            "report_generator_sha256": file_hash(__file__),
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "rave-eval-v1",
        }
    )
    figure_index = []

    def save(fig, name, title, metric, source):
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.07, 1, 0.94])
        caption = f"Dataset {opts['dataset']}; run {opts['run_id']}; model {opts['model']}; admission {opts['admission']}; mode {opts['evaluation_mode']}; source {source}; metric {metric}. Verification tokens = input + output provider counters."
        if synthetic:
            caption += " SYNTHETIC FIXTURE; no real model inference."
        fig.text(0.01, 0.01, textwrap.fill(caption, 150), fontsize=6)
        fig.savefig(
            figures / (name + ".png"),
            dpi=200,
            bbox_inches="tight",
            metadata={"Description": caption},
        )
        fig.savefig(
            figures / (name + ".svg"),
            bbox_inches="tight",
            metadata={"Description": caption, "Date": None},
        )
        plt.close(fig)
        figure_index.append(
            {"name": name, "title": title, "source": source, "metric": metric, "caption": caption}
        )

    completed = [r for r in rows if r["status"] == "completed"]
    plot_groups = {}
    for r in completed:
        plot_groups.setdefault((r["policy"], r["fusion"]), []).append(r)
    plot_groups = dict(sorted(plot_groups.items())[:30])
    fusion_label = {
        "shared_prior_evidence_fusion_v1": "Historical",
        "consistent_score": "Consistent",
        "rrf": "RRF",
        "prior_aware": "Prior aware",
    }
    names = [p + " / " + fusion_label[f] for p, f in plot_groups]
    plotted = list(plot_groups.values())
    if completed:
        fig, ax = plt.subplots(figsize=(11, max(4, len(plotted) * 0.3)))
        ax.barh(
            names,
            [statistics.mean(r["accounted_tokens"] for r in group) for group in plotted],
            color=colors[0],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean completed accounted verification tokens")
        save(
            fig,
            "verification_tokens",
            "Mean verification tokens by policy and fusion",
            "accounted input+output",
            "results.json",
        )
        fig, ax = plt.subplots(figsize=(11, max(4, len(plotted) * 0.3)))
        ax.barh(
            names,
            [statistics.mean(r["full_calls"] for r in group) for group in plotted],
            color=colors[2],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean completed Full inspections")
        save(fig, "full_inspections", "Selected Full inspections", "Full calls", "results.json")
        fig, ax = plt.subplots(figsize=(11, max(4, len(plotted) * 0.3)))
        ax.barh(
            names,
            [statistics.mean(r["unused_budget"] for r in group) for group in plotted],
            color=colors[1],
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean unused per-policy verification budget")
        save(
            fig,
            "unused_budget",
            "Admission and unused budget",
            "remaining verification tokens",
            "results.json",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for gran, color in [("file", colors[0]), ("method", colors[1])]:
            valid = [r for r in completed if r["metrics"].get(gran)]
            if valid:
                ax.scatter(
                    [r["accounted_tokens"] for r in valid],
                    [r["metrics"][gran]["MRR"] for r in valid],
                    label=gran,
                    color=color,
                )
        ax.set(xlabel="Accounted verification tokens", ylabel="MRR")
        ax.legend() if ax.collections else None
        save(
            fig,
            "cost_quality",
            "Cost versus localization quality",
            "MRR; completed labeled results only",
            "results.json",
        )
    if summary:
        settings = list(
            dict.fromkeys((r["language"], r["granularity"], r["denominator"]) for r in summary)
        )
        for number, setting in enumerate(settings):
            group = [
                r for r in summary if (r["language"], r["granularity"], r["denominator"]) == setting
            ][:30]
            fig, axs = plt.subplots(1, 3, figsize=(13, max(4, len(group) * 0.27)), sharey=True)
            labels = [r["policy"] + " / " + fusion_label[r["fusion"]] for r in group]
            for ax, key in zip(axs, ["Top-1", "Top-5", "MRR"]):
                ax.barh(range(len(group)), [r[key] for r in group], color=colors[0])
                ax.set(title=key, xlim=(0, 1.05))
                ax.set_yticks(range(len(group)))
            axs[0].set_yticklabels(labels, fontsize=8)
            axs[0].invert_yaxis()
            denominator = (
                "All attempted labeled bugs"
                if setting[2].startswith("end")
                else "Completed bugs with gold in pool"
            )
            save(
                fig,
                "accuracy" if number == 0 else "accuracy_" + str(number + 1),
                f"{setting[0]} / {setting[1]}: {denominator}",
                "Top-1 / Top-5 / MRR; first 30 settings per panel, all rows in CSV",
                "tables/aggregate.csv",
            )
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = defaultdict(int)
    for s in status["instances"]:
        counts[s["status"]] += 1
    ax.bar(list(counts), list(counts.values()), color=colors[: len(counts)])
    ax.set_ylabel("Attempted instances; variants may be correlated")
    save(
        fig,
        "completion",
        "Pipeline completion and failure states",
        "attempted bug statuses",
        "execution_status.json",
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    spend = status["spending"]
    values = [spend["physical_inference_tokens"], sum(r["accounted_tokens"] or 0 for r in rows)]
    ax.bar(["Physical real inference", "Accounted policy evaluations"], values, color=colors[:2])
    ax.set_ylabel("Input + output tokens")
    save(
        fig,
        "physical_accounted",
        "Physical inference versus policy accounting",
        "shared collection versus per-policy accounting",
        "execution_status.json; results.json",
    )
    if stages and completed:
        fig, ax = plt.subplots(figsize=(10, 5))
        totals = defaultdict(int)
        for r in stages:
            totals[r["stage"]] += r["tokens"]
        ax.bar(list(totals), list(totals.values()), color=colors[: len(totals)])
        ax.set_ylabel("Accounted tokens across selected arms")
        save(
            fig,
            "stage_costs",
            "Stage-by-stage verification costs",
            "paid action levels; repeated per-policy accounting",
            "tables/stage_tokens.csv",
        )
    # Additional diagnostics are emitted only when the corresponding measured
    # fields exist. Missing results are never rendered as zero.
    valid = [r for r in completed if r["metrics"].get("method") or r["metrics"].get("file")]
    if valid:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key in zip(axs, ["Top-1", "Top-5", "MRR"]):
            for r in valid:
                gran = "method" if r["metrics"].get("method") else "file"
                ax.scatter(r["accounted_tokens"], r["metrics"][gran][key], s=18)
            ax.set(xlabel="Accounted verification tokens", ylabel=key)
        save(
            fig,
            "accuracy_cost",
            "Accuracy versus verification tokens",
            "Per-bug Top-1/5 and MRR; granularity in CSV",
            "tables/per_bug.csv",
        )
        max_groups = max(
            sum(r["granularity"] == g and r["denominator"].startswith("end") for r in summary)
            for g in ["file", "method"]
        )
        fig, axs = plt.subplots(1, 2, figsize=(16, max(4, min(30, max_groups) * 0.27)))
        for gran, ax in zip(["file", "method"], axs):
            group = [
                r
                for r in summary
                if r["granularity"] == gran and r["denominator"].startswith("end")
            ][:30]
            if group:
                ax.barh(
                    [
                        r["policy"] + " / " + fusion_label[r["fusion"]] + " / " + str(r["language"])
                        for r in group
                    ],
                    [r["MRR"] for r in group],
                    color=colors[0],
                )
            ax.set(title=gran + " fusion comparison", xlabel="MRR")
            ax.tick_params(axis="y", labelsize=7)
        save(
            fig,
            "fusion_comparison",
            "Fusion-strategy comparison",
            "End-to-end operational MRR; language/granularity separated",
            "tables/aggregate.csv",
        )
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for gran, ax in zip(["file", "method"], axs):
            vv = [r for r in valid if r["metrics"].get(gran)]
            if vv:
                ax.hist(
                    [r["metrics"][gran]["candidate_pool_coverage"] for r in vv],
                    bins=[0, 0.001, 0.25, 0.5, 0.75, 1.001],
                    color=colors[0],
                )
            ax.set(
                title=gran + " pool coverage",
                xlabel="Fraction of gold locations present",
                ylabel="Labeled policy results (correlated within bug)",
            )
        save(
            fig,
            "candidate_coverage",
            "Candidate-pool coverage",
            "Offline gold coverage; policy rows are not independent bugs",
            "tables/per_bug.csv",
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ids = list(dict.fromkeys(r["instance_id"] for r in valid))[:12]
        for gran, ax in zip(["file", "method"], axes):
            for index, policy in enumerate(opts["policies"]):
                vv = [
                    r
                    for r in valid
                    if r["policy"] == policy
                    and r["fusion"] == opts["fusions"][0]
                    and r["instance_id"] in ids
                    and r["metrics"].get(gran)
                    and r["metrics"][gran]["first_relevant_rank"] is not None
                ]
                if vv:
                    ax.scatter(
                        [
                            ids.index(r["instance_id"])
                            + (index - (len(opts["policies"]) - 1) / 2) * 0.07
                            for r in vv
                        ],
                        [r["metrics"][gran]["first_relevant_rank"] for r in vv],
                        s=25,
                        label=policy,
                        color=colors[index % len(colors)],
                    )
            ax.set_xticks(range(len(ids)), ids, rotation=65, ha="right", fontsize=7)
            ax.set(title=gran, xlabel="Bug instance", ylabel="First relevant rank")
            ax.legend(fontsize=7) if ax.collections else None
        save(
            fig,
            "per_bug_ranks",
            "Per-bug fault ranking",
            "First 12 bugs and primary configured fusion; missing locations omitted, full table retained",
            "tables/per_bug.csv",
        )
    savings = []
    base = {(r["instance_id"], r["fusion"]): r for r in completed if r["policy"] == "full"}
    for r in completed:
        b = base.get((r["instance_id"], r["fusion"]))
        if b and b["accounted_tokens"] > 0:
            savings.append(
                {
                    "instance_id": r["instance_id"],
                    "policy": r["policy"],
                    "fusion": r["fusion"],
                    "saved_fraction": 1 - r["accounted_tokens"] / b["accounted_tokens"],
                }
            )
    csv_file(tables / "per_bug_savings.csv", savings)
    if savings:
        grouped = {}
        for r in savings:
            grouped.setdefault(r["policy"] + " / " + fusion_label[r["fusion"]], []).append(
                r["saved_fraction"]
            )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot(list(grouped.values()))
        ax.set_xticks(range(1, len(grouped) + 1), list(grouped))
        ax.tick_params(axis="x", labelrotation=80, labelsize=7)
        ax.set_ylabel("Fraction of verification tokens saved versus completed same-fusion Full")
        save(
            fig,
            "token_reduction",
            "Per-bug token savings distribution",
            "Box median/distribution, not statistical confidence intervals",
            "tables/per_bug_savings.csv",
        )
    if completed:
        fig, ax = plt.subplots(figsize=(11, max(4, len(plotted) * 0.25)))
        y = list(range(len(plotted)))
        inputs = [statistics.mean(r["input_tokens"] for r in group) for group in plotted]
        outputs = [statistics.mean(r["output_tokens"] for r in group) for group in plotted]
        ax.barh(y, inputs, label="Input")
        ax.barh(y, outputs, left=inputs, label="Output")
        ax.set_yticks(y, names)
        ax.legend()
        ax.set_xlabel("Mean accounted provider tokens")
        save(
            fig,
            "input_output",
            "Input versus output token costs",
            "First 30 policy/fusion settings; full tables retained",
            "tables/per_bug.csv",
        )
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].barh(names, [statistics.mean(r["wall_seconds"] for r in group) for group in plotted])
        axs[0].set_xlabel("Mean orchestration wall seconds (includes cache)")
        reasons = defaultdict(int)
        for r in completed:
            reasons[r["arm"]["stop_reason"]] += 1
        axs[1].barh(list(reasons), list(reasons.values()))
        axs[1].set_xlabel("Stop occurrences across policy results")
        save(
            fig,
            "runtime_decisions",
            "Runtime and controller stopping patterns",
            "Wall time is not GPU/model throughput; synthetic timing is not hardware validation",
            "results.json; decision_traces.csv",
        )
        first = completed[0]["instance_id"]
        pp = completed[0]["policy"]
        selected = [r for r in completed if r["instance_id"] == first and r["policy"] == pp]
        ids = selected[0]["arm"]["ranked_functions"]
        matrix = [[r["arm"]["ranked_functions"].index(i) + 1 for r in selected] for i in ids]
        fig, ax = plt.subplots(figsize=(8, max(4, len(ids) * 0.2)))
        ax.imshow(matrix, aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
        ax.set_xticks(
            range(len(selected)),
            [r["fusion"] for r in selected],
            rotation=50,
            ha="right",
            fontsize=7,
        )
        save(
            fig,
            "fusion_sensitivity",
            "Candidate rank sensitivity for " + first + " / " + pp,
            "Final ranks; policy mode may change selected actions, frozen mode isolates fusion",
            "candidate_rankings.csv; fusion_sensitivity.csv",
        )
    projects = []
    for repository in sorted({r.get("repository", "unavailable") for r in rows}):
        for r in aggregate([v for v in rows if v.get("repository", "unavailable") == repository]):
            projects.append({"repository": repository, **r})
    csv_file(tables / "per_repository_language.csv", projects)
    if projects:
        fig, ax = plt.subplots(figsize=(11, 5))
        vv = [r for r in projects if r["denominator"].startswith("end")]
        for r in vv:
            ax.scatter(
                r["repository"] + " / " + str(r["language"]) + " / " + r["granularity"],
                r["MRR"],
                s=22,
            )
        ax.tick_params(axis="x", rotation=65, labelsize=6)
        ax.set_ylabel("End-to-end operational MRR")
        save(
            fig,
            "project_language",
            "Per-repository and per-language results",
            "Granularity explicitly separated; policy rows correlated",
            "per_repository_language.csv",
        )
    failures = []
    fixed = {(r["instance_id"], r["fusion"]): r for r in completed if r["policy"] == "fixed_l1"}
    for r in completed:
        b = fixed.get((r["instance_id"], r["fusion"]))
        if not b or r["policy"] == "fixed_l1":
            continue
        for gran in ["file", "method"]:
            a, z = b["metrics"].get(gran), r["metrics"].get(gran)
            if a and z:
                delta = z["MRR"] - a["MRR"]
                failures.append(
                    {
                        "instance_id": r["instance_id"],
                        "policy": r["policy"],
                        "fusion": r["fusion"],
                        "granularity": gran,
                        "delta_MRR": delta,
                        "outcome": "improved"
                        if delta > 1e-12
                        else "harmed"
                        if delta < -1e-12
                        else "unchanged",
                    }
                )
    csv_file(tables / "failure_analysis.csv", failures)
    if failures:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, gran in zip(axs, ["file", "method"]):
            counts = {
                k: sum(r["granularity"] == gran and r["outcome"] == k for r in failures)
                for k in ["improved", "harmed", "unchanged"]
            }
            ax.bar(list(counts), list(counts.values()), color=colors[:3])
            ax.set_title(gran + " versus matched Fixed L1")
            ax.set_ylabel("Paired policy outcomes (not independent bug counts)")
        save(
            fig,
            "failure_analysis",
            "Full escalation and ranking changes",
            "Marginal final MRR versus matched Fixed L1; no causal claim across schedules",
            "failure_analysis.csv",
        )
    # A frontier uses exactly the same completed bugs per policy, and is never
    # pooled across language, gold granularity, fusion or admission semantics.
    pareto = []
    for language in sorted({r.get("language") for r in completed}):
        for fusion in opts["fusions"]:
            for gran in ["file", "method"]:
                groups = {
                    p: {
                        r["instance_id"]: r
                        for r in completed
                        if r["policy"] == p
                        and r["fusion"] == fusion
                        and r.get("language") == language
                        and r["metrics"].get(gran)
                    }
                    for p in opts["policies"]
                }
                groups = {p: g for p, g in groups.items() if g}
                common = set.intersection(*(set(g) for g in groups.values())) if groups else set()
                values = [
                    {
                        "language": language,
                        "fusion": fusion,
                        "granularity": gran,
                        "policy": p,
                        "matched_instances": sorted(common),
                        "mean_tokens": statistics.mean(g[i]["accounted_tokens"] for i in common),
                        "MRR": statistics.mean(g[i]["metrics"][gran]["MRR"] for i in common),
                    }
                    for p, g in groups.items()
                    if common
                ]
                for v in values:
                    v["nondominated"] = not any(
                        w["mean_tokens"] <= v["mean_tokens"]
                        and w["MRR"] >= v["MRR"]
                        and (w["mean_tokens"] < v["mean_tokens"] or w["MRR"] > v["MRR"])
                        for w in values
                    )
                pareto.extend(values)
    csv_file(tables / "matched_frontiers.csv", pareto)
    if pareto:
        settings = list(
            dict.fromkeys((r["language"], r["fusion"], r["granularity"]) for r in pareto)
        )[:4]
        fig, axes = plt.subplots(len(settings), 1, figsize=(10, 3.5 * len(settings)), squeeze=False)
        for ax, setting in zip(axes[:, 0], settings):
            vv = [r for r in pareto if (r["language"], r["fusion"], r["granularity"]) == setting]
            for v in vv:
                ax.scatter(
                    v["mean_tokens"],
                    v["MRR"],
                    label=v["policy"],
                    marker="D" if v["nondominated"] else "o",
                )
            frontier = sorted([v for v in vv if v["nondominated"]], key=lambda v: v["mean_tokens"])
            ax.plot(
                [v["mean_tokens"] for v in frontier],
                [v["MRR"] for v in frontier],
                color=colors[0],
                alpha=0.4,
            )
            ax.set(
                title=" / ".join(setting) + f"; {len(vv[0]['matched_instances'])} matched bugs",
                xlabel="Mean accounted verification tokens",
                ylabel="MRR",
            )
            ax.legend(fontsize=7)
        save(
            fig,
            "pareto",
            "Matched-bug cost and ranking frontiers",
            "First four language/fusion/granularity settings; full matched data in CSV; descriptive nondominance",
            "tables/matched_frontiers.csv",
        )
    if completed and completed[0].get("score_trajectories"):
        r = completed[0]
        history = r["score_trajectories"]
        steps = sorted({h["step"] for h in history})
        ids = r["arm"]["ranked_functions"]
        lookup = {(h["candidate_id"], h["step"]): h["rank"] for h in history}
        fig, ax = plt.subplots(figsize=(10, max(4, len(ids) * 0.2)))
        matrix = [[lookup[i, s] for s in steps] for i in ids]
        im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
        ax.set_xticks(range(len(steps)), steps, fontsize=7)
        ax.set_xlabel("Selected action step; zero is prior ranking")
        fig.colorbar(im, ax=ax, label="Candidate rank")
        save(
            fig,
            "ranking_evolution",
            "Ranking evolution: " + r["instance_id"] + " / " + r["policy"],
            "Candidate ranks after each revealed action; no oracle actions",
            "tables/score_trajectories.csv",
        )
    paired_fusions = []
    for r in completed:
        base = next(
            (
                v
                for v in completed
                if v["instance_id"] == r["instance_id"]
                and v["policy"] == r["policy"]
                and v["fusion"] == opts["fusions"][0]
            ),
            None,
        )
        if base is None or r["fusion"] == base["fusion"]:
            continue
        actions = lambda v: [c["candidate_ids"] for c in v["arm"]["calls"] if c["level"] == 2]
        for gran in ["file", "method"]:
            a, b = base["metrics"].get(gran), r["metrics"].get(gran)
            if a and b:
                paired_fusions.append(
                    {
                        "instance_id": r["instance_id"],
                        "policy": r["policy"],
                        "fusion": r["fusion"],
                        "baseline_fusion": base["fusion"],
                        "granularity": gran,
                        "delta_MRR": b["MRR"] - a["MRR"],
                        "delta_tokens": r["accounted_tokens"] - base["accounted_tokens"],
                        "Full_actions_changed": actions(r) != actions(base),
                        "mode": opts["evaluation_mode"],
                    }
                )
    csv_file(tables / "paired_fusion_differences.csv", paired_fusions)
    intervals = paired_uncertainty(completed, opts["seed"]) if not opts.get("replay_run") else []
    csv_file(tables / "statistical_intervals.csv", intervals)
    if intervals:
        fig, ax = plt.subplots(figsize=(11, max(4, len(intervals) * 0.3)))
        for index, r in enumerate(intervals):
            ax.errorbar(
                r["delta_MRR"],
                index,
                xerr=[
                    [r["delta_MRR"] - min(r["delta_MRR"], r["lower_95"])],
                    [max(r["delta_MRR"], r["upper_95"]) - r["delta_MRR"]],
                ],
                fmt="o",
                color=colors[0],
            )
        ax.set_yticks(
            range(len(intervals)),
            [r["policy"] + " / " + r["fusion"] + " / " + r["granularity"] for r in intervals],
            fontsize=7,
        )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Paired MRR difference versus Fixed L1; bootstrap 95% interval")
        save(
            fig,
            "paired_uncertainty",
            "Independent-defect paired comparisons",
            "Declared defect groups collapsed; project clusters when available; completed matched pairs only",
            "tables/statistical_intervals.csv",
        )
    ledger = directory / "raw_ledgers/requests.jsonl"
    throughput = []
    if ledger.exists() and not synthetic:
        for event in map(json.loads, ledger.read_text().splitlines()):
            if event.get("event") == "response" and event.get("duration_seconds", 0) > 0:
                response = event["response"]
                throughput.append(
                    {
                        "fingerprint": event["fingerprint"],
                        "duration_seconds": event["duration_seconds"],
                        "input_tokens": response["prompt_eval_count"],
                        "output_tokens": response["eval_count"],
                        "output_tokens_per_client_second": response["eval_count"]
                        / event["duration_seconds"],
                    }
                )
    csv_file(tables / "fresh_request_throughput.csv", throughput)
    if throughput:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].hist([r["duration_seconds"] for r in throughput], color=colors[0])
        axs[0].set(xlabel="Fresh request client wall seconds", ylabel="Requests")
        axs[1].hist([r["output_tokens_per_client_second"] for r in throughput], color=colors[2])
        axs[1].set(xlabel="Output tokens / client wall second", ylabel="Requests")
        save(
            fig,
            "request_throughput",
            "Observed provider request timing",
            "Client wall time includes transport; not GPU kernel time or peak memory",
            "tables/fresh_request_throughput.csv",
        )
    availability = {
        "budget_sensitivity": "One configured budget/call ceiling per run; compare multiple matched saved runs for sensitivity",
        "GPU_peak_memory": "Unavailable: not measured by this client",
        "statistical_intervals": "Available in tables/statistical_intervals.csv"
        if intervals
        else "Unavailable for synthetic fixtures or insufficient independent bugs/projects",
        "historical_timeline": "Preserved at reports/figures/01_research_timeline.* in the repository; never relabeled as new benchmark data",
    }
    write_json(directory / "figure_availability.json", availability)
    csv_file(tables / "figures.csv", figure_index)
    reproduce = (
        "bash scripts/rave_fl_run.sh run --config "
        + shlex.quote(str(directory / "reproduction_config.json"))
        + " --output-dir /ABSOLUTE/NEW_OUTPUT_DIRECTORY"
    )
    if (
        opts["backend"] != "fake"
        and not opts.get("replay_run")
        and opts["evaluation_mode"] == "policy"
    ):
        reproduce += " --allow-inference"
    lines = [
        "# RAVE-FL Universal Benchmark Evaluation",
        "",
        f"Dataset: **{opts['dataset']}**; run: `{opts['run_id']}`; model: `{opts['model']}`; admission: `{opts['admission']}`; mode: `{opts['evaluation_mode']}`.",
        "",
        "**SYNTHETIC FIXTURE EXECUTION. No real model inference or benchmark accuracy claim.**"
        if synthetic
        else "Localization/verification evaluation, not patch generation or SWE-bench issue resolution.",
        "",
        "## Execution and accounting",
        "",
        json.dumps(status["spending"], indent=2),
        "",
        "Physical counters count fresh requests once. Accounted costs charge every selected observation per policy. Frozen reranking and conditional replay are not fresh inference. Unknown usage remains separately reserved and blocks new requests. Cache savings do not establish whole-pipeline savings.",
        "",
        "## Aggregate results",
        "",
        markdown(
            summary,
            [
                "policy",
                "fusion",
                "granularity",
                "denominator",
                "n",
                "attempted",
                "completed",
                "failed_or_incomplete",
                "Top-1",
                "Top-5",
                "MRR",
                "Recall@5",
                "candidate_pool_coverage",
                "mean_tokens",
                "median_tokens",
                "mean_token_savings",
                "completion_rate",
            ],
        ),
        "",
        "End-to-end operational performance uses all attempted bugs with available labels, assigning zero credit to incomplete attempts by definition. Conditional verification uses completed bugs with at least one gold entity in the pool. Individual unavailable measurements remain null. First-relevant rank differs from all-location recall; gold patch changes are explicitly filtered reference proxies, not proof that each changed file caused the bug.",
        "",
        "## Configuration and provenance",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
        "## Reproduction",
        "",
        f"Report only: `bash scripts/rave_fl_run.sh report --run-dir {shlex.quote(str(directory))}`",
        "",
        "Re-execute with a new output directory (replace the absolute output placeholder; live inference requires authorization):",
        "",
        "```bash",
        reproduce,
        "```",
        "",
        "Resume requires unchanged source/evidence/pool/config/model fingerprints and explicit inference authorization for new real requests. Saved complete arms are checksum-validated. No failed bug is replaced. No A100 performance claim is inferred from fixtures.",
        "",
        "## Figures",
        "",
    ]
    for f in figure_index:
        lines.extend(
            [
                f"![{f['title']}](../figures/{f['name']}.png)",
                "",
                f"{f['caption']} [SVG](../figures/{f['name']}.svg).",
                "",
            ]
        )
    (reports / "evaluation.md").write_text("\n".join(lines).rstrip() + "\n")
    write_json(
        directory / "artifact_index.json",
        {
            "files": [
                str(p.relative_to(directory))
                for p in sorted(directory.rglob("*"))
                if p.is_file() and "cache" not in p.parts and "sqlite" not in p.name
            ]
        },
    )
    return {
        "report": str(reports / "evaluation.md"),
        "figures": len(figure_index),
        "summary_rows": len(summary),
    }


def compare(runs, output, descriptive=False):
    if len(runs) < 2:
        raise CompatibilityError("Compare requires at least two saved runs")
    output = Path(output)
    if output.exists():
        raise CompatibilityError("Comparison output exists; choose a new directory")
    manifests = [json.loads((Path(r) / "manifest.json").read_text()) for r in runs]
    for directory, manifest in zip(map(Path, runs), manifests):
        if fingerprint({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
            raise CompatibilityError("Comparison manifest checksum mismatch")
        if (
            file_hash(directory / "results.json")
            != json.loads((directory / "audits/results.json").read_text())["results_sha256"]
        ):
            raise CompatibilityError("Comparison results checksum mismatch")
    rows = [json.loads((Path(r) / "results.json").read_text()) for r in runs]
    environments = [json.loads((Path(r) / "environment.json").read_text()) for r in runs]
    keys = [
        "dataset",
        "dataset_version",
        "model",
        "temperature",
        "context_window",
        "output_tokens",
        "representation",
        "granularity",
        "candidate_provider",
        "candidate_count",
        "admission",
        "seed",
        "fusions",
        "evaluation_mode",
        "tie",
        "rounding",
    ]
    differences = {
        k: [m["config"].get(k) for m in manifests]
        for k in keys
        if len({json.dumps(m["config"].get(k), sort_keys=True) for m in manifests}) > 1
    }
    if len({json.dumps(e["provider"], sort_keys=True) for e in environments}) > 1:
        differences["provider"] = "digest/version/settings differ"
    if len({fingerprint(m.get("implementation", {})) for m in manifests}) > 1:
        differences["implementation"] = "Collection implementation hashes differ"
    pools = {}
    for index, rr in enumerate(rows):
        for r in rr:
            pools.setdefault((r["dataset"], r["instance_id"]), {})[index] = (
                r.get("pool_hash"),
                r.get("input_hash"),
                r.get("offline_gold_hash"),
            )
    if any(len(set(v.values())) > 1 for v in pools.values() if len(v) > 1):
        differences["inputs"] = "source/evidence/pool fingerprints differ"
    if differences and not descriptive:
        raise CompatibilityError(
            "Incompatible paired comparison: "
            + ", ".join(differences)
            + ". Use --descriptive for labeled cross-setting summaries."
        )
    output.mkdir(parents=True)
    allrows = []
    paired = []
    for index, rr in enumerate(rows):
        for r in aggregate(rr):
            allrows.append({"run": str(runs[index]), **r})
    if not differences:
        left = {
            (r["dataset"], r["instance_id"], r["policy"], r["fusion"]): r
            for r in rows[0]
            if r["status"] == "completed"
        }
        for run_index in range(1, len(rows)):
            for r in rows[run_index]:
                key = (r["dataset"], r["instance_id"], r["policy"], r["fusion"])
                if key not in left or r["status"] != "completed":
                    continue
                for gran in ["file", "method"]:
                    a, b = left[key]["metrics"].get(gran), r["metrics"].get(gran)
                    if a is None or b is None:
                        continue
                    paired.append(
                        {
                            "reference_run": str(runs[0]),
                            "comparison_run": str(runs[run_index]),
                            "instance_id": r["instance_id"],
                            "policy": r["policy"],
                            "fusion": r["fusion"],
                            "granularity": gran,
                            "reference_budget": manifests[0]["config"]["budget"],
                            "comparison_budget": manifests[run_index]["config"]["budget"],
                            "reference_max_Full": manifests[0]["config"]["max_full"],
                            "comparison_max_Full": manifests[run_index]["config"]["max_full"],
                            "delta_MRR": b["MRR"] - a["MRR"],
                            "delta_tokens": r["accounted_tokens"] - left[key]["accounted_tokens"],
                        }
                    )
    csv_file(output / "aggregate.csv", allrows)
    csv_file(output / "paired.csv", paired)
    write_json(
        output / "comparison.json",
        {
            "mode": "descriptive_cross_setting" if differences else "paired_matched_bugs",
            "differences": differences,
            "paired": paired,
            "runs": [str(r) for r in runs],
        },
    )
    (output / "comparison.md").write_text(
        "# Saved-run comparison\n\n"
        + (
            "Descriptive cross-setting comparison; no paired causal claims."
            if differences
            else "Paired comparisons on compatible completed bugs; no confidence intervals from repeated states."
        )
        + "\n\n"
        + markdown(
            allrows,
            [
                "run",
                "policy",
                "fusion",
                "granularity",
                "denominator",
                "n",
                "Top-5",
                "MRR",
                "mean_tokens",
            ],
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = list(
        dict.fromkeys(
            (r["dataset"], r["language"], r["granularity"], r["fusion"])
            for r in allrows
            if r["denominator"].startswith("end")
        )
    )[:4]
    fig, axes = plt.subplots(
        max(1, len(settings)), 1, figsize=(11, max(4, 3.8 * len(settings))), squeeze=False
    )
    run_labels = {str(r): "Run " + str(i + 1) for i, r in enumerate(runs)}
    for ax, setting in zip(axes[:, 0], settings):
        for row in allrows:
            if (
                row["denominator"].startswith("end")
                and row["mean_tokens"] is not None
                and (row["dataset"], row["language"], row["granularity"], row["fusion"]) == setting
            ):
                ax.scatter(
                    row["mean_tokens"],
                    row["MRR"],
                    label=run_labels[row["run"]] + " / " + row["policy"],
                )
        ax.set(xlabel="Mean completed verification tokens", ylabel="MRR", title=" / ".join(setting))
        ax.legend(fontsize=6, ncol=2) if ax.collections else None
    caption = "Source: saved results.json and manifest.json for listed runs. Verification cost = provider input + output. Paired deltas use completed matching bugs/granularity only; cross-setting points are descriptive."
    caption += (
        " "
        + "; ".join(
            f"Run {i+1}: {m['config']['dataset']}, {m['config']['run_id']}, model {m['config']['model']}, admission {m['config']['admission']}"
            for i, m in enumerate(manifests)
        )
        + "."
    )
    if any(r.get("synthetic") for rr in rows for r in rr):
        caption += (
            " SYNTHETIC FIXTURE COMPARISON; no real model inference or benchmark accuracy claim."
        )
        with (output / "comparison.md").open("a") as handle:
            handle.write(
                "\nSYNTHETIC FIXTURE COMPARISON. No real model inference or benchmark accuracy claim.\n"
            )
    caption += " First four dataset/language/granularity/fusion panels; full data and run identities in aggregate.csv."
    fig.text(0.01, 0.01, textwrap.fill(caption, 150), fontsize=6)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(output / "comparison.png", dpi=200, metadata={"Description": caption})
    fig.savefig(output / "comparison.svg", metadata={"Description": caption})
    plt.close(fig)
    sensitivity = []
    for index, rr in enumerate(rows):
        for row in aggregate(rr):
            sensitivity.append(
                {
                    "run": str(runs[index]),
                    "configured_budget": manifests[index]["config"]["policy_budgets"].get(
                        row["policy"], manifests[index]["config"]["budget"]
                    ),
                    "max_Full": manifests[index]["config"]["max_full"],
                    **row,
                }
            )
    csv_file(output / "budget_call_sensitivity.csv", sensitivity)
    settings = list(
        dict.fromkeys(
            (r["dataset"], r["language"], r["granularity"], r["fusion"])
            for r in sensitivity
            if r["denominator"].startswith("end")
        )
    )[:6]
    if settings:
        fig, axes = plt.subplots(len(settings), 1, figsize=(10, 3.5 * len(settings)), squeeze=False)
        for ax, setting in zip(axes[:, 0], settings):
            vv = [
                r
                for r in sensitivity
                if (r["dataset"], r["language"], r["granularity"], r["fusion"]) == setting
                and r["denominator"].startswith("end")
            ]
            for policy, cap in sorted({(r["policy"], r["max_Full"]) for r in vv}):
                values = sorted(
                    [r for r in vv if r["policy"] == policy and r["max_Full"] == cap],
                    key=lambda r: r["configured_budget"],
                )
                ax.plot(
                    [r["configured_budget"] for r in values],
                    [r["MRR"] for r in values],
                    marker="o",
                    label=policy + f" / repair Full cap {cap}",
                )
            ax.set(
                title=" / ".join(setting),
                xlabel="Configured per-policy verification budget",
                ylabel="End-to-end operational MRR",
            )
            ax.legend(fontsize=6)
        fig.text(
            0.01,
            0.01,
            caption
            + " Different policy-budget overrides remain recorded in each manifest; missing executions receive zero operational credit.",
            fontsize=6,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(output / "budget_sensitivity.png", dpi=200, metadata={"Description": caption})
        fig.savefig(output / "budget_sensitivity.svg", metadata={"Description": caption})
        plt.close(fig)
    criteria = []
    for row in allrows:
        if row["denominator"].startswith("end") and row["mean_tokens"] is not None:
            same = [
                v
                for v in allrows
                if (v["run"], v["fusion"], v["granularity"], v.get("language"), v["denominator"])
                == (
                    row["run"],
                    row["fusion"],
                    row["granularity"],
                    row.get("language"),
                    row["denominator"],
                )
                and v["mean_tokens"] is not None
                and v["completion_rate"] == 1
            ]
            if row["completion_rate"] == 1 and not any(
                v["MRR"] > row["MRR"]
                or (v["MRR"] == row["MRR"] and v["mean_tokens"] < row["mean_tokens"])
                for v in same
            ):
                criteria.append(
                    {
                        **row,
                        "criterion": "Highest operational MRR, then lowest mean tokens; all compared policies complete; configured per-run budget",
                    }
                )
    csv_file(output / "best_completed_policies.csv", criteria)
    return {"output": str(output), "paired_rows": len(paired), "differences": differences}

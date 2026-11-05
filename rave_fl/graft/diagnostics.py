"""Regenerate structural diagnostics and historical baselines without inference."""
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
from rave_fl.eval.schema import OnlineInstance, CodeEntity, file_hash, fingerprint, safe_name
from rave_fl.pilot import write_json
from rave_fl.graft.archive import analyze
from rave_fl.graft.graph import build_graph
from rave_fl.graft.selection import relevance_select, ByteCounter
from rave_fl.graft.paths import preserve_paths
from rave_fl.graft.repair import missing_support, repair_support


def table(path, rows, columns):
    with Path(path).with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    Path(path).with_suffix(".md").write_text("\n".join(lines) + "\n")


def generate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new diagnostic output; preserve previous results")
    inputs = sorted(source.glob("*/instances/*/input.json"))
    if not inputs:
        raise ValueError("No archived prepared inputs found")
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    (output / "figures").mkdir()
    print("Re-evaluating frozen archive rankings; no provider calls", flush=True)
    archive_summary = analyze(source, output / "archive_evaluation")
    inventory = []
    structural, graphs, costs = [], [], []
    raw_costs = defaultdict(list)
    all_arms = {}
    for p in sorted(source.glob("*/results.json")):
        arms = json.loads(p.read_text())
        all_arms[p.parent.name] = arms
    cost_common = {
        bug
        for bug, arms in all_arms.items()
        if arms and all(a["status"] == "completed" for a in arms)
    }
    for bug in sorted(cost_common):
        for arm in all_arms[bug]:
            raw_costs[arm["policy"]].append(arm["accounted_tokens"])
    for policy, values in sorted(raw_costs.items()):
        q = (
            statistics.quantiles(values, n=100, method="inclusive")
            if len(values) > 1
            else values * 99
        )
        costs.append(
            {
                "policy": policy,
                "matched_completed_n": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p75": q[74],
                "p90": q[89],
                "p95": q[94],
                "maximum": max(values),
            }
        )
    for count, path in enumerate(inputs, 1):
        data = json.loads(path.read_text())
        instance = OnlineInstance.from_dict(data["online"])
        entities = [CodeEntity(**e) for e in data["entities"]]
        graph = build_graph(instance, entities)
        target = data["candidates"][0]["function_id"]
        # First prior candidate only: this bounded diagnostic is not a policy
        # evaluation or evidence about every possible target package.
        graphs.append(
            {
                "bug": instance.instance_id,
                "target": target,
                "graph_hash": graph.hash,
                "nodes": len(graph.nodes),
                "edges": dict(Counter(e.relation for e in graph.edges)),
                "parse_errors": sum(d["kind"] == "parse_error" for d in graph.diagnostics),
            }
        )
        write_json(output / "graphs" / (safe_name(instance.instance_id) + ".json"), graph.to_dict())
        for budget in [2048, 4096, 8192]:
            base = relevance_select(graph, target, budget)
            linked = preserve_paths(graph, base)
            variants = [
                ("relevance", base),
                ("relevance_paths", linked),
                ("relevance_repair", repair_support(graph, base)),
                ("relevance_paths_repair", repair_support(graph, linked)),
            ]
            for variant, package in variants:
                structural.append(
                    {
                        "bug": instance.instance_id,
                        "target": target,
                        "budget_units": budget,
                        "counter": package.counter,
                        "variant": variant,
                        "status": package.status,
                        "input_units": package.input_units,
                        "units_selected": len(package.ids),
                        "missing_support": len(missing_support(graph, package.ids)),
                        "path_coverage": package.diagnostics.get("path_coverage_after"),
                        "unreachable_terminals": len(
                            package.diagnostics.get("unreachable_terminals", [])
                        ),
                        "reason": package.reason,
                        "selection_hash": package.hash,
                    }
                )
        inventory.append({"path": str(path), "sha256": file_hash(path)})
        if count % 10 == 0:
            print(f"Completed {count}/{len(inputs)} archived structural inputs", flush=True)
    grouped = defaultdict(list)
    for row in structural:
        grouped[(row["budget_units"], row["variant"])].append(row)
    feasibility = [
        {
            "budget_units": b,
            "variant": v,
            "inputs": len(rows),
            "feasible": sum(r["status"] == "ready" for r in rows),
            "missing_support_links": sum(r["missing_support"] for r in rows),
            "packages_with_missing_support": sum(r["missing_support"] > 0 for r in rows),
            "infeasible": sum(r["status"] != "ready" for r in rows),
        }
        for (b, v), rows in sorted(grouped.items())
    ]
    # Detect accidental modifications of any directly consumed archive input.
    consumed = {r["path"]: r["sha256"] for r in inventory}
    frozen = json.loads((output / "archive_evaluation/summary.json").read_text())
    consumed.update({r["path"]: r["sha256"] for r in frozen["input_files"]})
    if any(file_hash(p) != checksum for p, checksum in consumed.items()):
        raise ValueError("Archive changed during diagnostics; stop and preserve artifacts")
    manifests = sorted(source.glob("*/manifest.json"))
    consumed.update({str(p): file_hash(p) for p in manifests})
    settings = [json.loads(p.read_text())["config"] for p in manifests]
    model_settings = sorted(
        {(s["model"], s["context_window"], s["output_tokens"], s["budget"]) for s in settings}
    )
    commits = subprocess.check_output(
        ["git", "log", "-9", "--format=%h %s"], text=True
    ).splitlines()
    from rave_fl.eval.runner import implementation_identity

    summary = {
        "schema": "graft-structural-diagnostics-v1",
        "new_model_calls": 0,
        "fresh_inference": False,
        "implementation_files": implementation_identity(),
        "inputs": len(inputs),
        "source": str(source),
        "source_condition": "Archived observed failure evidence and original buggy candidate sources",
        "target_scope": "First prior candidate per input; no gold selection",
        "counter": ByteCounter.name,
        "graph_nodes": sum(r["nodes"] for r in graphs),
        "parse_errors": sum(r["parse_errors"] for r in graphs),
        "edges": dict(sum((Counter(g["edges"]) for g in graphs), Counter())),
        "feasibility": feasibility,
        "historical_verification_costs": costs,
        "historical_model_settings": model_settings,
        "recommended_policy_budgets": [16000, 24000, 32000],
        "primary_policy_budget": 24000,
        "budget_note": "Development planning values informed by observed verification cost, not empirically optimized. Strict window reservations can reject an action despite expected low cost.",
        "historical_full_reference_budget": sorted({s["budget"] for s in settings}),
        "consumed_input_checksums": consumed,
        "implementation_commits": commits,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "structural.json", structural)
    write_json(output / "graph_summary.json", graphs)
    write_json(output / "historical_cost_samples.json", dict(raw_costs))
    write_json(
        output / "provenance.json",
        {
            "input_sha256": fingerprint(consumed),
            "consumed": consumed,
            "archive_files_unchanged": True,
            "new_model_calls": 0,
        },
    )
    table(
        output / "tables/feasibility",
        feasibility,
        [
            "budget_units",
            "variant",
            "inputs",
            "feasible",
            "infeasible",
            "missing_support_links",
            "packages_with_missing_support",
        ],
    )
    table(
        output / "tables/historical_costs",
        costs,
        ["policy", "matched_completed_n", "mean", "median", "p75", "p90", "p95", "maximum"],
    )
    matched = [r for r in archive_summary if r["cohort"] == "matched_completed_labeled"]
    table(
        output / "tables/frozen_ranking_metrics",
        matched,
        ["profile", "policy", "fusion", "n", "Top-1", "Top-5", "MRR"],
    )
    plot(output)
    progress(output)
    return {
        "output": str(output),
        "report": str(output / "RESEARCH_PROGRESS.md"),
        "inputs": len(inputs),
        "new_model_calls": 0,
        "archive_files_unchanged": True,
        "feasibility": feasibility,
    }


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    s = json.loads((output / "summary.json").read_text())
    model_label = (
        ", ".join(sorted({r[0] for r in s["historical_model_settings"]})) or "unrecorded model"
    )
    context_label = ", ".join(
        str(v) for v in sorted({r[1] for r in s["historical_model_settings"]})
    )
    budget_label = ", ".join(str(v) for v in sorted({r[3] for r in s["historical_model_settings"]}))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.dpi": 220,
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

    def save(fig, name, caption):
        fig.text(0.02, 0.01, caption, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        for ext in ["png", "svg"]:
            fig.savefig(output / "figures" / (name + "." + ext), facecolor="white")
        plt.close(fig)

    variants = ["relevance", "relevance_paths", "relevance_repair", "relevance_paths_repair"]
    labels = ["Relevance", "+ Paths", "+ Repair", "+ Paths + Repair"]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    x = np.arange(3)
    for n, (v, label) in enumerate(zip(variants, labels)):
        values = [r["feasible"] for r in s["feasibility"] if r["variant"] == v]
        ax.bar(x + (n - 1.5) * 0.19, values, width=0.19, label=label, color=colors[n])
    ax.set_xticks(x, ["2,048", "4,096", "8,192"])
    ax.set(
        xlabel="Complete-message input allowance (byte proxy units)",
        ylabel="Feasible packages",
        title=f"Structural feasibility: {s['inputs']} archived inputs, first prior target only",
        ylim=(0, s["inputs"] + 8),
    )
    ax.legend(ncol=2, loc="lower right")
    save(
        fig,
        "01_feasibility",
        f"Source: structural.json. Archived {model_label} candidate/evidence text; no inference.\nUTF-8 content bytes + 256 proxy; excludes generation. Feasibility is not localization quality.",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n, (v, label) in enumerate(zip(variants, labels)):
        ax.bar(
            n,
            next(
                r["missing_support_links"]
                for r in s["feasibility"]
                if r["variant"] == v and r["budget_units"] == 4096
            ),
            color=colors[n],
        )
    ax.set_xticks(range(4), labels)
    ax.set(
        ylabel="Missing selected-to-required links",
        title="Observable support at a fixed 4,096-unit allowance",
    )
    save(
        fig,
        "02_support_closure",
        "Source: structural.json, all archived first-prior packages, including infeasible inputs.\nREQUIRES / PART_OF only; no semantic-completeness or model-accuracy claim.",
    )
    rows = json.loads((output / "archive_evaluation/summary.json").read_text())["rows"]
    rows = [
        r
        for r in rows
        if r["cohort"] == "matched_completed_labeled" and r["fusion"] == "historical"
    ]
    policies = ["prior", "full", "fixed_l1", "v2_utility"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(4)
    for n, profile in enumerate(["legacy_v1", "java_strict_v2"]):
        ax.bar(
            x + (n - 0.5) * 0.34,
            [
                next(r["MRR"] for r in rows if r["policy"] == p and r["profile"] == profile)
                for p in policies
            ],
            width=0.34,
            label=profile,
            color=colors[n],
        )
    n = rows[0]["n"]
    ax.set_xticks(x, ["Prior", "Full", "Fixed L1", "V2 Utility"])
    ax.set(
        ylabel="Mean reciprocal rank",
        title=f"Frozen historical rankings: {n} jointly completed labeled bugs",
        ylim=(0, 1),
    )
    ax.legend()
    save(
        fig,
        "03_evaluation_profiles",
        f"Source: archive_evaluation/summary.json; recorded {model_label}.\nHistorical budget {budget_label}, context {context_label}. No new responses or held-out claim.",
    )
    samples = json.loads((output / "historical_cost_samples.json").read_text())
    policies = list(sorted(samples))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot([samples[p] for p in policies], showmeans=True)
    ax.set_xticks(range(1, len(policies) + 1), policies)
    for budget in s["recommended_policy_budgets"]:
        ax.axhline(budget, alpha=0.5, linestyle="--", label=f"{budget / 1000:g}K planning budget")
    ax.set(
        ylabel="Accounted verification input + output tokens",
        title=f"Historical cost distribution: {len(samples[policies[0]])} matched completed bugs",
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "04_historical_costs",
        f"Source: historical_cost_samples.json. Same archived {model_label} generation.\nExcludes retrieval/embeddings. Reused observations retain logical cost. Lines are future planning values.",
    )
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    phases = [
        ("0–1", "Evaluation + typed graph", "Strict labels; bounded provenance"),
        ("2–3", "Evidence selection", "BM25 control; optional dense topology"),
        ("4–5", "Paths + dependency repair", "Fixed-budget structural safeguards"),
        ("6–7", "Staged execution + protocol", "Strict accounting; fixture-validated"),
    ]
    for n, (phase, title, detail) in enumerate(phases):
        xx = 0.02 + n * 0.25
        ax.text(
            xx + 0.11,
            0.60,
            f"Phases {phase}\n{title}\n\n{detail}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=.9", facecolor=colors[n], alpha=0.18, edgecolor=colors[n]
            ),
            transform=ax.transAxes,
        )
        if n < 3:
            ax.annotate(
                "",
                xy=(xx + 0.25, 0.6),
                xytext=(xx + 0.22, 0.6),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->"},
            )
    ax.set_title("GRAFT implementation progression; real-model quality gate remains open")
    save(
        fig,
        "05_development",
        "Source: docs/graft_fl/phase00–phase07.md and saved Git subjects in summary.json.\nBoxes describe implemented modules. They do not imply real-model or held-out validation.",
    )


def progress(output):
    output = Path(output)
    s = json.loads((output / "summary.json").read_text())
    f = lambda variant: next(
        r for r in s["feasibility"] if r["budget_units"] == 4096 and r["variant"] == variant
    )
    baseline, repair, paths = f("relevance"), f("relevance_repair"), f("relevance_paths")
    text = f"""# GRAFT-FL Development Progress

## Scope and outcome

The eight implementation phases are available as opt-in modules. Historical RAVE-FL behavior and raw archives remain intact. This run performed no real model inference, benchmark builds or benchmark tests. The user will run small open-weight-model experiments later, including on an A100. Synthetic fixtures validate orchestration, not research accuracy. No new method has demonstrated superiority to Highest L1 Suspicion or the prior ranking.

## Architecture and completed phases

0. A versioned strict Java evaluator separates exact empty signatures, constructors and case from the historical permissive matcher. Both profiles remain reportable.
1. Typed evidence graphs retain source hashes, spans and extraction provenance. Static calls are MAY_CALL, lexical prerequisites are not reaching-definition proofs, and absent coverage is explicit.
2. BM25 relevance is the default evidence control. Complete system/user messages, wrappers and selected links are charged. Coherent statements are admitted whole.
3. A TopoCompress-inspired dense/sparse selector supports alignment, relevance-gated acceleration, semantic/sequential graph propagation and redundancy penalties. Typed edges and propagation ablations are explicit. A local-only BGE-M3 export adapter is supplied; real weights and real-encoder execution were unavailable in this run. Synthetic vectors exist only in tests.
4. Bounded typed paths augment evidence, then prune under the same input allowance. Unreachable or infeasible protected paths remain reported. Reachability is not causality.
5. Dependency closure restores observable prerequisites and evicts dependent bundles without increasing the allowance. This rule-based repair is not a trained dangling-reference detector and does not guarantee semantic completeness.
6. Sketch, graph and method_full are distinct observations. A bounded untrained router records current features, predicted cost, reservations, selections, rejections and paid outcomes. Legacy controllers are reused, with a small unchanged-identity extension hook in the universal backend.
7. A serial CLI, immutable manifests, exact-response journaling, resource preflight, resume checks, fixture creation and report regeneration support the subsequent experiment. No model/repository downloads occur implicitly.

## Measured offline diagnostics

Source: `{s['source']}`. There are {s['inputs']} prepared inputs, {s['graph_nodes']} extracted nodes and {s['parse_errors']} candidate AST parse errors. These graphs are bounded by the supplied candidate pool, not repository-wide runtime graphs. Source/frame ambiguity can still limit links.

At 4,096 byte-proxy units, relevance alone produced {baseline['feasible']} feasible packages and {baseline['missing_support_links']} missing observable support links in {baseline['packages_with_missing_support']} first-prior packages. Repair produced {repair['feasible']} feasible packages with {repair['missing_support_links']} missing links. Path preservation retained {paths['feasible']} feasible packages; protected evidence can exceed the allowance. Every diagnostic targets only the first prior candidate, so these rates must not be generalized to all candidate/action states. Missing links are structural counts, not model error labels.

![Structural feasibility](figures/01_feasibility.png)
![Observable closure](figures/02_support_closure.png)

## Historical quality and strongest controls

Both evaluation profiles and frozen historical/RRF re-rankings are in [the metric table](tables/frozen_ranking_metrics.md). The prior remains a mandatory control because historical verification did not establish a consistent gain over it. Strict signature matching changes some apparent successes. RRF here is post-hoc reranking of saved observations, not fresh inference and not proof that an RRF-driven controller would choose the same actions. The older single-bug Llama3.1 generation and the initial Qwen/A100 Time-25 generation are not controller ablations of this archived Qwen2.5 multi-bug generation.

![Evaluation profiles](figures/03_evaluation_profiles.png)

## Resource choice for the user's later experiment

[Cost distributions](tables/historical_costs.md) are calculated from matched completed archive arms, including unlabeled completed bugs for resource planning. The original policy budget was {s['historical_full_reference_budget']}; it was not the 8,192-token model context. The proposed primary policy budget is 24K verification input+output tokens, with 16K and 32K sensitivity controls. These values are development planning choices informed by observed costs, not experimentally optimized settings. Start with context 8,192 and output cap 160 per candidate, matching the archived generation where the model supports it; use a separately named 16,384-context condition when complete inputs cannot fit. Strict context-window reservations may reject requests even if their expected cost is small.

Keep model tag, immutable digest, endpoint, context, generation options, logical policy budget, per-bug physical cap and whole-run cap separately configurable. Model parameter count does not determine a universal context or budget. An A100's ability to serve a particular large/quantized model must be checked on its actual server; no hardware feasibility or latency claim is made here. Embedding compute is outside generation-token accounting and must be logged separately.

![Historical costs](figures/04_historical_costs.png)

## Selection decision and next gate

Use BM25 plus observable repair as an explicit candidate condition; keep path preservation and topology opt-in. Compare fixed graph actions before the adaptive router so evidence-selection changes can be isolated. Run Prior, Fixed L1, Highest L1 Suspicion, unchanged V2 Utility, L1-plus-graph and the experimental router under the same pool, failure evidence, generation settings, frozen fusion, logical budget and Full-call cap. Full-all is a budget-constrained control at equal budget; a separate 120K reference arm must not be presented as an equal-budget comparison. Add max-Full 0/1/2 and max-graph 0/1/2 ablations as resources permit.

New graph prompts require new model judgments. Historical Full responses cannot be replayed as responses to these changed contexts. Dense selection requires genuine exact-text embeddings. No utility model should be trained until multiple independent defects with reliable labels and complete action histories exist. Do not tune on these archived development bugs and call them held out.

## Implementation Change Log and Experimental Impact

| Phase | Files/modules changed | Previous behavior | New behavior | Measured improvement | Regression/limitation |
|---|---|---|---|---|---|
| 0 | graft/evaluation.py, archive.py | One historical matcher | Versioned strict matching and frozen comparisons | Reproduced legacy metrics; exposed signature sensitivity | Gold labels remain externally supplied |
| 1 | graft/graph.py | Candidate sketches only | Bounded typed graph with source provenance | Recorded graph coverage and parse status | No repository-wide runtime data flow |
| 2 | graft/selection.py | No graph evidence packing | BM25 and complete-message admission | Recorded feasible-package counts | Byte counter is a proxy; selected prerequisites can be absent |
| 3 | graft/topology.py, encode.py | Lexical selection only | Explicit dense/sparse propagation and ablations | Synthetic algorithm tests only | Real encoder/quality not measured |
| 4 | graft/paths.py, selection.py renderer | Independent fragment selection | Path augmentation and link serialization | Recorded reachable/infeasible path outcomes | Protected paths can reduce feasible coverage |
| 5 | graft/repair.py, graph.py | Observable support can dangle | Transitive closure and cascading eviction | Missing observable links reduced at unchanged allowance | Useful optional context may be evicted; semantic gaps remain |
| 6 | graft/controller.py, eval/backend.py identity hook | Legacy sketch/Full stages | Separate graph stage and shared strict accounting | Fixture-tested reuse, caps and paid failures | Untrained utility has no demonstrated ranking advantage |
| 7 | graft/runner.py, cli.py, reporting.py, diagnostics.py, configs/docs/tests | Separate experimental entry points | Immutable staged-run CLI and regenerable reports | Synthetic execution and offline archive diagnostics | Real-model and held-out gates still open |

Phase 7 also hardened immutable graph lookup, nonnegative path costs, encoder batch-length validation, required graph uncertainty, all-arm resume checks and failed-preparation denominators. These are correctness changes, not measured localization gains. Phase commits and focused validation records are in the phase documents; the final phase commit is identifiable by its Git subject. Existing universal tests are reported as existing, not newly added.

## Limitations and unresolved work

Real BGE-M3 execution, real graph-verdict parsing across model families, independent-defect localization quality, runtime coverage extraction and repository-wide call/data-flow grounding remain unvalidated or external. Java/Python method parsing is supported; the GRAFT wrapper rejects unsupported languages/granularities rather than claiming universal structural support. Test fixtures are synthetic and must never be calibration data. Existing nine empty labels and candidate-recall gaps need independent curation. Path/source coherence does not imply causal fault evidence. More evidence can displace useful details, and graph versus method_full is not an information hierarchy.

## Reproducibility and change log

Generated input checksums are retained in `provenance.json`; all directly consumed archive inputs were unchanged at completion. `summary.json` records model settings, cost distributions, graph counts, feasibility and Git subjects. The module changes, tests and decisions are documented separately in `docs/graft_fl/phase00.md` through `phase07.md`. Exact commands, future-run manifests and protocol are in `docs/GRAFT_FL.md` and `configs/graft_*`.

![Development progression](figures/05_development.png)

Regenerate into a new output directory:

```bash
python3 -m rave_fl.graft.cli diagnostics --archive results-100-bugs/results/d4j_multibug/d4j97_promptfix_20261027_122406 --output /tmp/graft-diagnostics-new
```

This entry point reads saved inputs and rankings, computes structural diagnostics, preserves consumed-source hashes, and generates all tables and PNG/SVG figures without contacting a model. It does not reproduce historical inference. Per-run real/synthetic reports use `python3 -m rave_fl.graft.cli report --run-dir RUN_DIR`.
"""
    (output / "RESEARCH_PROGRESS.md").write_text(text)

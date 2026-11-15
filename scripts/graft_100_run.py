"""Run one frozen five-bug Defects4J evaluation, or audit all 100 IDs.

All inference is explicit. A batch is never declared complete unless every
selected bug, policy, fusion and both graph treatments actually completed.
Missing data remain visible. Offline aggregation makes zero model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from scripts.graft_100_dataset import expected_batch, validate_cohort

POLICIES = (
    "l1_anchor", "l1_graph_focused",
    "l1_graph_test_aware_nostop", "l1_graph_test_aware",
)
FUSIONS = ("rrf", "suspicion_rank_blend")
SOURCE = Path("configs/graft_independent_qwen7b_control.json")


def write_new(path: Path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def folder(root: Path, batch: int) -> Path:
    return (root / "batches" / f"batch_{batch:02d}").resolve()


def batch_config(root: Path, cohort: dict, batch: int):
    expected = [r["bug"] for r in expected_batch(cohort, batch)]
    parent = folder(root, batch)
    prepared = parent / "prepared"
    record = json.loads((prepared / "manifest.json").read_text())
    if record.get("cohort_selection_sha256") != cohort["selection_sha256"]:
        raise ValueError("Frozen cohort changed after collection")
    cases = record["cases"]
    if [r["bug"] for r in cases] != expected:
        raise ValueError("No substitution or partial-batch model inference allowed")
    report = json.loads((prepared / "preparation_report.json").read_text())
    if report["status"] != "ready" or report["predeclared_bug_ids"] != expected:
        raise ValueError("All five official bugs must be authentically prepared")
    if any(
        c.get("evidence_kind") != "observed_failure"
        or not Path(c.get("checkout", "")).is_dir()
        or not Path(c.get("evidence_file", "")).is_file()
        or not c.get("gold", {}).get("files")
        or c.get("gold", {}).get("functions")
        for c in cases
    ):
        raise ValueError("Authentic buggy code, failing test, and file-proxy labels required")

    source = json.loads(SOURCE.read_text())
    benchmark = source["benchmark"]
    graft = source["graft"]
    benchmark.update({
        "dataset": "defects4j",
        "dataset_file": str((prepared / "manifest.json").resolve()),
        "bugs": expected,
        "candidate_provider": "bm25",
        "candidate_count": 24,
        "granularity": "method",
        "representation": "l1",
        "policies": ["fixed_l1"],
        "fusions": list(FUSIONS),
        "backend": "ollama",
        "model": "qwen2.5:7b",
        "batch_size": 1,
        "context_window": 8192,
        "output_tokens": 160,
        "temperature": 0,
        "admission": "strict",
        "require_observed_failure": True,
        "budget": 90000,
        "max_new_calls": 220,
        "physical_budget": 500000,
        "per_bug_physical_budget": 100000,
        "workers": 1,
        "evaluation_mode": "policy",
        "cache_dir": str((parent / "experiment" / "cache").resolve()),
        "cache_mode": "use",
        "run_id": f"graft_100_batch_{batch:02d}_{cohort['selection_sha256'][:12]}",
    })
    graft.update({
        "input_budget": 4096,
        "failure_digest_mode": "bounded_excerpt",
        "max_graph": 2,
        "max_full": 1,
    })
    source["graft_policies"] = list(POLICIES)
    from rave_fl.graft.runner import validate_config
    canonical = validate_config(source)
    experiment = parent / "experiment"
    config_file = experiment / "config.json"
    if not config_file.exists():
        write_new(config_file, canonical)
        write_new(experiment / "protocol.json", {
            "schema": "graft-100-batch-protocol-v1",
            "cohort_selection_sha256": cohort["selection_sha256"],
            "batch": batch,
            "split": expected_batch(cohort, batch)[0]["split"],
            "expected_bugs": expected,
            "full_official_test_plus_observed_failures": True,
            "graph_prompt": "bounded_exact_source_lines_v2",
            "gold_is_offline_modified_classes_file_proxy": True,
            "max_new_calls": 220,
            "physical_token_cap_batch": 500000,
            "physical_token_cap_bug": 100000,
            "no_outcome_driven_replacement": True,
            "no_automatic_model_calls_during_config": True,
        })
    elif json.loads(config_file.read_text()) != canonical:
        raise ValueError("Prior config changed; never reuse old run with new settings")
    return config_file


def check_preflight(config_path: Path):
    exp = config_path.parent
    dst = exp / "preflight.json"
    stderr = exp / "preflight.stderr.log"
    if dst.exists():
        info = json.loads(dst.read_text())
    else:
        with dst.open("w") as out, stderr.open("w") as err:
            run = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-m",
                 "rave_fl.graft.cli", "preflight", "--config", str(config_path)],
                stdout=out, stderr=err, check=False,
            )
        if run.returncode:
            raise RuntimeError(
                f"Preflight failed (exit {run.returncode}). See {stderr}"
            )
        info = json.loads(dst.read_text())
    b = json.loads(config_path.read_text())["benchmark"]
    statuses = info.get("candidate_preparations", [])
    if (
        info.get("ready") is not True
        or info.get("inference_performed") is not False
        or {r["instance_id"] for r in statuses} != set(b["bugs"])
        or len(statuses) != len(b["bugs"])
        or any(r["status"] != "ready" or r["candidates"] != 24 for r in statuses)
    ):
        raise RuntimeError("Candidate preflight incomplete, no Qwen calls: " + str(statuses))
    print("PREFLIGHT READY:", b["bugs"], flush=True)


def graph_readiness(config_path: Path):
    """Gold-free check of one feasible v2 graph package per policy and bug."""
    exp = config_path.parent
    dst = exp / "graph_readiness.json"
    if dst.exists():
        value = json.loads(dst.read_text())
    else:
        from rave_fl.eval.adapters import load_instances
        from rave_fl.eval.runner import prepare_one
        from rave_fl.graft.graph import build_graph
        from rave_fl.graft.focused import focus_select
        from rave_fl.graft.test_aware import test_aware_select
        from rave_fl.graft.selection import ByteCounter
        config = json.loads(config_path.read_text())
        opts = config["benchmark"]
        instances, _ = load_instances(opts)
        rows = []
        for instance in instances:
            prepared, entities, candidates, _, _, _ = prepare_one(instance, opts)
            graph = build_graph(prepared, entities)
            for selector, function in (
                ("focused", focus_select),
                ("test_aware", test_aware_select),
            ):
                options = []
                for c in candidates[:5]:
                    try:
                        picked = function(
                            graph, c.function_id, 4096, ByteCounter(),
                            "", failure_digest_mode="bounded_excerpt",
                        )
                        options.append({"target": c.function_id, "status": picked.status,
                                        "reason": picked.reason,
                                        "input_units": picked.input_units})
                    except (ValueError, RuntimeError) as exc:
                        options.append({"target": c.function_id, "status": "infeasible",
                                        "reason": str(exc)})
                rows.append({
                    "bug": instance.instance_id,
                    "selector": selector,
                    "top5": options,
                    "ready": any(o["status"] == "ready" for o in options),
                })
        value = {
            "schema": "graft-100-v2-graph-feasibility-v1",
            "new_model_calls": 0,
            "uses_gold": False,
            "all_ready": len(rows) == len(instances) * 2 and all(r["ready"] for r in rows),
            "rows": rows,
        }
        write_new(dst, value)
    if not value["all_ready"]:
        print("INFEASIBLE GRAPH PACKAGES:", [
            (r["bug"], r["selector"]) for r in value["rows"] if not r["ready"]
        ], flush=True)
        raise RuntimeError("Graph evidence not viable for all five preselected bugs; no inference")
    print("GRAPH READINESS PASSED without Qwen", flush=True)


def check_batch(audit: dict, expected: list[str]) -> dict:
    records = audit.get("records", [])
    indexed = {(r["bug"], r["fusion"], r["policy"]): r for r in records}
    expected_keys = {
        (bug, fusion, policy)
        for bug in expected for fusion in FUSIONS for policy in POLICIES
    }
    missing = [list(k) for k in sorted(expected_keys - set(indexed))]
    incomplete = [
        list(key) for key in sorted(expected_keys & set(indexed))
        if indexed[key].get("status") != "completed"
    ]
    graph_free = [
        list(key) for key in sorted(expected_keys & set(indexed))
        if key[2] in {"l1_graph_focused", "l1_graph_test_aware_nostop"}
        and indexed[key].get("graph_call_count", 0) == 0
    ]
    accounting = audit.get("physical_provider_accounting", {})
    safe = (
        accounting.get("synthetic") is False
        and accounting.get("pending") is False
        and accounting.get("usage_unknown") is False
        and accounting.get("simulated_provider_tokens") == 0
    )
    pairs = audit.get("paired_comparisons", [])
    expected_pairs = {(b, f) for b in expected for f in FUSIONS}
    complete_pairs = {
        (row.get("bug"), row.get("fusion"))
        for row in pairs if row.get("status") == "paired"
    }
    complete = (
        not missing and not incomplete and not graph_free
        and set(indexed) == expected_keys
        and expected_pairs == complete_pairs
        and audit.get("all_pairs_completed") is True
        and safe
    )
    return {
        "schema": "graft-100-five-bug-completion-v1",
        "expected_bugs": expected,
        "all_pairs_completed": not missing and not incomplete and expected_pairs == complete_pairs,
        "actual_graph_evidence_queried": not graph_free,
        "completed": complete,
        "missing": missing,
        "incomplete": incomplete,
        "graph_treatments_missing": graph_free,
        "physical_provider_accounting": accounting,
        "new_model_calls_for_audit": 0,
    }


def run_batch(root: Path, cohort: dict, batch: int, allow: bool):
    if not allow:
        raise ValueError("Explicit --allow-inference required")
    cfg = batch_config(root, cohort, batch)
    check_preflight(cfg)
    graph_readiness(cfg)
    expected = [r["bug"] for r in expected_batch(cohort, batch)]
    exp = cfg.parent
    run_path = exp / "run"
    has_manifest = (run_path / "manifest.json").exists()
    args = (
        [sys.executable, "-m", "rave_fl.graft.cli", "resume",
         "--run-dir", str(run_path), "--allow-inference"]
        if has_manifest else
        [sys.executable, "-m", "rave_fl.graft.cli", "run",
         "--config", str(cfg), "--output", str(run_path), "--allow-inference"]
    )
    log = exp / "model_run.log"
    mode = "a" if has_manifest else "w"
    with log.open(mode) as stream:
        outcome = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=False)
    print("MODEL RUN:", batch, "exit", outcome.returncode, "log:", log, flush=True)
    if (run_path / "results.json").is_file() and (run_path / "accounting.json").is_file():
        from scripts.audit_graft_phase3_multibug import inspect as audit_run
        audit = audit_run(run_path)
        audit_path = exp / "paired_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        gate = check_batch(audit, expected)
        (exp / "completion_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        print("COMPLETION GATE:", gate["completed"], flush=True)
        if not gate["completed"]:
            print("Missing or incomplete arms:", gate["missing"], gate["incomplete"],
                  "No graph:", gate["graph_treatments_missing"], flush=True)
    else:
        print("No complete results and accounting files; preserve run log", flush=True)
        return 3
    return 0 if outcome.returncode == 0 and gate["completed"] else 3


def aggregate(root: Path, cohort: dict):
    validate_cohort(cohort)
    rows = cohort["bugs"]
    outcomes = []
    spent = {"new_calls": 0, "physical_inference_tokens": 0}
    for batch in range(20):
        expected = [r["bug"] for r in expected_batch(cohort, batch)]
        base = folder(root, batch)
        audit_path = base / "experiment" / "paired_audit.json"
        gate_path = base / "experiment" / "completion_gate.json"
        preparation = base / "prepared" / "preparation_report.json"
        prep = json.loads(preparation.read_text()) if preparation.exists() else {}
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        accounts = audit.get("physical_provider_accounting", {})
        if accounts:
            spent["new_calls"] += accounts.get("new_calls", 0)
            spent["physical_inference_tokens"] += accounts.get("physical_inference_tokens", 0)
        status_by_bug = {
            i["bug"]: (
                "complete" if gate.get("completed")
                else "incomplete_evaluation" if i["bug"] in prep.get("prepared_bug_ids", []) and audit
                else "prepared_only" if i["bug"] in prep.get("prepared_bug_ids", [])
                else "unavailable_or_not_collected"
            )
            for i in expected_batch(cohort, batch)
        }
        metric_rows = {
            (r["bug"], r["fusion"], r["policy"]): r for r in audit.get("records", [])
        }
        for b in expected_batch(cohort, batch):
            for fusion in FUSIONS:
                metrics = {
                    policy: metric_rows.get((b["bug"], fusion, policy)) for policy in POLICIES
                }
                if all(v and v.get("status") == "completed" for v in metrics.values()):
                    label = (
                        "complete_with_graph"
                        if all(metrics[p]["graph_call_count"] > 0
                               for p in ("l1_graph_focused", "l1_graph_test_aware_nostop"))
                        else "complete_without_graph"
                    )
                else:
                    label = status_by_bug[b["bug"]]
                outcomes.append({
                    "bug": b["bug"], "project": b["project"], "split": b["split"],
                    "batch": batch, "fusion": fusion, "status": label,
                    "file_mrr": {
                        k: v.get("file_mrr") if v and v.get("status") == "completed" else None
                        for k, v in metrics.items()
                    },
                    "file_top1": {
                        k: v.get("file_hit_at_1") if v and v.get("status") == "completed" else None
                        for k, v in metrics.items()
                    },
                    "graph_calls": {
                        k: v.get("graph_call_count") if v else None
                        for k, v in metrics.items()
                    },
                })
    summaries = []
    for split in ("development", "evaluation"):
        for fusion in FUSIONS:
            eligible = [r for r in outcomes if r["split"] == split and r["fusion"] == fusion]
            usable = [r for r in eligible if r["status"] == "complete_with_graph"]
            means = {
                p: round(sum(r["file_mrr"][p] for r in usable) / len(usable), 5)
                if usable else None for p in POLICIES
            }
            diffs = {
                p: round(
                    sum(r["file_mrr"][p] - r["file_mrr"]["l1_anchor"] for r in usable)
                    / len(usable), 5
                ) if usable else None
                for p in POLICIES if p != "l1_anchor"
            }
            summaries.append({
                "split": split, "fusion": fusion,
                "predeclared_n": len(eligible),
                "fully_paired_graph_n": len(usable),
                "unavailable_or_incomplete_n": len(eligible) - len(usable),
                "mean_file_mrr_paired_only": means,
                "mean_file_mrr_difference_vs_l1_paired_only": diffs,
            })
    result = {
        "schema": "graft-100-aggregate-v1",
        "cohort_selection_sha256": cohort["selection_sha256"],
        "n_predeclared_unique_bugs": 100,
        "bug_count_by_project": dict(Counter(r["project"] for r in rows)),
        "development_predeclared": 20,
        "evaluation_predeclared": 80,
        "physical_new_usage_across_batches": spent,
        "gold_semantics": "official_d4j_modified_classes_file_proxy_not_method_truth",
        "no_imputation_for_missing_or_failed": True,
        "notes": (
            "The cohort and splits were selected before experimental outcomes. "
            "Only completed, actually graph-treated, paired bugs contribute to "
            "mean MRR. Incomplete/unavailable bugs are reported, never replaced."
        ),
        "summary": summaries,
        "instances": outcomes,
    }
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# GRAFT-FL, 100 predeclared Defects4J bugs",
        "",
        "Only real observed failure inputs and file-level modified-class proxy gold.",
        "No outcomes are fabricated. Incomplete and unavailable bugs are excluded",
        "from paired means and counted separately.",
        "",
        f"Selection SHA256: {cohort['selection_sha256']}",
        f"New physical Qwen calls across archived batches: {spent['new_calls']}",
        f"Physical provider input+output tokens across batches: {spent['physical_inference_tokens']}",
        "",
        "| Split | Fusion | Predeclared | Fully paired with real graph | L1 MRR | Focused MRR | Test-aware MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        m = r["mean_file_mrr_paired_only"]
        def disp(value):
            return "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {r['split']} | {r['fusion']} | {r['predeclared_n']} | "
            f"{r['fully_paired_graph_n']} | {disp(m['l1_anchor'])} | "
            f"{disp(m['l1_graph_focused'])} | {disp(m['l1_graph_test_aware'])} |"
        )
    lines.extend([
        "",
        "The evaluation split (80 frozen bugs) is not a guarantee of 80 completed",
        "runs. Top-1, method-level ground truth and latency require separate",
        "grounded reporting, not interpretation of file-level MRR.",
    ])
    (output / "aggregate.md").write_text("\n".join(lines) + "\n")
    print("AGGREGATE:", output / "aggregate.md")
    print("Paired, actually graph-treated evaluation bugs:",
          next(x["fully_paired_graph_n"] for x in summaries
               if x["split"] == "evaluation" and x["fusion"] == "rrf"), "/ 80")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for action in ("config", "preflight", "run", "aggregate"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--root", type=Path, required=True)
        if action != "aggregate":
            cmd.add_argument("--batch", required=True, type=int)
        if action == "run":
            cmd.add_argument("--allow-inference", action="store_true")
    args = p.parse_args(argv)
    cohort = json.loads((args.root / "cohort.json").read_text())
    validate_cohort(cohort)
    if args.command == "aggregate":
        aggregate(args.root, cohort)
        return 0
    config = batch_config(args.root, cohort, args.batch)
    if args.command == "config":
        print("Config:", config)
        return 0
    if args.command == "preflight":
        check_preflight(config)
        graph_readiness(config)
        return 0
    return run_batch(args.root, cohort, args.batch, args.allow_inference)


if __name__ == "__main__":
    raise SystemExit(main())

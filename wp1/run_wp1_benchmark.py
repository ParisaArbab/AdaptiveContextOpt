#!/usr/bin/env python3
"""Run the AdaptiveContextOpt Defects4J RAW vs LeanCTX benchmark.

Pipeline per bug and model:
  Defects4J checkout -> Graphify -> one test capture
  -> RAW / real LeanCTX
  -> Agent4SR
  -> merge 5 SBIR + 5 Ochiai + 5 BoostN + 5 Agent4SR
  -> Agent4LR
  -> Top-1 / Top-3 / Top-5 / MAP / MRR
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

from defects4j_harness import checkout_bug, run_tests_once
from evaluation import aggregate, evaluate_top5
from flexfl_data import (
    TRADITIONAL_METHODS,
    flexfl_root,
    merge_top20,
    read_bug_report,
    read_ground_truth,
    read_trigger_test,
)
from flexfl_pipeline import run_agent4lr, run_agent4sr
from graphify_structure import GraphifyIndex
from leanctx_compressor import compress_captured_shell_output
from llm_backends import ChatBackend


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flexfl-repo", type=Path, default=Path("references/FlexFL_OriginalReplication"))
    ap.add_argument("--work-root", type=Path, default=Path("data/defects4j"))
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--bugs", default="Time-25", help="Comma-separated FlexFL/Defects4J IDs")
    ap.add_argument("--bug-list", type=Path, default=None, help="One bug ID per line")
    ap.add_argument("--all-flexfl-bugs", action="store_true", help="Run all bugs having SBIR, Ochiai and BoostN files")
    ap.add_argument("--backend", choices=["ollama", "openai", "openai-compatible", "vllm", "anthropic"], default="ollama")
    ap.add_argument("--models", default="llama3:8b,qwen2:7b,mistral:7b", help="Comma-separated model names")
    ap.add_argument("--base-url", default=None, help="Provider URL, for example Ollama or vLLM endpoint")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-agent-steps", type=int, default=12)
    ap.add_argument("--test-timeout", type=int, default=1800)
    ap.add_argument("--force-graphify", action="store_true")
    ap.add_argument("--fresh-checkout", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def resolve_bugs(args: argparse.Namespace) -> list[str]:
    if args.all_flexfl_bugs:
        root = flexfl_root(args.flexfl_repo)
        sets: list[set[str]] = []
        for method in TRADITIONAL_METHODS:
            folder = root / "data" / "FL_results" / method / "Defects4J"
            ids = {p.name.removesuffix("_method-susps.csv") for p in folder.glob("*_method-susps.csv")}
            sets.append(ids)
        bugs = sorted(set.intersection(*sets)) if sets else []
    elif args.bug_list:
        bugs = [
            line.strip()
            for line in args.bug_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        bugs = [x.strip() for x in args.bugs.split(",") if x.strip()]
    return list(dict.fromkeys(bugs))


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    if not args.flexfl_repo.exists():
        raise SystemExit(
            f"FlexFL reference repo not found: {args.flexfl_repo}\n"
            "Run scripts/clone_reference_repos.sh first, or pass --flexfl-repo."
        )

    bugs = resolve_bugs(args)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not bugs or not models:
        raise SystemExit("At least one bug and one model are required")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        run_dir / "config.json",
        {
            "bugs": bugs,
            "backend": args.backend,
            "models": models,
            "flexfl_repo": str(args.flexfl_repo.resolve()),
            "work_root": str(args.work_root.resolve()),
            "design": "same Defects4J test capture -> RAW vs real LeanCTX ctx_compare shell pipeline",
        },
    )

    rows: list[dict] = []
    errors: list[dict] = []

    for bug in bugs:
        bug_dir = run_dir / bug
        try:
            print(f"\n=== {bug}: checkout ===", flush=True)
            repo = checkout_bug(bug, args.work_root, reuse=not args.fresh_checkout)

            print(f"=== {bug}: Graphify ===", flush=True)
            graph = GraphifyIndex.build(repo, force=args.force_graphify)
            graph.save_compact(bug_dir / "graphify_structure.json")

            print(f"=== {bug}: defects4j test, captured ONCE ===", flush=True)
            test_capture = run_tests_once(repo, timeout=args.test_timeout)
            raw_output = test_capture.output
            (bug_dir / "raw_output.txt").write_text(raw_output)
            dump_json(bug_dir / "test_capture.json", test_capture.to_dict())

            print(f"=== {bug}: LeanCTX production shell compression ===", flush=True)
            lean = compress_captured_shell_output(raw_output, repo, command="defects4j test")
            (bug_dir / "leanctx_output.txt").write_text(lean.text)
            (bug_dir / "leanctx_preview.txt").write_text(lean.report)
            dump_json(bug_dir / "compression.json", lean.to_dict())

            bug_report = read_bug_report(args.flexfl_repo, bug)
            trigger_test = read_trigger_test(args.flexfl_repo, bug)
            truths = read_ground_truth(args.flexfl_repo, bug)
            dump_json(
                bug_dir / "reference_inputs.json",
                {"ground_truth": truths, "has_bug_report": bool(bug_report), "has_trigger_test": bool(trigger_test)},
            )

            arms = {"raw": raw_output, "leanctx": lean.text}
            for model in models:
                backend = ChatBackend(
                    provider=args.backend,
                    model=model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                )
                for condition, runtime_output in arms.items():
                    print(f"=== {bug}: {model}: {condition}: Agent4SR ===", flush=True)
                    sr = run_agent4sr(
                        backend,
                        graph,
                        bug,
                        bug_report,
                        trigger_test,
                        runtime_output,
                        max_steps=args.max_agent_steps,
                    )
                    candidates, candidate_parts = merge_top20(args.flexfl_repo, bug, sr.predictions)

                    print(f"=== {bug}: {model}: {condition}: Agent4LR ===", flush=True)
                    lr = run_agent4lr(
                        backend,
                        graph,
                        bug,
                        bug_report,
                        trigger_test,
                        runtime_output,
                        candidates,
                        max_steps=args.max_agent_steps,
                    )
                    metrics = evaluate_top5(lr.predictions, truths)
                    model_dir = bug_dir / safe_name(model) / condition
                    dump_json(model_dir / "agent4sr.json", sr.to_dict())
                    dump_json(
                        model_dir / "merged_candidates.json",
                        {"parts": candidate_parts, "top20": candidates},
                    )
                    dump_json(model_dir / "agent4lr.json", lr.to_dict())
                    dump_json(model_dir / "evaluation.json", {"ground_truth": truths, **metrics})

                    row = {
                        "bug": bug,
                        "model": model,
                        "condition": condition,
                        "sr_count": len(sr.predictions),
                        "candidate_count": len(candidates),
                        "lr_count": len(lr.predictions),
                        **metrics,
                        "raw_bytes": len(raw_output.encode()),
                        "context_bytes": len(runtime_output.encode()),
                        "compression_saved_percent": lean.saved_percent if condition == "leanctx" else 0.0,
                    }
                    rows.append(row)
                    print(
                        f"    Top1={metrics['top1']} Top3={metrics['top3']} Top5={metrics['top5']} "
                        f"rank={metrics['first_relevant_rank']}",
                        flush=True,
                    )
        except Exception as exc:
            error = {"bug": bug, "error": str(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            dump_json(bug_dir / "ERROR.json", error)
            print(f"ERROR {bug}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                dump_json(run_dir / "errors.json", errors)
                return 2

    write_summary(run_dir, rows, errors)
    print(f"\nDone. Results: {run_dir}", flush=True)
    return 0 if not errors else 2


def write_summary(run_dir: Path, rows: list[dict], errors: list[dict]) -> None:
    if rows:
        fields = list(rows[0].keys())
        with (run_dir / "per_run.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[str, dict] = {}
    for model in sorted({r["model"] for r in rows}):
        grouped[model] = {}
        for condition in ("raw", "leanctx"):
            subset = [r for r in rows if r["model"] == model and r["condition"] == condition]
            grouped[model][condition] = aggregate(subset)

    compression_tax: list[dict] = []
    keyed = {(r["bug"], r["model"], r["condition"]): r for r in rows}
    for bug, model in sorted({(r["bug"], r["model"]) for r in rows}):
        raw = keyed.get((bug, model, "raw"))
        lean = keyed.get((bug, model, "leanctx"))
        if raw and lean and raw["top5"] and not lean["top5"]:
            compression_tax.append(
                {
                    "bug": bug,
                    "model": model,
                    "raw_rank": raw["first_relevant_rank"],
                    "leanctx_rank": lean["first_relevant_rank"],
                }
            )

    dump_json(
        run_dir / "summary.json",
        {
            "aggregate_by_model": grouped,
            "compression_tax_top5": compression_tax,
            "errors": errors,
            "row_count": len(rows),
        },
    )
    dump_json(run_dir / "errors.json", errors)


if __name__ == "__main__":
    raise SystemExit(main())

"""
run_wp1_benchmark.py — WP1 orchestrator

Revised pipeline order (per the updated architecture):

    Graphify (structure, once per repo)
        -> compressor (Control/raw, rtk, lean-ctx — three conditions)
        -> feedback loop (agent double-checks, <=2 rounds, compressed
           conditions only — raw has nothing to reveal)
        -> evaluation framework (compression_tax_analyzer.py)

Every condition sees the SAME Graphify structure map and the SAME raw
pytest capture, so the only variable between conditions is the compressor
itself — same apples-to-apples discipline as the original rtk-only run
(where a missing --include='*.py' on one side was caught and fixed for
exactly this reason).

Usage:
    python run_wp1_benchmark.py --instances data/instances.json \\
        --repos-dir data/repos --local-fallback --backend heuristic \\
        --out results/wp1_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import agent_localizer
import docker_harness
import graphify_structure
import leanctx_compressor
import rtk_compressor
from compression_tax_analyzer import InstanceOutcome, classify_taxonomy, score_file_level


def get_chat_fn(backend: str):
    if backend == "claude":
        return agent_localizer.make_anthropic_chat_fn(), "claude"
    if backend == "gpt":
        return agent_localizer.make_openai_compatible_chat_fn("OPENAI_API_KEY"), "gpt"
    if backend == "deepseek":
        return (
            agent_localizer.make_openai_compatible_chat_fn(
                "DEEPSEEK_API_KEY", base_url="https://api.deepseek.com", model="deepseek-chat"
            ),
            "deepseek",
        )
    if backend == "qwen":
        return (
            agent_localizer.make_openai_compatible_chat_fn(
                "DASHSCOPE_API_KEY",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen-max",
            ),
            "qwen",
        )
    return None, "heuristic"


def run_one_condition(
    instance_id: str,
    condition: str,  # "raw" | "rtk" | "leanctx"
    raw_output: str,
    structure_map: dict,
    problem_statement: str,
    ground_truth_files: list[str],
    backend: str,
    chat_fn,
) -> InstanceOutcome:
    compressor_mode = "n/a"
    if condition == "raw":
        agent_text = raw_output
    elif condition == "rtk":
        cr = rtk_compressor.compress(raw_output)
        agent_text = cr.text
        compressor_mode = cr.mode
    elif condition == "leanctx":
        cr = leanctx_compressor.compress(raw_output)
        agent_text = cr.text
        compressor_mode = cr.mode
    else:
        raise ValueError(condition)

    structure_text = graphify_structure.format_for_agent(structure_map)

    if backend == "heuristic":
        result = agent_localizer.HeuristicBackend().localize(
            agent_text, structure_map, problem_statement
        )
        result.instance_id = instance_id
        rounds_used = 0
    else:
        result = agent_localizer.localize_with_llm(
            instance_id=instance_id,
            tool_output=agent_text,
            structure_map_text=structure_text,
            problem_statement=problem_statement,
            chat_fn=chat_fn,
            backend_name=backend,
            use_feedback_loop=(condition != "raw"),
            raw_tool_output=raw_output if condition != "raw" else None,
        )
        rounds_used = result.feedback_rounds_used

    file_ok = score_file_level(result.predicted_files, ground_truth_files)
    tags = classify_taxonomy(raw_output, agent_text) if condition != "raw" else []

    return InstanceOutcome(
        instance_id=instance_id,
        condition=condition,
        compressor_mode=compressor_mode,
        predicted_files=result.predicted_files,
        ground_truth_files=ground_truth_files,
        file_level_correct=file_ok,
        taxonomy_tags=tags,
        provisional=(compressor_mode == "reference"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, default="data/instances.json")
    ap.add_argument("--repos-dir", type=str, default="data/repos")
    ap.add_argument("--local-fallback", action="store_true",
                     help="use local pytest instead of the SWE-bench Docker images")
    ap.add_argument("--backend", type=str, default="heuristic",
                     choices=["heuristic", "claude", "gpt", "deepseek", "qwen"])
    ap.add_argument("--out", type=str, default="results/wp1_results.json")
    args = ap.parse_args()

    instances = json.loads(Path(args.instances).read_text())
    chat_fn, backend_name = get_chat_fn(args.backend)

    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    all_outcomes = []
    for inst in instances:
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        print(f"=== {instance_id} ===")

        repo_local_path = repos_dir / instance_id.replace("/", "_")
        # NOTE: cloning/checkout for local-fallback happens inside
        # docker_harness.run_local_fallback; graphify then runs against
        # that same checkout so the structure map matches the exact
        # commit the test output came from.
        run_result = docker_harness.run_local_fallback(
            instance_id=instance_id,
            repo=repo,
            base_commit=inst["base_commit"],
            test_patch="",  # WP1 baseline audit does not need the gold test_patch applied
            workdir=repo_local_path,
        ) if args.local_fallback else docker_harness.run_in_docker(instance_id)

        try:
            structure_map = graphify_structure.build_structure_map(repo_local_path)
        except Exception as e:
            print(f"  graphify failed ({e}); skipping instance")
            continue

        raw_output = run_result.stdout + "\n" + run_result.stderr

        for condition in ("raw", "rtk", "leanctx"):
            outcome = run_one_condition(
                instance_id=instance_id,
                condition=condition,
                raw_output=raw_output,
                structure_map=structure_map,
                problem_statement=inst["problem_statement"],
                ground_truth_files=inst["files"],
                backend=backend_name,
                chat_fn=chat_fn,
            )
            all_outcomes.append(outcome)
            print(f"  [{condition:8s}] mode={outcome.compressor_mode:10s} "
                  f"file_ok={outcome.file_level_correct} tags={outcome.taxonomy_tags}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([o.__dict__ for o in all_outcomes], indent=2))
    print(f"\nwrote {len(all_outcomes)} outcomes -> {out_path}")


if __name__ == "__main__":
    main()

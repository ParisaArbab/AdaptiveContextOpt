"""
run_wp1_benchmark.py — WP1 orchestrator

Pipeline (rtk removed; lean-ctx is now the sole compression condition):

    Graphify (structure map + real call graph, once per repo)
        -> compressor: raw (control) | lean-ctx (smart)
        -> feedback loop (agent double-checks fidelity, <=2 rounds, lean-ctx only)
        -> FlexFL (Agent4SR space reduction -> Agent4LR refinement)
           + GraphLocator (real call-graph expansion from symptom vertices)
        -> evaluation framework (compression_tax_analyzer.py)

Every condition sees the SAME Graphify structure map and call graph, and
runs through the SAME localization pipeline (agent_localizer.py) — the only
variable between conditions is what the localizer's input tool_output looks
like (raw vs lean-ctx-compressed).

Usage:
    python run_wp1_benchmark.py --instances data/instances.json \\
        --repos-dir data/repos --local-fallback --backend heuristic \\
        --out results/wp1_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import agent_localizer
import docker_harness
import graphify_structure
import leanctx_compressor
from compression_tax_analyzer import InstanceOutcome, classify_taxonomy, score_file_level

CONDITIONS = ("raw", "leanctx")


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
    condition: str,  # "raw" | "leanctx"
    raw_output: str,
    structure_map: dict,
    call_graph: dict,
    repo_root: Path,
    problem_statement: str,
    ground_truth_files: list[str],
    backend: str,
    chat_fn,
) -> InstanceOutcome:
    compressor_mode = "n/a"
    if condition == "raw":
        agent_text = raw_output
    elif condition == "leanctx":
        cr = leanctx_compressor.compress(raw_output)
        agent_text = cr.text
        compressor_mode = cr.mode
    else:
        raise ValueError(condition)

    if backend == "heuristic":
        result = agent_localizer.HeuristicBackend().localize(
            agent_text, structure_map, problem_statement,
            call_graph=call_graph, repo_root=repo_root,
        )
        result.instance_id = instance_id
        rounds_used = 0
    else:
        result = agent_localizer.localize_with_llm(
            instance_id=instance_id,
            tool_output=agent_text,
            structure_map=structure_map,
            problem_statement=problem_statement,
            chat_fn=chat_fn,
            backend_name=backend,
            call_graph=call_graph,
            repo_root=repo_root,
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
        run_result = docker_harness.run_local_fallback(
            instance_id=instance_id,
            repo=repo,
            base_commit=inst["base_commit"],
            test_patch="",
            workdir=repo_local_path,
        ) if args.local_fallback else docker_harness.run_in_docker(instance_id)

        try:
            structure_map = graphify_structure.build_structure_map(repo_local_path)
            call_graph = graphify_structure.build_call_graph(repo_local_path)
        except Exception as e:
            print(f"  graphify failed ({e}); skipping instance")
            continue

        raw_output = run_result.stdout + "\n" + run_result.stderr

        for condition in CONDITIONS:
            outcome = run_one_condition(
                instance_id=instance_id,
                condition=condition,
                raw_output=raw_output,
                structure_map=structure_map,
                call_graph=call_graph,
                repo_root=repo_local_path,
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

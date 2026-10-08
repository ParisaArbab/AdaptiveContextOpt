"""Run a SWE-bench fault-localization experiment with program slicing context.

This runner creates a dependency-guided backward slice from the failing Python
test and can optionally send that slice to Agent4SR. It is intended for
comparison with RAW and LeanCTX context selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wp1.graphify_structure import GraphifyIndex
from wp1.llm_backends import ChatBackend
from wp1.program_slicing import expand_production_dependencies, slice_test_file
from wp1.run_swebench_agent4sr_pair import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a failure-oriented static backward slice for one SWE-bench instance."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--criterion-line", type=int, default=None)
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--ollama-num-predict", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = Path.home() / "AdaptiveContextOpt"
    work = root / "data/swebench_workspaces" / args.instance
    repo = work / "repo"
    outputs = work / "outputs"

    metadata_path = work / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    fail_to_pass = list(metadata.get("FAIL_TO_PASS") or [])
    if not fail_to_pass:
        raise SystemExit("metadata.json has no FAIL_TO_PASS test")

    test_id = fail_to_pass[0]
    result = slice_test_file(
        repo,
        test_id,
        criterion_line=args.criterion_line,
    )

    graph_path = repo / "graphify-out/graph.json"
    graph = None
    if graph_path.exists():
        graph = GraphifyIndex.from_json(repo, graph_path)
        expand_production_dependencies(repo, graph, result)

    out_dir = args.output_dir or outputs / "slicing"
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_json = out_dir / "slice.json"
    slice_txt = out_dir / "slice_context.txt"
    slice_json.write_text(json.dumps(result.to_dict(), indent=2))
    production_text = "\n\n".join(
        f"## {dep.entity}\n"
        f"# relation: {dep.relation}\n"
        f"{dep.source}"
        for dep in result.production_dependencies
    )

    slice_txt.write_text(
        "# Dependency-guided backward slice\n"
        f"# instance: {args.instance}\n"
        f"# failing test: {test_id}\n"
        f"# criterion line: {result.criterion_line}\n"
        f"# criterion: {result.criterion_text}\n\n"
        + result.text
        + "\n\n# Production dependencies\n"
        + (production_text or "(none)")
        + "\n"
    )

    print("=== PROGRAM SLICE ===")
    print(f"Instance: {args.instance}")
    print(f"Failing test: {test_id}")
    print(f"Criterion line: {result.criterion_line}")
    print(f"Selected lines: {len(result.selected_lines)}")
    print(f"Production dependencies: {len(result.production_dependencies)}")
    print()
    print(result.text)
    if result.production_dependencies:
        print()
        print("=== PRODUCTION DEPENDENCIES ===")
        for dep in result.production_dependencies:
            print(f"- {dep.entity} [{dep.relation}]")
    print()
    print(f"Saved: {slice_json}")
    print(f"Saved: {slice_txt}")

    if not args.run_agent:
        return

    if graph is None:
        raise SystemExit(f"Missing Graphify graph: {graph_path}")
    backend = ChatBackend(
        provider="ollama",
        model=args.model,
        timeout=1800,
        ollama_num_predict=args.ollama_num_predict,
    )

    problem = metadata.get("problem_statement", "")
    failing = ", ".join(fail_to_pass)

    # The slice and only dependency-derived production evidence are passed as
    # context. Gold information is never used to build this context.
    production_context = "\n\n".join(
        f"PRODUCTION DEPENDENCY: {dep.entity}\n"
        f"RELATION: {dep.relation}\n"
        f"{dep.source}"
        for dep in result.production_dependencies
    )

    slicing_context = (
        "DEPENDENCY-GUIDED STATIC BACKWARD SLICE OF THE FAILING TEST:\n"
        + result.text
        + "\n\nDEPENDENCY-GUIDED PRODUCTION CONTEXT:\n"
        + (production_context or "(none)")
    )

    agent = run_agent(
        backend,
        graph,
        problem,
        failing,
        slicing_context,
        max_steps=args.max_agent_steps,
    )

    agent_path = out_dir / "agent4sr_slicing.json"
    agent_path.write_text(
        json.dumps(
            {
                "instance_id": args.instance,
                "model": args.model,
                "context_mode": "static_backward_cross_scope_with_graphify_dependencies",
                "gold_used_by_agent": False,
                "slice": result.to_dict(),
                "agent": agent,
            },
            indent=2,
        )
    )

    print()
    print("=== Agent4SR WITH SLICING CONTEXT ===")
    for index, candidate in enumerate(agent.get("predictions") or [], 1):
        print(f"{index}. {candidate}")
    print(f"Saved: {agent_path}")


if __name__ == "__main__":
    main()

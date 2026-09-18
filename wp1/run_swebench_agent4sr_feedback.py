from __future__ import annotations

import argparse
import json
from pathlib import Path

from wp1.adaptive_feedback import (
    evaluate_context_sufficiency,
    next_density,
    normalize_density_schedule,
)
from wp1.graphify_structure import GraphifyIndex
from wp1.leanctx_density import compress_to_density, resolve_density_helper
from wp1.llm_backends import ChatBackend
from wp1.run_swebench_agent4sr_pair import run_agent


def parse_schedule(value: str) -> list[float]:
    return normalize_density_schedule(
        [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Agent4SR with gold-free adaptive LeanCTX density feedback."
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--initial-density", type=float, default=0.30)
    parser.add_argument(
        "--density-schedule",
        default="0.30,0.50,0.70,1.00",
        help="Increasing target densities used when feedback asks for more context.",
    )
    parser.add_argument(
        "--max-feedback-rounds",
        type=int,
        default=3,
        help="Maximum retries after the initial localization run.",
    )
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--density-helper", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.max_feedback_rounds < 0:
        raise SystemExit("--max-feedback-rounds must be >= 0")

    schedule = parse_schedule(args.density_schedule)
    initial = round(args.initial_density, 6)
    if initial not in schedule:
        schedule = normalize_density_schedule([*schedule, initial])

    root = Path.home() / "AdaptiveContextOpt"
    work = root / "data/swebench_workspaces" / args.instance
    outputs = work / "outputs"
    out_dir = args.output_dir or outputs / "adaptive_feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    context_dir = out_dir / "contexts"
    context_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((work / "metadata.json").read_text())
    problem = metadata.get("problem_statement", "")
    failing = ", ".join(metadata.get("FAIL_TO_PASS", []))
    raw_output = (outputs / "raw_test_output.txt").read_text(errors="replace")

    graph = GraphifyIndex.from_json(
        work / "repo",
        work / "repo/graphify-out/graph.json",
    )
    backend = ChatBackend(
        provider="ollama",
        model=args.model,
        timeout=1800,
    )
    helper = resolve_density_helper(args.density_helper)

    context_cache: dict[float, tuple[str, dict]] = {}

    def context_for_density(density: float) -> tuple[str, dict]:
        density = round(float(density), 6)
        if density in context_cache:
            return context_cache[density]

        if density >= 1.0 - 1e-9:
            info = {
                "target_density": 1.0,
                "original_tokens": None,
                "compressed_tokens": None,
                "saved_percent": 0.0,
                "helper": None,
                "mode": "raw",
            }
            text = raw_output
        else:
            result = compress_to_density(raw_output, density, helper=helper)
            text = result.text
            info = {**result.to_dict(), "mode": "leanctx_density_research"}

        name = f"density_{round(density * 100):03d}.txt"
        (context_dir / name).write_text(text)
        context_cache[density] = (text, info)
        return text, info

    rounds: list[dict] = []
    density = initial
    feedback_rounds_used = 0
    stop_reason = ""

    while True:
        runtime_output, compression = context_for_density(density)
        run_index = len(rounds) + 1
        print(
            f"\n=== Adaptive Agent4SR run {run_index}: target density={density:.2f} ===",
            flush=True,
        )
        agent = run_agent(
            backend,
            graph,
            problem,
            failing,
            runtime_output,
            max_steps=args.max_agent_steps,
        )

        upcoming = next_density(density, schedule)
        decision = None
        if feedback_rounds_used >= args.max_feedback_rounds:
            stop_reason = "max_feedback_rounds"
        elif upcoming is None:
            stop_reason = "no_higher_density"
        else:
            decision = evaluate_context_sufficiency(
                backend,
                problem=problem,
                failing_test=failing,
                runtime_output=runtime_output,
                agent_result=agent,
                current_density=density,
                next_density_value=upcoming,
                feedback_round=feedback_rounds_used + 1,
                max_feedback_rounds=args.max_feedback_rounds,
            )
            if not decision.expand:
                stop_reason = "feedback_stop"

        record = {
            "run_index": run_index,
            "target_density": density,
            "compression": compression,
            "predictions": agent.get("predictions", []),
            "agent": agent,
            "feedback": decision.to_dict() if decision else None,
        }
        rounds.append(record)
        (out_dir / "adaptive_feedback.partial.json").write_text(
            json.dumps(rounds, indent=2)
        )

        if stop_reason:
            break

        feedback_rounds_used += 1
        density = upcoming

    result = {
        "instance_id": args.instance,
        "model": args.model,
        "gold_used_by_agent_or_feedback": False,
        "initial_density": initial,
        "final_density": density,
        "density_schedule": schedule,
        "max_feedback_rounds": args.max_feedback_rounds,
        "feedback_rounds_used": feedback_rounds_used,
        "localization_runs": len(rounds),
        "stop_reason": stop_reason,
        "final_predictions": rounds[-1]["predictions"] if rounds else [],
        "rounds": rounds,
    }
    result_path = out_dir / "adaptive_feedback.json"
    result_path.write_text(json.dumps(result, indent=2))

    print("\n=== ADAPTIVE FEEDBACK COMPLETE ===", flush=True)
    print(f"Stop reason: {stop_reason}", flush=True)
    print(f"Feedback retries used: {feedback_rounds_used}", flush=True)
    print(f"Final target density: {density:.2f}", flush=True)
    print("Final Top-5:", flush=True)
    for index, candidate in enumerate(result["final_predictions"], 1):
        print(f"{index}. {candidate}", flush=True)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wp1.adaptive_feedback import (
    evaluate_context_sufficiency,
    next_density,
    normalize_density_schedule,
    retrieve_targeted_evidence,
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
        description=(
            "Run Agent4SR with gold-free evidence-guided feedback. "
            "Targeted evidence retrieval is attempted before density expansion."
        )
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--initial-density", type=float, default=0.30)
    parser.add_argument(
        "--density-schedule",
        default="0.30,0.50,0.70,1.00",
        help=(
            "Fallback density schedule. A higher density is used only when "
            "targeted evidence cannot be retrieved or feedback explicitly "
            "cannot name a concrete evidence target."
        ),
    )
    parser.add_argument(
        "--max-feedback-rounds",
        type=int,
        default=3,
        help="Maximum retries after the initial localization run.",
    )
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=120,
        help="Maximum Ollama output tokens per Agent4SR call.",
    )
    parser.add_argument(
        "--feedback-num-predict",
        type=int,
        default=320,
        help=(
            "Maximum Ollama output tokens for the structured feedback JSON. "
            "This is intentionally larger than the Agent4SR tool-call budget."
        ),
    )
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
        ollama_num_predict=args.ollama_num_predict,
    )
    feedback_backend = ChatBackend(
        provider="ollama",
        model=args.model,
        timeout=1800,
        ollama_num_predict=args.feedback_num_predict,
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

    # Evidence and localization hypotheses persist across retries.
    evidence_ledger: list[dict] = []
    evidence_keys: set[str] = set()
    previous_predictions: list[str] = []

    while True:
        runtime_output, compression = context_for_density(density)
        run_index = len(rounds) + 1
        ledger_size_before = len(evidence_ledger)

        print(
            f"\n=== Adaptive Agent4SR run {run_index}: "
            f"target density={density:.2f}, "
            f"targeted evidence={ledger_size_before} ===",
            flush=True,
        )

        prior_predictions = list(previous_predictions)

        agent = run_agent(
            backend,
            graph,
            problem,
            failing,
            runtime_output,
            max_steps=args.max_agent_steps,
            targeted_evidence=evidence_ledger,
            previous_predictions=prior_predictions,
        )

        current_predictions = list(agent.get("predictions") or [])
        if current_predictions:
            previous_predictions = current_predictions

        upcoming = next_density(density, schedule)
        decision = None
        action_taken = "stop"
        retrieved: list[dict] = []
        next_density_used = None

        if feedback_rounds_used >= args.max_feedback_rounds:
            stop_reason = "max_feedback_rounds"
        else:
            decision = evaluate_context_sufficiency(
                feedback_backend,
                problem=problem,
                failing_test=failing,
                runtime_output=runtime_output,
                agent_result=agent,
                current_density=density,
                next_density_value=upcoming,
                feedback_round=feedback_rounds_used + 1,
                max_feedback_rounds=args.max_feedback_rounds,
                evidence_ledger=evidence_ledger,
                previous_predictions=prior_predictions,
            )

            if decision.decision == "STOP":
                stop_reason = "feedback_stop"

            elif decision.targeted:
                print(
                    "[Feedback] TARGETED_EXPAND: retrieving concrete evidence "
                    "before changing density.",
                    flush=True,
                )
                retrieved = retrieve_targeted_evidence(
                    graph,
                    decision.missing_evidence,
                    raw_runtime_output=raw_output,
                    existing_keys=evidence_keys,
                )

                if retrieved:
                    evidence_ledger.extend(retrieved)
                    action_taken = "targeted_evidence"
                    feedback_rounds_used += 1
                    print(
                        f"[Feedback] retrieved {len(retrieved)} new evidence "
                        f"item(s); keeping density at {density:.2f}.",
                        flush=True,
                    )
                elif upcoming is not None:
                    action_taken = "density_fallback"
                    next_density_used = upcoming
                    feedback_rounds_used += 1
                    print(
                        "[Feedback] targeted retrieval produced no new evidence; "
                        f"falling back to density {upcoming:.2f}.",
                        flush=True,
                    )
                else:
                    stop_reason = "targeted_retrieval_failed_no_higher_density"

            else:
                # EXPAND_DENSITY is intentionally the fallback path.
                if upcoming is not None:
                    action_taken = "density_fallback"
                    next_density_used = upcoming
                    feedback_rounds_used += 1
                    print(
                        "[Feedback] no concrete targeted request was available; "
                        f"falling back to density {upcoming:.2f}.",
                        flush=True,
                    )
                else:
                    stop_reason = "no_higher_density"

        feedback_dict = decision.to_dict() if decision else None
        if feedback_dict is not None:
            feedback_dict["action_taken"] = action_taken
            feedback_dict["retrieved_evidence_count"] = len(retrieved)
            feedback_dict["next_density_used"] = next_density_used

        record = {
            "run_index": run_index,
            "target_density": density,
            "compression": compression,
            "predictions": agent.get("predictions", []),
            "agent": agent,
            "previous_predictions_supplied": prior_predictions,
            "evidence_ledger_size_before": ledger_size_before,
            "feedback": feedback_dict,
            "retrieved_evidence": retrieved,
        }
        rounds.append(record)

        (out_dir / "adaptive_feedback.partial.json").write_text(
            json.dumps(rounds, indent=2)
        )
        (out_dir / "evidence_ledger.json").write_text(
            json.dumps(evidence_ledger, indent=2)
        )

        if stop_reason:
            break

        if action_taken == "density_fallback":
            density = float(next_density_used)
        elif action_taken == "targeted_evidence":
            # Keep the same compressed context and rerun with only the newly
            # retrieved evidence added.
            pass
        else:
            stop_reason = "no_feedback_action"
            break

    result = {
        "instance_id": args.instance,
        "model": args.model,
        "agent_ollama_num_predict": args.ollama_num_predict,
        "feedback_ollama_num_predict": args.feedback_num_predict,
        "gold_used_by_agent_or_feedback": False,
        "feedback_policy": "targeted_evidence_first",
        "initial_density": initial,
        "final_density": density,
        "density_schedule": schedule,
        "max_feedback_rounds": args.max_feedback_rounds,
        "feedback_rounds_used": feedback_rounds_used,
        "localization_runs": len(rounds),
        "targeted_evidence_items": len(evidence_ledger),
        "stop_reason": stop_reason,
        "final_predictions": previous_predictions,
        "evidence_ledger": evidence_ledger,
        "rounds": rounds,
    }

    result_path = out_dir / "adaptive_feedback.json"
    result_path.write_text(json.dumps(result, indent=2))

    print("\n=== EVIDENCE-GUIDED ADAPTIVE FEEDBACK COMPLETE ===", flush=True)
    print(f"Stop reason: {stop_reason}", flush=True)
    print(f"Feedback retries used: {feedback_rounds_used}", flush=True)
    print(f"Final target density: {density:.2f}", flush=True)
    print(
        f"Targeted evidence items retrieved: {len(evidence_ledger)}",
        flush=True,
    )
    print("Final Top-5:", flush=True)
    for index, candidate in enumerate(result["final_predictions"], 1):
        print(f"{index}. {candidate}", flush=True)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()

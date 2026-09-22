from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wp1.graphify_structure import GraphifyIndex
from wp1.leanctx_density import compress_to_density, resolve_density_helper
from wp1.llm_backends import ChatBackend
from wp1.run_swebench_agent4sr_pair import run_agent


def density_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if start < stop:
        raise ValueError("start must be >= stop")

    values: list[float] = []
    current = start
    while current >= stop - 1e-9:
        values.append(round(current, 6))
        current -= step
    return values


def context_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def exact_gold_rank(predictions: list[str], gold_entity: str) -> int | None:
    for index, prediction in enumerate(predictions, 1):
        if prediction.strip() == gold_entity.strip():
            return index
    return None


def gold_file_rank(predictions: list[str], gold_entity: str) -> int | None:
    gold_file = gold_entity.split("::", 1)[0].strip()
    for index, prediction in enumerate(predictions, 1):
        pred_file = prediction.split("::", 1)[0].strip()
        if pred_file == gold_file:
            return index
    return None


def classify(predictions: list[str], gold_entity: str) -> dict:
    if len(predictions) != 5:
        return {
            "status": "INVALID",
            "exact_gold_rank": None,
            "gold_file_rank": None,
            "exact_gold_hit": False,
            "gold_file_hit": False,
        }

    exact_rank = exact_gold_rank(predictions, gold_entity)
    file_rank = gold_file_rank(predictions, gold_entity)
    return {
        "status": "HIT" if exact_rank is not None else "MISS",
        "exact_gold_rank": exact_rank,
        "gold_file_rank": file_rank,
        "exact_gold_hit": exact_rank is not None,
        "gold_file_hit": file_rank is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LeanCTX target density in fine increments and run Graphify-assisted "
            "Agent4SR on each unique compressed context. Gold is evaluation-only."
        )
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument(
        "--gold-entity",
        required=True,
        help="Evaluation-only gold entity, for example path.py::ClassName.",
    )
    parser.add_argument("--density-start", type=float, default=1.00)
    parser.add_argument("--density-stop", type=float, default=0.70)
    parser.add_argument("--density-step", type=float, default=0.01)
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=120,
        help="Maximum Ollama output tokens per Agent4SR call.",
    )
    parser.add_argument("--density-helper", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = Path.home() / "AdaptiveContextOpt"
    work = root / "data/swebench_workspaces" / args.instance
    outputs = work / "outputs"
    out_dir = args.output_dir or outputs / "fine_density_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir = out_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)

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
    helper = resolve_density_helper(args.density_helper)

    requested = density_values(
        args.density_start,
        args.density_stop,
        args.density_step,
    )

    context_cache: dict[str, dict] = {}
    rows: list[dict] = []

    for density in requested:
        if density >= 1.0 - 1e-9:
            runtime_output = raw_output
            compression = {
                "target_density": 1.0,
                "original_tokens": None,
                "compressed_tokens": None,
                "saved_percent": 0.0,
                "helper": None,
                "mode": "raw",
            }
        else:
            compressed = compress_to_density(
                raw_output,
                density,
                helper=helper,
            )
            runtime_output = compressed.text
            compression = {
                **compressed.to_dict(),
                "mode": "leanctx_density_research",
            }

        sha = context_sha(runtime_output)

        if sha in context_cache:
            source = context_cache[sha]
            row = {
                "target_density": density,
                "actual_saved_percent": compression["saved_percent"],
                "original_tokens": compression["original_tokens"],
                "compressed_tokens": compression["compressed_tokens"],
                "context_sha256": sha,
                "deduplicated": True,
                "reused_from_density": source["target_density"],
                "predictions": source["predictions"],
                **source["evaluation"],
            }
            rows.append(row)
            print(
                f"density={density:.2f} actual_reduction={compression['saved_percent']:.2f}% "
                f"DEDUP -> density={source['target_density']:.2f} "
                f"status={source['evaluation']['status']}",
                flush=True,
            )
            continue

        context_path = contexts_dir / f"density_{round(density * 100):03d}.txt"
        context_path.write_text(runtime_output)

        print(
            f"\n=== target density={density:.2f} | "
            f"actual reduction={compression['saved_percent']:.2f}% ===",
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
        predictions = list(agent.get("predictions") or [])
        evaluation = classify(predictions, args.gold_entity)

        record = {
            "target_density": density,
            "compression": compression,
            "context_sha256": sha,
            "predictions": predictions,
            "agent": agent,
            "evaluation": evaluation,
        }
        context_cache[sha] = record

        row = {
            "target_density": density,
            "actual_saved_percent": compression["saved_percent"],
            "original_tokens": compression["original_tokens"],
            "compressed_tokens": compression["compressed_tokens"],
            "context_sha256": sha,
            "deduplicated": False,
            "reused_from_density": None,
            "predictions": predictions,
            **evaluation,
        }
        rows.append(row)

        (out_dir / "sweep_results.partial.json").write_text(
            json.dumps(rows, indent=2)
        )

        rank_text = (
            str(evaluation["exact_gold_rank"])
            if evaluation["exact_gold_rank"] is not None
            else "-"
        )
        print(
            f"RESULT target={density:.2f} actual_reduction={compression['saved_percent']:.2f}% "
            f"status={evaluation['status']} exact_gold_rank={rank_text}",
            flush=True,
        )

    valid_rows = [row for row in rows if row["status"] in {"HIT", "MISS"}]
    misses = [row for row in valid_rows if row["status"] == "MISS"]
    file_misses = [row for row in valid_rows if not row["gold_file_hit"]]

    first_exact_miss = (
        min(misses, key=lambda row: row["actual_saved_percent"])
        if misses
        else None
    )
    first_file_miss = (
        min(file_misses, key=lambda row: row["actual_saved_percent"])
        if file_misses
        else None
    )

    summary = {
        "instance_id": args.instance,
        "model": args.model,
        "gold_entity": args.gold_entity,
        "gold_file": args.gold_entity.split("::", 1)[0],
        "gold_used_by_agent": False,
        "density_start": args.density_start,
        "density_stop": args.density_stop,
        "density_step": args.density_step,
        "ollama_num_predict": args.ollama_num_predict,
        "requested_density_count": len(requested),
        "unique_context_count": len(context_cache),
        "first_exact_gold_miss": first_exact_miss,
        "first_gold_file_miss": first_file_miss,
        "rows": rows,
    }

    (out_dir / "fine_density_results.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== FINE DENSITY SWEEP COMPLETE ===", flush=True)
    print(f"Requested densities: {len(requested)}", flush=True)
    print(f"Unique contexts actually run: {len(context_cache)}", flush=True)

    if first_exact_miss:
        print(
            "First observed exact-gold MISS: "
            f"{first_exact_miss['actual_saved_percent']:.2f}% actual reduction "
            f"(target density {first_exact_miss['target_density']:.2f})",
            flush=True,
        )
    else:
        print("No exact-gold MISS observed in this range.", flush=True)

    if first_file_miss:
        print(
            "First observed gold-file MISS: "
            f"{first_file_miss['actual_saved_percent']:.2f}% actual reduction "
            f"(target density {first_file_miss['target_density']:.2f})",
            flush=True,
        )
    else:
        print("No gold-file MISS observed in this range.", flush=True)

    print(f"Saved: {out_dir / 'fine_density_results.json'}", flush=True)


if __name__ == "__main__":
    main()

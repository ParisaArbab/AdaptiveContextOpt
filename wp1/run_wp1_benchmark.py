"""
run_wp1_benchmark.py — WP1 orchestrator (ablation matrix).

    per instance:
      capture real trigger-test failure (docker | local fallback)
      graphify extract  -> structure map + real call graph      [cached per repo]
      for each ablation arm:
        [leanctx]  compress the captured output
        [feedback] bidirectional restore/prune over the compressed text
        [graphify] structural briefing primes FlexFL Stage 1
        FlexFL Stage 1 (Agent4SR) -> Stage 2 (Agent4LR)
        [graphify] GraphLocator causal expansion
        score: Top-1/3/5, MAP, MRR at method AND file level
        record: tokens, per pipeline stage, with a real tokenizer

Every arm sees the SAME captured output and the SAME structure map, so the
only variables are the three pipeline elements. Token accounting is per-arm
and per-stage, which is what makes both halves of the evaluation answerable:
how much each element saves, and what it costs in localization accuracy.

Usage:
    python run_wp1_benchmark.py --instances data/instances.json \\
        --arms default --backend heuristic --local-fallback \\
        --out results/wp1_results.json

    # full 2^3 factorial on a real model
    python run_wp1_benchmark.py --instances data/instances.json \\
        --arms all --backend claude --out results/wp1_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import ablation
import agent_localizer
import benchmarks
import docker_harness
import graphify_structure
import leanctx_compressor
import llm_backends
import metrics
import token_meter
from compression_tax_analyzer import InstanceOutcome, classify_taxonomy




def run_one_arm(
    arm: ablation.Arm,
    instance: dict,
    raw_output: str,
    structure_map: dict,
    call_graph: dict,
    repo_root: Path,
    llm: llm_backends.LLMConfig,
    chat_fn,
    capture_mode: str,
    target_density: float,
    tokenizer_model: Optional[str] = None,
    coverage_json: Optional[dict] = None,
    coverage_error: str = "",
) -> InstanceOutcome:
    backend = llm.label
    is_heuristic = llm.spec.kind == "heuristic"
    meter = token_meter.TokenMeter(
        token_meter.TokenCounter(llm.spec.name, tokenizer_model or llm.resolved_model)
    )
    meter.record_context("tool_output_raw_tokens", raw_output)

    compressor_mode = "n/a"
    if arm.use_leanctx:
        cr = leanctx_compressor.compress(raw_output, target_density=target_density)
        agent_text = cr.text
        compressor_mode = cr.mode
    else:
        agent_text = raw_output
    meter.record_context("tool_output_after_compression_tokens", agent_text)

    started = time.time()
    if is_heuristic:
        result = agent_localizer.HeuristicBackend().localize(
            agent_text, structure_map, instance["problem_statement"],
            call_graph=call_graph, repo_root=repo_root,
            coverage_json=coverage_json, coverage_error=coverage_error,
            failing_test_ids=instance.get("fail_to_pass", []),
            use_graph=arm.use_graphify,
            use_feedback_loop=arm.use_feedback,
            raw_tool_output=raw_output if arm.use_feedback else None,
            meter=meter,
        )
        result.instance_id = instance["instance_id"]
    else:
        result = agent_localizer.localize_with_llm(
            instance_id=instance["instance_id"],
            coverage_json=coverage_json,
            coverage_error=coverage_error,
            failing_test_ids=instance.get("fail_to_pass", []),
            tool_output=agent_text,
            structure_map=structure_map,
            problem_statement=instance["problem_statement"],
            chat_fn=chat_fn,
            backend_name=backend,
            call_graph=call_graph,
            repo_root=repo_root,
            use_graph=arm.use_graphify,
            use_feedback_loop=arm.use_feedback,
            raw_tool_output=raw_output if arm.use_feedback else None,
            meter=meter,
        )
    elapsed = round(time.time() - started, 2)

    scores = metrics.score_instance(
        result.predicted_functions, result.predicted_files,
        instance.get("functions", []), instance.get("files", []),
    )
    tags = classify_taxonomy(raw_output, agent_text) if arm.use_leanctx else []

    return InstanceOutcome(
        instance_id=instance["instance_id"],
        arm=arm.name,
        arm_flags=arm.as_dict(),
        backend=backend,
        dataset=instance.get("dataset", ""),
        language=instance.get("language", "python"),
        capture_mode=capture_mode,
        compressor_mode=compressor_mode,
        predicted_functions=result.predicted_functions,
        predicted_files=result.predicted_files,
        ground_truth_functions=instance.get("functions", []),
        ground_truth_files=instance.get("files", []),
        method_scores=scores["method_level"],
        file_scores=scores["file_level"],
        token_report=meter.report(),
        feedback_rounds_used=result.feedback_rounds_used,
        feedback_restores=result.feedback_restores,
        feedback_prunes=result.feedback_prunes,
        feedback_stop_reason=result.feedback_stop_reason,
        graph_expanded=result.graph_expanded,
        taxonomy_tags=tags,
        wall_seconds=elapsed,
        agent4sr_top5=result.agent4sr_top5,
        flexfl_merge=result.flexfl_merge,
        protocol_stats=result.protocol_stats,
        stage_transcripts=result.stage_transcripts,
        provisional=(compressor_mode == "reference"
                     or capture_mode == "local_fallback"
                     or is_heuristic),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/instances.json")
    ap.add_argument("--repos-dir", default="data/repos")
    ap.add_argument("--arms", nargs="*", default=["default"],
                    help="'default' (5 arms), 'all' (full factorial), or explicit names")
    ap.add_argument("--local-fallback", action="store_true",
                    help="run trigger tests on the host instead of the SWE-bench images")
    ap.add_argument("--llm", "--backend", dest="llm", default="heuristic",
                    help="provider name or alias; --list-providers to see them all")
    ap.add_argument("--model", default=None,
                    help="model id (required for providers with no default, "
                         "e.g. ollama/vllm/openrouter)")
    ap.add_argument("--base-url", default=None,
                    help="override the provider's endpoint (any OpenAI-compatible server)")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the key, if not the provider default")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 keeps arms comparable; raise only for variance studies")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tokenizer-model", default=None,
                    help="HF repo id to count tokens with, when the served model "
                         "id isn't one (e.g. an ollama tag)")
    ap.add_argument("--list-providers", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="test the provider with one cheap call before running")
    ap.add_argument("--target-density", type=float, default=0.4,
                    help="lean-ctx density target (fraction of original tokens kept)")
    ap.add_argument("--limit", type=int, default=0, help="cap instances processed (0 = all)")
    ap.add_argument("--out", default="results/wp1_results.json")
    args = ap.parse_args()

    if args.list_providers:
        print("providers:\n" + llm_backends.list_providers())
        return

    llm = llm_backends.resolve(
        args.llm, model=args.model, base_url=args.base_url,
        api_key_env=args.api_key_env, temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.preflight:
        print(llm_backends.preflight(llm))

    arms = ablation.resolve_arms(args.arms)
    instances = json.loads(Path(args.instances).read_text())
    if args.limit:
        instances = instances[: args.limit]
    chat_fn = llm_backends.build_chat_fn(llm)
    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    print(f"arms: {', '.join(a.name for a in arms)}")
    print(f"llm: {llm.label} | instances: {len(instances)}")

    all_outcomes: List[InstanceOutcome] = []
    skipped: List[dict] = []

    for inst in instances:
        instance_id = inst["instance_id"]
        language = inst.get("language", "python")
        adapter = benchmarks.LANGUAGES.get(language, benchmarks.LANGUAGES["python"])
        print(f"=== {instance_id} ({language}) ===")

        repo_local_path = repos_dir / instance_id.replace("/", "_")
        try:
            if args.local_fallback:
                run_result = docker_harness.run_local_fallback(
                    instance_id=instance_id,
                    pass_to_pass=inst.get("pass_to_pass", []),
                    repo=inst["repo"],
                    base_commit=inst["base_commit"],
                    test_patch=inst.get("test_patch", ""),
                    fail_to_pass=inst.get("fail_to_pass", []),
                    adapter=adapter,
                    workdir=repo_local_path,
                )
            else:
                run_result = docker_harness.run_in_docker(
                    instance_id=instance_id,
                    test_patch=inst.get("test_patch", ""),
                    fail_to_pass=inst.get("fail_to_pass", []),
                    adapter=adapter,
                )
                # graphify parses the host filesystem, so the checkout is
                # needed even when the tests ran inside the image
                docker_harness.ensure_checkout(inst["repo"], inst["base_commit"],
                                               repo_local_path)
        except Exception as e:
            print(f"  capture failed ({e}); skipping instance")
            skipped.append({"instance_id": instance_id, "reason": f"capture: {e}"})
            continue

        for note in run_result.notes:
            print(f"  note: {note}")
        if not run_result.has_failure_evidence:
            # Localizing from a capture with no failure signal measures
            # nothing about compression; keeping such an instance in the
            # aggregate would dilute every arm identically and hide it.
            print("  no failure evidence captured; skipping instance")
            skipped.append({"instance_id": instance_id, "reason": "no failure evidence",
                            "command": run_result.command})
            continue

        try:
            structure_map = graphify_structure.build_structure_map(
                repo_local_path, language=adapter.graphify_language)
            call_graph = graphify_structure.build_call_graph(repo_local_path)
        except Exception as e:
            print(f"  graphify failed ({e}); skipping instance")
            skipped.append({"instance_id": instance_id, "reason": f"graphify: {e}"})
            continue

        raw_output = run_result.text

        for arm in arms:
            try:
                outcome = run_one_arm(
                    arm=arm, instance=inst, raw_output=raw_output,
                    coverage_json=run_result.coverage_json,
                    coverage_error=run_result.coverage_error,
                    structure_map=structure_map, call_graph=call_graph,
                    repo_root=repo_local_path, llm=llm, chat_fn=chat_fn,
                    capture_mode=run_result.mode, target_density=args.target_density,
                    tokenizer_model=args.tokenizer_model,
                )
            except Exception as e:
                print(f"  [{arm.name:<12s}] FAILED: {e}")
                traceback.print_exc()
                skipped.append({"instance_id": instance_id, "arm": arm.name,
                                "reason": f"arm: {e}"})
                continue
            all_outcomes.append(outcome)
            tok = outcome.token_report
            ms = outcome.method_scores or {}
            print(f"  [{arm.name:<12s}] top1={ms.get('top1', 0):.0f} top5={ms.get('top5', 0):.0f} "
                  f"mrr={ms.get('mrr', 0):.2f} | ctx={tok['context_tokens'].get('agent_input_tokens', 0)} "
                  f"llm={tok['llm_total_tokens']} tok | {outcome.wall_seconds}s")

            # Why those candidates. A Top-1 of zero is unattributable without
            # this: it separates "the model never produced a ranking" from
            # "Ochiai was unavailable" from "everything ran and still missed".
            fm = outcome.flexfl_merge or {}
            if fm:
                contributing = [
                    f"{name}:{src.get('n_entries', 0)}"
                    for name, src in (fm.get("sources") or {}).items()
                    if src.get("available")
                ]
                print(f"       merge: {fm.get('mode', '?')}"
                      f" | sources[{', '.join(contributing) or 'none'}]"
                      f" | agent4sr={len(fm.get('agent4sr_used') or [])}"
                      f" -> {fm.get('n_candidates', 0)} candidates")
                for miss in (fm.get("unavailable") or []):
                    print(f"       unavailable: {miss}")
                if fm.get("note"):
                    print(f"       note: {fm['note']}")
            ps = getattr(outcome, "protocol_stats", None) or {}
            if ps and not ps.get("agent4sr_produced_ranking", True):
                print("       WARNING: Agent4SR never produced a parseable Top_k block "
                      "— the model is not following the ReAct/output protocol. "
                      "This is expected below roughly 7B parameters; the merge ran "
                      "on the traditional localizers alone.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "arms": [a.as_dict() for a in arms],
            "backend": llm.label,
            "provider": llm.spec.name,
            "model": llm.resolved_model,
            "base_url": llm.base_url or llm.spec.base_url,
            "temperature": llm.temperature,
            "capture_mode": "local_fallback" if args.local_fallback else "docker",
            "target_density": args.target_density,
            "instances_file": args.instances,
        },
        "outcomes": [asdict(o) for o in all_outcomes],
        "skipped": skipped,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(all_outcomes)} outcomes ({len(skipped)} skipped) -> {out_path}")


if __name__ == "__main__":
    main()

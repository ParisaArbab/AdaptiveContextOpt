"""Reproducible small real-input adaptive RAVE-FL experiment.

python -m rave_fl.pilot --manifest configs/rave_pilot_time25.json --dry-run --output results/pilot_preflight
python -m rave_fl.pilot --manifest configs/rave_pilot_time25.json --model qwen2.5:7b --output results/pilot_live
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from rave_fl.adaptive import AdaptivePilot
from rave_fl.code_sketch import prepare_candidates
from rave_fl.compare_expanded import load_context
from rave_fl.expanded_pool import union_pool
from rave_fl.pilot_backend import OllamaPilotBackend
from rave_fl.pilot_evaluation import evaluate, risk_rows, summarize
from rave_fl.pilot_types import PolicyConfig
from rave_fl.risk import BootstrapRisk, LearnedRisk
from rave_fl.rank_repair import REPAIR_MODES, RepairConfig, RankRepairPilot
from rave_fl.source_spans import extract_pool_sources
from wp1.flexfl_data import read_ground_truth


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def prepare_case(case, top_per_method=10):
    repo = Path(case["flexfl_repo"])
    candidates, metadata = union_pool(repo, case["bug"], top_per_method)
    sources, paths = extract_pool_sources(Path(case["checkout"]), metadata)
    context, evidence = load_context(repo, case["bug"], Path(case["evidence_file"]) if case.get("evidence_file") else None)
    prepared = prepare_candidates(candidates, sources, metadata, context)
    revision = subprocess.run(["git", "-C", str(case["checkout"]), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    provenance = {"bug": case["bug"], "candidate_ids": [c.function_id for c in prepared],
                  "source_paths": paths, "sources_sha256": digest(sources), "evidence": evidence,
                  "source_revision": revision,
                  "top_per_method": top_per_method,
                  "candidate_features": {c.function_id: c.features for c in prepared},
                  "sketch_metadata": {c.function_id: c.sketch_metadata for c in prepared}}
    return prepared, context, provenance


def run_experiment(args):
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    if not cases or len(cases) > args.max_bugs:
        raise ValueError("Manifest must contain between 1 and --max-bugs cases (small pilot guard)")
    if len({c["bug"] for c in cases}) != len(cases):
        raise ValueError("Use one case per bug for paired summaries; separate evidence ablations into separate runs")
    arms = args.arms.split(",")
    allowed = {"full", "rank_threshold", "fixed_batch", "fixed_sketch", "confidence", "adaptive"} | REPAIR_MODES
    if not arms or len(set(arms)) != len(arms) or set(arms) - allowed:
        raise ValueError("Unknown or duplicate experiment arms")
    if args.collect_risk and "full" not in arms:
        raise ValueError("--collect-risk requires --arms containing full")
    if args.risk_model and args.collect_risk:
        raise ValueError("Collect training probes separately from learned-policy evaluation")
    if args.risk_model and set(arms) & REPAIR_MODES:
        raise ValueError("The rank-repair heuristic does not use a trained risk model")
    if not args.dry_run and not args.model:
        raise ValueError("Live experiments require --model")
    config = PolicyConfig(args.token_budget, args.batch_size, args.risk_threshold,
                          args.uncertainty_threshold, early_stop=not args.no_early_stop,
                          max_steps=args.max_steps)
    repair_config = RepairConfig(getattr(args, "max_full_repairs", 4), getattr(args, "repair_top_k", 5))
    policy_budget = getattr(args, "policy_budget", None)
    if policy_budget is not None and policy_budget < 0:
        raise ValueError("Policy budget must be nonnegative")
    # Validate every case and model artifact before spending tokens.
    ready = []
    for case in cases:
        if not case["bug"] or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in case["bug"]):
            raise ValueError("Invalid bug ID")
        candidates, context, provenance = prepare_case(case, args.top_per_method)
        risk = LearnedRisk(args.risk_model, case["bug"]) if args.risk_model else BootstrapRisk()
        ready.append((case, candidates, context, provenance, risk))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty/new; do not append or overwrite experiment ledgers")
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    code = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    code_hash = digest(code)
    for name, source in code.items():
        snapshot = args.output / "code_snapshot" / "rave_fl" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(source)
    run = {"schema": 1, "dry_run": args.dry_run, "model": args.model, "base_url": args.base_url,
           "seed": args.seed, "code_revision": revision, "tracked_changes": bool(dirty), "code_sha256": code_hash,
           "python": sys.version, "platform": platform.platform(), "arms_requested": arms,
           "risk_model_artifact_sha256": digest(json.loads(args.risk_model.read_text())) if args.risk_model else None,
           "config": asdict(config), "policy_budget": policy_budget, "cases": [],
           "scope": "post-localizer verification pilot; not a reproduced SieveFL pipeline",
           "limitations": ["Preflight token estimates are not a provider-enforced hard cap.",
                           "Shared pilot fusion ranker is not the original SieveFL final ranker.",
                           "Bootstrap risk is untrained; candidate harm calibration does not certify issue-level stopping.",
                           "All per-arm model requests, including invalid replies and repeated context, are charged.",
                           "Upstream retrieval tokens are not included; this is the verification-stage comparison.",
                           "Ollama counters measure logical tokens; monetary API cost is not available."]}
    write_json(args.output / "run.json", run)
    for case, candidates, context, provenance, risk in ready:
        bug = case["bug"]
        result = {**provenance, "arms": {}, "ground_truth_evaluation_only": True}
        result["inference_fingerprint"] = digest({"provenance": provenance, "model": args.model, "seed": args.seed,
                                                  "output_cap": args.max_output_tokens, "context_window": args.context_window})
        write_json(args.output / bug / "input_snapshot.json",
                   {"provenance": provenance, "context": context,
                    "candidates": [asdict(c) for c in candidates]})
        start = time.monotonic()
        modes = arms + (["probe_l0", "probe_l1"] if args.collect_risk else [])
        try:
            for name in modes:
                cfg = replace(config, mode="fixed_batch" if name == "probe_l1" else name,
                              token_budget=policy_budget if policy_budget is not None and name not in
                              {"full", "probe_l0", "probe_l1"} else config.token_budget)
                backend = OllamaPilotBackend(args.model, context, args.base_url, args.max_output_tokens,
                                             args.timeout, args.output / bug / (name + "_calls.jsonl"),
                                             args.seed, args.context_window)
                if args.dry_run:
                    from rave_fl.core import Action
                    result["arms"][name] = {"real_verifier": False, "estimated_full_tokens": sum(
                        backend.estimate([c], Action.FULL_VERIFY, 2) for c in candidates)}
                    continue
                backend.preflight()
                engine = (RankRepairPilot(candidates, backend, cfg, repair_config)
                    if name in REPAIR_MODES else AdaptivePilot(candidates, backend, cfg, risk))
                result["arms"][name] = engine.execute()
                result["arms"][name]["model_metadata"] = backend.model_metadata
                write_json(args.output / bug / (name + ".json"), result["arms"][name])
                print(f"{bug} {name}: {result['arms'][name]['call_count']} calls, {result['arms'][name]['actual_tokens']} tokens", flush=True)
            if not args.dry_run:
                # Labels are first read only after all inference arms have finished.
                truths = read_ground_truth(Path(case["flexfl_repo"]), bug)
                for arm in result["arms"].values():
                    arm["accuracy"] = evaluate(arm["ranked_functions"], truths, arm["states"])
                for name, arm in result["arms"].items():
                    write_json(args.output / bug / (name + ".json"), arm)
                if args.collect_risk:
                    rows = risk_rows(bug, result, truths)
                    with (args.output / "risk_dataset.jsonl").open("a") as out:
                        for row in rows:
                            out.write(json.dumps(row, allow_nan=False) + "\n")
            result["elapsed_seconds"] = time.monotonic() - start
            write_json(args.output / bug / "result.json", result)
            run["cases"].append(result)
            run["summary"] = summarize(run["cases"]) if not args.dry_run else {}
            write_json(args.output / "run.json", run)
        except Exception as exc:
            write_json(args.output / bug / "failure.json", {"error": str(exc), "completed_arms": result,
                                                          "usage_may_be_incomplete": True})
            raise
    run["status"] = "preflight_only" if args.dry_run else "complete"
    write_json(args.output / "run.json", run)
    write_report(args.output / "report.md", run)
    return run


def write_report(path, run):
    lines = ["# RAVE-FL adaptive pilot", "", f"Status: {run['status']}. Model: {run['model']}. Seed: {run['seed']}.",
             "", "Verification-stage experiment; shared pilot fusion ranker; no claim of full SieveFL reproduction.", "",
             "| Bug | Arm | Calls | Screened/full | Actual tokens | Top-1 | Top-5 | MRR | Stop |",
             "|---|---|---:|---:|---:|---|---|---:|---|"]
    for case in run["cases"]:
        for name, a in case["arms"].items():
            if "accuracy" not in a:
                continue
            e = a["accuracy"]
            lines.append(f"| {case['bug']} | {name} | {a['call_count']} | {a['screened_count']}/{a['full_verified_count']} | {a['actual_tokens']} | {e['top1']} | {e['top5']} | {e['MRR']:.3f} | {a['stop_reason']} |")
    lines += ["", "## Limits", ""] + ["- " + s for s in run["limitations"]]
    lines += ["", "Paired confidence intervals are omitted for one bug. No tiny pilot establishes noninferiority or novelty.", ""]
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collect-risk", action="store_true")
    p.add_argument("--risk-model", type=Path)
    p.add_argument("--arms", default="full,rank_threshold,fixed_batch,confidence,adaptive")
    p.add_argument("--token-budget", type=int, default=16000)
    p.add_argument("--policy-budget", type=int, help="Optional smaller budget for non-Full evaluation arms; Full and training probes retain --token-budget")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-output-tokens", type=int, default=160, help="Per candidate; batch cap multiplies by batch size")
    p.add_argument("--context-window", type=int, default=16384)
    p.add_argument("--max-bugs", type=int, default=5)
    p.add_argument("--top-per-method", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-full-repairs", type=int, default=4)
    p.add_argument("--repair-top-k", type=int, default=5)
    p.add_argument("--risk-threshold", type=float, default=.35)
    p.add_argument("--uncertainty-threshold", type=float, default=.45)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    try:
        result = run_experiment(args)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    print(json.dumps({"status": result["status"], "summary": result.get("summary", {}),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

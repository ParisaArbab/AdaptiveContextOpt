"""Offline held-out policy evaluation from real paired action-outcome rows.

Compares STOP, cheapest-action, estimated-value policy and posthoc oracle
under identical per-bug candidate actions and an input-token ceiling.
The oracle may read action outcomes and is never deployable. Training
and calibration project rows are explicitly excluded.

Metrics are downstream *change in predicted changed-method correctness*,
not a new run of the language model and not verified patch causality.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .marginal_action_policy import EvidenceAction, decide_next_action
from .paired_action_outcomes import validate_pairs
from .train_marginal_outcomes import LearnedMarginalEstimator


def _summarize(decisions: list[dict]) -> dict:
    acquired = [r for r in decisions if r["action_id"] is not None]
    return {
        "bugs": len(decisions),
        "actions_acquired": len(acquired),
        "coverage_fraction": (len(acquired) / len(decisions)
                              if decisions else 0.0),
        "helped_bugs": sum(r["outcome"] == 1 for r in decisions),
        "harmed_bugs": sum(r["outcome"] == -1 for r in decisions),
        "unchanged_bugs": sum(r["outcome"] == 0 for r in decisions),
        "harm_fraction_among_acquisitions": (
            sum(r["outcome"] == -1 for r in acquired) / len(acquired)
            if acquired else 0.0),
        "net_correctness_change": sum(r["outcome"] for r in decisions),
        "total_extra_input_tokens": sum(r["extra_input_tokens"] for r in decisions),
        "total_expected_extra_latency_ms": sum(r["extra_latency_ms"]
                                               for r in decisions),
    }


def evaluate(rows: list[dict], artifact: dict,
             *, token_penalty: float = 0.001) -> dict:
    summary = validate_pairs(rows)
    if not artifact.get("independent_project_calibration_verified"):
        raise ValueError("cannot evaluate an uncalibrated policy")
    if artifact.get("source_pairs_sha256") is None:
        raise ValueError("missing training-data provenance")
    if artifact.get("model_id") != summary["model_id"]:
        raise ValueError("evaluation model differs from trained model")
    if artifact.get("outcome_definition") != summary["outcome_definition"]:
        raise ValueError("mixed file-level and method-level outcome labels")
    if artifact.get("common_input_token_ceiling") != summary["common_input_token_ceiling"]:
        raise ValueError("evaluation ceiling differs from trained model")
    for row in rows:
        if row["split"] == "test" and artifact.get(
                "project_split", {}).get(row["project"]) != "test":
            raise ValueError("test project missing from independently frozen artifact")
    test_rows = [r for r in rows if r["split"] == "test"]
    if not test_rows:
        raise ValueError("no held-out test outcomes")

    groups = defaultdict(list)
    for row in test_rows:
        groups[(row["project"], row["bug"], row["seed"])].append(row)
    estimator = LearnedMarginalEstimator(artifact)
    per_policy = {name: [] for name in
                  ("stop_always", "force_cheapest", "learned_net_value",
                   "posthoc_oracle")}
    for (_, bug, seed), candidates in sorted(groups.items()):
        ceilings = {r["common_input_token_ceiling"] for r in candidates}
        baseline_tokens = {r["actual_baseline_input_tokens"] for r in candidates}
        baseline_prompts = {r["baseline_prompt_sha256"] for r in candidates}
        failure_hashes = {r["failure_sha256"] for r in candidates}
        baseline_predictions = {r["baseline_prediction_sha256"] for r in candidates}
        if any(len(x) != 1 for x in (ceilings, baseline_tokens,
                                     baseline_prompts, failure_hashes,
                                     baseline_predictions)):
            raise ValueError(f"unmatched baseline across actions for {bug}")
        ceiling = next(iter(ceilings))
        remaining = ceiling - next(iter(baseline_tokens))
        options = [EvidenceAction(**r["action"]) for r in candidates]
        by_id = {r["action"]["action_id"]: r for r in candidates}
        eligible = [
            r for r in candidates
            if r["action"]["incremental_input_tokens"] <= remaining
            and (r["action"]["kind"] != "GRAPH"
                 or r["action"].get("compiler_binding_verified", False))
            and (not r["action"].get("requires_call_binding", False)
                 or r["action"].get("compiler_binding_verified", False))
        ]
        # An oracle sees outcomes, but never modifies policy estimates.
        oracle_choice = sorted(
            eligible,
            key=lambda r: (-r["outcome"],
                           r["action"]["incremental_input_tokens"],
                           r["action"]["action_id"])
        )[0] if eligible else None
        # Cost-aware oracle can always STOP, even when all actions hurt.
        if oracle_choice is not None and oracle_choice["outcome"] <= 0:
            oracle_choice = None
        cheapest = sorted(
            eligible, key=lambda r: (r["action"]["incremental_input_tokens"],
                                     r["action"]["action_id"])
        )[0] if eligible else None
        action_decision = decide_next_action(
            options, estimator, remaining, token_penalty=token_penalty)
        choices = {
            "stop_always": None,
            "force_cheapest": cheapest,
            "learned_net_value": by_id.get(action_decision.action_id),
            "posthoc_oracle": oracle_choice,
        }
        for name, chosen in choices.items():
            act = chosen["action"] if chosen else None
            per_policy[name].append({
                "bug": bug, "seed": seed, "action_id": act["action_id"] if act else None,
                "outcome": chosen["outcome"] if chosen else 0,
                "extra_input_tokens": (
                    chosen["actual_action_input_tokens"]
                    - chosen["actual_baseline_input_tokens"] if chosen else 0
                ),
                "extra_latency_ms": act["incremental_latency_ms"] if act else 0,
            })
    return {
        "scope": "Held-out offline action-policy counterfactual evaluation",
        "model_id": summary["model_id"],
        "source_pairs_sha256": summary["sha256"],
        "trained_artifact_pairs_sha256": artifact["source_pairs_sha256"],
        "oracle_is_posthoc_not_deployable": True,
        "actual_llm_rerun_during_evaluation": False,
        "outcome_is_changed_method_correctness_not_patch_success": True,
        "policies": {
            name: {"metrics": _summarize(decisions), "decisions": decisions}
            for name, decisions in per_policy.items()
        },
        "limitations": [
            "Offline outcomes only cover frozen candidate actions and states",
            "Choosing actions can change later states; no sequential policy value proven",
            "Held-out test projects must remain excluded from training/calibration",
            "Patch-changed method correctness is not proof of bug causality",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-jsonl", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.pairs_jsonl.read_text().splitlines()
            if x.strip()]
    artifact = json.loads(args.artifact.read_text())
    result = evaluate(rows, artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved held-out policy comparison to", args.output)


if __name__ == "__main__":
    main()

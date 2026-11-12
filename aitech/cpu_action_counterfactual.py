"""CPU-only STOP versus single L0->L1 upgrade, with frozen paired prompts.

A development feasibility pilot on PREVIOUSLY EXAMINED Defects4J snapshots.
It does not train the controller or claim independent generalization.
Gold patch methods are loaded only after writing raw paired predictions.
Same model, decoding, failure text, candidate IDs and total input ceiling.
Actual input tokens and CPU inference times are recorded per condition.

An upgrade is tested only when it fits the same evidence budget and increases
actual model-token detail. If no such upgrade exists, the bug is skipped and
an explicit reason recorded; never fabricate a counterfactual outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cpu_llm_localization import (
    MODEL_ID, BUG_INPUTS, _token_count, build_messages, parse_prediction,
    text_for_candidate, with_real_costs,
)
from .failure_conditioned_selector import failure_anchors


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_upgrade_pair(snapshot: dict, tokenizer, *,
                        evidence_budget: int, max_total_input: int) -> dict:
    """Freeze fixed-candidate STOP/UPGRADE_L1 prompts without reading gold."""
    if evidence_budget <= 0 or max_total_input <= 0:
        raise ValueError("positive budgets required")
    source = with_real_costs(snapshot, tokenizer)
    items = source["candidates"]
    chosen = {}
    used = 0
    # Cap the initial signature-only shortlist so there is a genuine
    # opportunity to test an L0 -> L1 upgrade under the SAME total
    # evidence ceiling. The shortlist size is fixed before seeing gold.
    max_baseline_candidates = 8
    for i, item in enumerate(items):
        if len(chosen) >= max_baseline_candidates:
            break
        cost = _token_count(tokenizer, text_for_candidate(i, item, 1))
        if used + cost <= evidence_budget:
            chosen[i] = 1
            used += cost
    if not chosen:
        raise ValueError("no baseline candidate can fit")
    anchors = failure_anchors(items, source["context"])
    candidates = [i for i in anchors if i in chosen] + [
        i for i in sorted(chosen) if i not in anchors]
    eligible = []
    for i in candidates:
        old = _token_count(tokenizer, text_for_candidate(i, items[i], 1))
        new = _token_count(tokenizer, text_for_candidate(i, items[i], 2))
        if new > old and used + new - old <= evidence_budget:
            eligible.append((i, new - old))
    if not eligible:
        return {
            "skip_reason": "no_positive_cost_L1_upgrade_fits_same_budget",
            "chosen_ids": [items[i]["function_id"] for i in sorted(chosen)],
        }
    upgrade_i, delta = eligible[0]
    after = dict(chosen)
    after[upgrade_i] = 2
    prompts = []
    for name, levels in (("STOP", chosen), ("UPGRADE_L1", after)):
        messages = build_messages(source, levels)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("tokenizer must support chat templates")
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        length = _token_count(tokenizer, prompt)
        if length > max_total_input:
            return {
                "skip_reason": "full_prompt_exceeds_matched_input_ceiling",
                "condition": name, "full_input_tokens": length,
            }
        prompts.append({
            "condition": name, "prompt": prompt,
            "prompt_sha256": _digest(prompt),
            "actual_total_input_tokens": length,
            "selected_levels": {str(i): levels[i] for i in sorted(levels)},
        })
    return {
        "skip_reason": None, "prompts": prompts,
        "target_method": items[upgrade_i]["function_id"],
        "upgrade_index": upgrade_i, "incremental_evidence_tokens": delta,
        "evidence_budget": evidence_budget,\n        "initial_signature_candidate_cap": max_baseline_candidates,
        "common_total_input_ceiling": max_total_input,
        "candidate_id_by_index": {str(i): items[i]["function_id"] for i in chosen},
        "frozen_candidate_ids": [items[i]["function_id"] for i in chosen],
        "failure_text_sha256": _digest(source["context"]),
    }


def score_pair(pair: dict, gold_methods: list[str]) -> dict:
    """Posthoc only. Changed source methods are imperfect causal labels."""
    if pair.get("skip_reason"):
        return {"bug": pair.get("bug"), "skipped": True,
                "reason": pair["skip_reason"]}
    raw = pair["predictions"]
    if [v["condition"] for v in raw] != ["STOP", "UPGRADE_L1"]:
        raise ValueError("expected frozen STOP and action outputs")
    gold = set(gold_methods)
    initial = raw[0]["predicted_method"] in gold
    upgraded = raw[1]["predicted_method"] in gold
    return {
        "bug": pair["bug"], "skipped": False,
        "stop_correct": initial, "upgraded_correct": upgraded,
        "outcome": int(upgraded) - int(initial),
        "input_tokens_stop": raw[0]["actual_total_input_tokens"],
        "input_tokens_upgrade": raw[1]["actual_total_input_tokens"],
        "model_latency_seconds_stop": raw[0]["generation_seconds"],
        "model_latency_seconds_upgrade": raw[1]["generation_seconds"],
        "changed_method_gold_is_imperfect_causal_label": True,
        "prior_studied_development_bug": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bugs", nargs="+", default=["Lang-1", "Time-25"])
    p.add_argument("--evidence-budget", type=int, default=650)
    p.add_argument("--max-input-tokens", type=int, default=1800)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(42)
    revision = HfApi().model_info(MODEL_ID).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=torch.float32)
    model.eval()

    frozen = {}
    for bug in args.bugs:
        if bug not in BUG_INPUTS:
            raise ValueError(f"unknown bug {bug}")
        data = BUG_INPUTS[bug].read_bytes()
        spec = freeze_upgrade_pair(
            json.loads(data), tokenizer, evidence_budget=args.evidence_budget,
            max_total_input=args.max_input_tokens)
        spec["bug"] = bug
        spec["source_sha256"] = hashlib.sha256(data).hexdigest()
        frozen[bug] = spec

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_pair_manifest.json").write_text(
        json.dumps({
            bug: {k: v if k != "prompts" else [
                {x: y for x, y in prompt.items() if x != "prompt"}
                for prompt in v] for k, v in spec.items()}
            for bug, spec in frozen.items()
        }, indent=2) + "\n")
    (args.output_dir / "frozen_prompts.json").write_text(
        json.dumps({
            bug: [dict(row) for row in spec.get("prompts", [])]
            for bug, spec in frozen.items()
        }, indent=2) + "\n")

    records = []
    for bug, spec in frozen.items():
        item = {"bug": bug, "skip_reason": spec["skip_reason"],
                "source_sha256": spec["source_sha256"],
                "model_id": MODEL_ID, "model_revision": revision,
                "upgrade_index": spec.get("upgrade_index"),
                "target_method": spec.get("target_method"),
                "predictions": []}
        if not spec["skip_reason"]:
            for condition in spec["prompts"]:
                prompt = condition["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt",
                                   add_special_tokens=False)
                assert inputs["input_ids"].shape[1] <= args.max_input_tokens
                started = time.perf_counter()
                with torch.no_grad():
                    result = model.generate(
                        **inputs, do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id)
                seconds = time.perf_counter() - started
                completion = result[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(completion,
                                             skip_special_tokens=True)
                levels = {int(k): int(v) for k, v in
                          condition["selected_levels"].items()}
                prediction = parse_prediction(response, levels)
                item["predictions"].append({
                    "condition": condition["condition"],
                    "prompt_sha256": condition["prompt_sha256"],
                    "actual_total_input_tokens": condition["actual_total_input_tokens"],
                    "response": response,
                    "output_tokens": len(completion),
                    "predicted_method": (spec["candidate_id_by_index"][str(prediction)]
                                         if prediction is not None else None),
                    "generation_seconds": seconds,
                })
                print(bug, condition["condition"], "prediction",
                      item["predictions"][-1]["predicted_method"], flush=True)
        records.append(item)

    # STRICT BARRIER: save predictions before reading official patch methods.
    (args.output_dir / "raw_pair_predictions.json").write_text(
        json.dumps(records, indent=2) + "\n")
    label_path = Path("data/aitech/three_bug_posthoc_patch_labels.json")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    if labels.get("scoring_only") is not True:
        raise ValueError("patch labels must be evaluation-only")
    results = [score_pair(row, labels["bugs"][row["bug"]]["changed_methods"])
               for row in records]
    (args.output_dir / "posthoc_action_outcomes.json").write_text(
        json.dumps({
            "model_executed": True,
            "split": "previously_studied_development_only",
            "not_usable_as_independent_project_test": True,
            "pairs": results,
        }, indent=2) + "\n")
    print("Saved raw prompts, predictions and posthoc outcome scores")


if __name__ == "__main__":
    main()

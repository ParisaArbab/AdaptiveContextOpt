"""CPU-only paired LLM fault-localization feasibility experiment.

Uses a small public code model, actual tokenizer, fixed failure context,
and three paired evidence selection policies. No paid inference API, GPU,
AIScalar, or gold labels during selection/generation.

IMPORTANT: The model must choose only among displayed evidence candidates.
This evaluates constrained code localization, not free-form repository search.
All bugs were previously studied; not a clean held-out test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from .failure_conditioned_study import BUG_INPUTS
from .multiresolution_selector import select_tiered
from .tiered_evidence_pilot import select_tiers

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
POLICIES = ("rank_fill", "anchor_rank_detail", "tiered_marginal")
PROMPT_INSTRUCTION = (
    "You are locating a software bug using a failed test and candidate Java methods. "
    "Choose the ONE most suspicious method from the provided candidates. "
    "Reply with the candidate number only, for example 2. "
    "Do not invent new candidates or explain."
)


def text_for_candidate(index: int, item: dict, level: int) -> str:
    if level not in (1, 2):
        raise ValueError("level must be L0=1 or L1=2")
    content = item["levels"][level - 1]
    return f"[{index + 1}] {content}\n"


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def with_real_costs(snapshot: dict, tokenizer) -> dict:
    """Replace stored sketch proxies with actual exact formatted tokenizer costs."""
    copied = dict(snapshot)
    new_items = []
    for i, item in enumerate(snapshot["candidates"]):
        clone = dict(item)
        features = dict(item["features"])
        l0 = _token_count(tokenizer, text_for_candidate(i, item, 1))
        l1 = _token_count(tokenizer, text_for_candidate(i, item, 2))
        if l0 < 1 or l1 < 1:
            raise ValueError("invalid tokenizer estimate")
        features["l0_tokens"] = l0
        # Report exact L1 token counts separately. For packing, conservatively
        # charge at least the L0 cost to any L1 structural representation.
        features["l1_tokens"] = max(l0, l1)
        clone["features"] = features
        new_items.append(clone)
    copied["candidates"] = new_items
    return copied


def choose_evidence(snapshot: dict, budget: int, policy: str) -> dict[int, int]:
    """Map original candidate index to 1=L0 or 2=L1."""
    items = snapshot["candidates"]
    if policy == "rank_fill":
        used, result = 0, {}
        for i, item in enumerate(items):
            cost = int(item["features"]["l1_tokens"])
            if used + cost <= budget:
                result[i] = 2
                used += cost
        return result
    if policy == "anchor_rank_detail":
        tier = select_tiers(snapshot, budget, "anchor_rank_detail")
        names = {id_: i for i, id_ in enumerate(x["function_id"] for x in items)}
        return {names[name]: 2 if level == "L1" else 1
                for name, level in tier["candidate_id_tier"].items()}
    if policy == "tiered_marginal":
        chosen = select_tiered(items, snapshot["context"], budget)
        return {i: level for i, level in enumerate(chosen.levels) if level}
    raise ValueError("unknown selection policy")


def build_messages(snapshot: dict, selected: dict[int, int]) -> list[dict]:
    if not selected:
        raise ValueError("no evidence selected")
    evidence = "".join(
        text_for_candidate(i, snapshot["candidates"][i], selected[i])
        for i in sorted(selected)
    )
    body = (
        "FAILED TEST AND ERROR:\n" + snapshot["context"] +
        "\n\nCANDIDATE JAVA CODE:\n" + evidence +
        "\nAnswer with only the number of one displayed method."
    )
    return [{"role": "system", "content": PROMPT_INSTRUCTION},
            {"role": "user", "content": body}]


def parse_prediction(reply: str, selected: dict[int, int]) -> int | None:
    """Parse an unambiguous numeric choice, or the previous INDEX: N form.

    Some small code models return bare or bracketed integers despite format instructions.
    Accept only a complete, single numeric response. Never guess from free
    prose, a malformed answer, or an unlisted candidate number.
    """
    match = re.fullmatch(r"\s*(?:INDEX\s*:\s*)?\[?(\d+)\]?\s*", reply, flags=re.I)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if index in selected else None


def freeze_selection(snapshots: dict[str, dict], tokenizer,
                     evidence_budget: int) -> list[dict]:
    frozen = []
    for bug, record in snapshots.items():
        source = with_real_costs(record["source"], tokenizer)
        items = source["candidates"]
        for policy in POLICIES:
            chosen = choose_evidence(source, evidence_budget, policy)
            candidate_text = "".join(
                text_for_candidate(i, items[i], chosen[i]) for i in sorted(chosen)
            )
            actual_evidence_tokens = _token_count(tokenizer, candidate_text)
            if actual_evidence_tokens > evidence_budget:
                raise AssertionError(
                    f"actual evidence tokens exceeded budget for {bug}/{policy}: "
                    f"{actual_evidence_tokens} > {evidence_budget}"
                )
            messages = build_messages(source, chosen)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            else:
                raise ValueError("tokenizer requires a chat template")
            frozen.append({
                "bug": bug,
                "snapshot_sha256": record["sha256"],
                "policy": policy,
                "selected_indices": sorted(chosen),
                "selected_levels": {str(i): chosen[i] for i in sorted(chosen)},
                "selected_ids": [items[i]["function_id"] for i in sorted(chosen)],
                "actual_evidence_tokens": actual_evidence_tokens,
                "actual_total_input_tokens": _token_count(tokenizer, prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "_prompt": prompt,
            })
    return frozen


def score_frozen_predictions(predictions: list[dict], labels: dict) -> dict:
    if labels.get("scoring_only") is not True:
        raise ValueError("gold labels must be scoring-only")
    results = []
    for record in predictions:
        gold = set(labels["bugs"][record["bug"]]["changed_methods"])
        pred = record.get("predicted_method")
        present = set(record["selected_ids"]) & gold
        detailed = {
            record["candidate_id_by_index"][str(i)]
            for i, level in record["selected_levels"].items() if level == 2
        } & gold
        results.append({
            "bug": record["bug"], "policy": record["policy"],
            "gold_methods": sorted(gold),
            "predicted_method": pred,
            "predicted_gold_changed_method": pred in gold if pred else False,
            "parse_valid": pred is not None,
            "changed_methods_in_context": len(present),
            "changed_methods_with_L1": len(detailed),
            "gold_total": len(gold),
            "actual_input_tokens": record["actual_total_input_tokens"],
            "actual_evidence_tokens": record["actual_evidence_tokens"],
            "generation_seconds": record["generation_seconds"],
        })
    return {
        "scope": "Small CPU feasibility run, constrained candidate-choice localization",
        "model_inference_performed": True,
        "gold_read_only_after_predictions_frozen": True,
        "not_unseen_test_data": True,
        "results": results,
        "limitations": [
            "Previously studied bugs; cannot claim independent generalization",
            "0.5B model may be too weak; negative result is a feasibility signal",
            "Model forced to choose one of retrieved candidates, not free repository search",
            "Actual output was measured on CPU and model tokenizer, not paid APIs",
            "Changed method is imperfect fault causality label",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bugs", nargs="+", default=["Lang-1", "Time-25"])
    p.add_argument("--budget", type=int, default=500)
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
    snapshots = {}
    for bug in args.bugs:
        if bug not in BUG_INPUTS:
            raise ValueError(f"unknown bug: {bug}")
        data = BUG_INPUTS[bug].read_bytes()
        snapshots[bug] = {
            "source": json.loads(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    frozen = freeze_selection(snapshots, tokenizer, args.budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = [{k: v for k, v in row.items() if k != "_prompt"} for row in frozen]
    (args.output_dir / "frozen_selections.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8")

    predictions = []
    for record in frozen:
        prompt = record["_prompt"]
        # Chat template already includes special tokens; avoid duplicate BOS/EOS.
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] > args.max_input_tokens:
            raise RuntimeError(f"model input too long: {record['bug']}/{record['policy']}")
        started = time.perf_counter()
        with torch.no_grad():
            result = model.generate(
                **inputs, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion_ids = result[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(completion_ids, skip_special_tokens=True)
        chosen = {int(k): int(v) for k, v in record["selected_levels"].items()}
        predicted_index = parse_prediction(reply, chosen)
        items = snapshots[record["bug"]]["source"]["candidates"]
        predictions.append({
            **{k: v for k, v in record.items() if k != "_prompt"},
            "model_id": MODEL_ID, "model_revision_sha": revision,
            "response": reply, "predicted_index": predicted_index,
            "predicted_method": (
                items[predicted_index]["function_id"] if predicted_index is not None
                else None
            ),
            "candidate_id_by_index": {
                str(i): items[i]["function_id"] for i in chosen
            },
            "generation_seconds": elapsed,
            "output_tokens": len(completion_ids),
        })
        print(record["bug"], record["policy"],
              "selected", len(chosen),
              "input", inputs["input_ids"].shape[1],
              "predicted", predictions[-1]["predicted_method"],
              "seconds", round(elapsed, 2), flush=True)

    # Persist predictions before opening gold labels. Do not let labels
    # influence source selection or model generation.
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    labels = json.loads(Path(
        "data/aitech/three_bug_posthoc_patch_labels.json"
    ).read_text(encoding="utf-8"))
    scored = score_frozen_predictions(predictions, labels)
    (args.output_dir / "scores.json").write_text(
        json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print("Saved predictions and independent posthoc scoring", args.output_dir)


if __name__ == "__main__":
    main()

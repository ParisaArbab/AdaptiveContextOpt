"""One-bug, CPU-only, real LLM localization comparison with actual model tokens.

Uses Qwen/Qwen2.5-Coder-0.5B-Instruct under fixed full-chat token budgets.
Two evidence conditions: anchor-first and forced historically verified
support. The unverified-binding gate reuses anchor-first exactly.

IMPORTANT: This script never loads any patch labels. Evaluate those
separately only after predictions/prompts are frozen on disk.
The 0.5B model is an exploratory feasibility baseline, not SOTA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from .time1_verified_retrieval_study import SNAPSHOT, SOURCE_RECORD

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SYSTEM = (
    "You are a Java fault-localization assistant. Examine the failing "
    "test evidence and candidate buggy-source method excerpts. "
    "Reply with exactly ONE METHOD ID from the candidate list. "
    "Do not output source explanations or invent methods."
)
BUDGETS = (1024, 1536)


def _name(identifier: str) -> str:
    return identifier.split("(", 1)[0].rsplit(".", 1)[-1]


def messages(snapshot: dict, chosen: list[int], helper: dict | None) -> list[dict]:
    candidates = snapshot["candidates"]
    sections = [
        "Failing Java test and stack trace:\n" + snapshot["context"],
        "Select ONE of these METHOD IDs (only these are eligible):",
    ]
    for i in chosen:
        item = candidates[i]
        sections.append(
            "METHOD ID: " + item["function_id"] +
            "\nSOURCE SKETCH:\n" + item["levels"][1]
        )
    if helper is not None:
        sections.append(
            "SUPPORTING DEFINITION, NOT AN ELIGIBLE METHOD ID:\n"
            + helper["target"] + "\n" + helper["method_source"]
        )
    sections.append("Answer with one exact METHOD ID from the eligible methods above.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(sections)}]


def select_prompt(
    tokenizer, snapshot: dict, helper: dict, *,
    max_input_tokens: int, force_helper: bool,
) -> dict:
    """Greedy selection, measuring the ENTIRE formatted chat at every step."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    candidates = snapshot["candidates"]
    anchors = [
        i for i, item in enumerate(candidates)
        if _name(item["function_id"]) + "(" in snapshot["context"]
    ]
    chosen: list[int] = []
    support = None

    def count(candidate_idxs, supporting):
        chat = messages(snapshot, candidate_idxs, supporting)
        return len(tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
        ))

    if count([], None) > max_input_tokens:
        raise ValueError("failure evidence and instructions exceed input budget")
    for i in anchors:
        if i not in chosen and count(chosen + [i], support) <= max_input_tokens:
            chosen.append(i)
    if force_helper and count(chosen, helper) <= max_input_tokens:
        support = helper
    for i in range(len(candidates)):
        if i not in chosen and count(chosen + [i], support) <= max_input_tokens:
            chosen.append(i)
    chat = messages(snapshot, chosen, support)
    total = count(chosen, support)
    if total > max_input_tokens:
        raise AssertionError("prompt budget exceeded")
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)
    return {
        "input_token_count": total,
        "max_input_tokens": max_input_tokens,
        "source_helper_included": support is not None,
        "failure_named_candidates": [
            candidates[i]["function_id"] for i in anchors
        ],
        "selected_candidate_ids": [
            candidates[i]["function_id"] for i in chosen
        ],
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "messages": chat,
    }


def _extract_prediction(text: str, eligible: list[str]) -> str | None:
    # Parse ONLY an ID from eligible candidates, abstain otherwise.
    matches = [(text.find(identifier), identifier)
               for identifier in eligible if identifier in text]
    matches = [x for x in matches if x[0] >= 0]
    return min(matches)[1] if matches else None


def run_inference(
    *, output: Path, budgets: tuple[int, ...] = BUDGETS,
    model_id: str = MODEL_ID, max_new_tokens: int = 72,
) -> dict:
    # Do not import external compute dependencies during regular unit tests.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32).eval().to("cpu")
    revision_sha = HfApi().model_info(model_id).sha
    load_time = time.perf_counter() - started

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    helper = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    if helper["buggy_revision"] != "8612f9e5b88c1bea933ef9ab1e431f5db3006b48":
        raise ValueError("historical buggy source changed")
    if helper["compiler_binding_verified"] is not False:
        raise ValueError("binding provenance is no longer unverified")

    results = []
    cached_predictions = {}
    for budget in budgets:
        for condition in ("anchor_first", "force_verified_source",
                          "gate_unverified_binding"):
            evidence = select_prompt(
                tokenizer, snapshot, helper, max_input_tokens=budget,
                force_helper=condition == "force_verified_source")
            prompt_sha = evidence["prompt_sha256"]
            if prompt_sha in cached_predictions:
                prediction = dict(cached_predictions[prompt_sha])
                prediction["reused_identical_prompt"] = True
                prediction["inference_seconds"] = 0.0
            else:
                prompt_ids = tokenizer.apply_chat_template(
                    evidence["messages"], tokenize=True,
                    add_generation_prompt=True, return_tensors="pt")
                count = prompt_ids.shape[-1]
                if count != evidence["input_token_count"]:
                    raise AssertionError("token count mismatch")
                start = time.perf_counter()
                with torch.inference_mode():
                    response = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=torch.ones_like(prompt_ids),
                        max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                seconds = time.perf_counter() - start
                generated = response[0][count:]
                answer = tokenizer.decode(generated, skip_special_tokens=True)
                prediction = {
                    "model_output": answer,
                    "predicted_candidate_id": _extract_prediction(
                        answer, evidence["selected_candidate_ids"]),
                    "output_token_count": generated.shape[-1],
                    "inference_seconds": round(seconds, 3),
                    "reused_identical_prompt": False,
                }
                cached_predictions[prompt_sha] = dict(prediction)
            results.append({
                "condition": condition,
                "max_input_tokens": budget,
                **evidence,
                **prediction,
            })
            # Freeze raw predictions before patch scoring.
            output.parent.mkdir(parents=True, exist_ok=True)
            interim = {
                "scope": "one previously studied real Time-1 bug, CPU LLM feasibility",
                "model_id": model_id,
                "model_revision_sha": revision_sha,
                "model_load_seconds": round(load_time, 3),
                "seed": 42, "decoding": "greedy",
                "gold_patch_loaded": False,
                "buggy_revision": helper["buggy_revision"],
                "rows": results,
            }
            output.write_text(json.dumps(interim, indent=2) + "\n",
                              encoding="utf-8")
    return interim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/time1_cpu_llm_predictions.json"))
    parser.add_argument("--max-new-tokens", type=int, default=72)
    args = parser.parse_args()
    result = run_inference(output=args.output,
                           max_new_tokens=args.max_new_tokens)
    for row in result["rows"]:
        print(f"{row['max_input_tokens']} {row['condition']}: "
              f"prompt={row['input_token_count']} tokens, "
              f"helper={row['source_helper_included']}, "
              f"prediction={row['predicted_candidate_id']}, "
              f"inference={row['inference_seconds']} seconds")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

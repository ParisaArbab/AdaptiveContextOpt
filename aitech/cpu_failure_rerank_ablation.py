"""Compare old candidate ordering, failure-based reranking and reversed order.

This *one-shot development ablation* uses archived pre-gold buggy-project
snapshots from a previously completed GitHub Actions real Defects4J
run. All prompts are frozen and saved before modified-class scoring
records are opened. The class labels are read from a separate posthoc
artifact after all raw predictions are persisted.

No future benchmark project is part of this exploratory experiment.
No calibration or trained policy is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cpu_llm_localization import (
    MODEL_ID, _token_count, parse_prediction, with_real_costs,
)
from .failure_aligned_reranker import failure_rerank, shortlist
from .cpu_live_defects4j_pairs import _predicted_class

CONDITIONS = ("original_top8", "failure_rerank", "failure_rerank_reversed")


def build_frozen_prompts(snapshot: dict, tokenizer, *,
                         max_input_tokens: int = 1800, limit: int = 8) -> dict:
    """Use the same underlying bug, source data and model input ceiling."""
    if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
        raise ValueError("only genuine pre-gold buggy-state snapshots supported")
    normalized = with_real_costs(snapshot, tokenizer)
    items = normalized["candidates"]
    if len(items) < 2:
        raise ValueError("at least two candidate methods needed")
    order_old = list(range(min(limit, len(items))))
    order_ranked = shortlist(normalized, limit=limit)
    orders = {
        "original_top8": order_old,
        "failure_rerank": order_ranked,
        "failure_rerank_reversed": list(reversed(order_ranked)),
    }
    frozen = {}
    for condition, indices in orders.items():
        # All rows share one failure text. Each row is an L0-only
        # candidate choice, and prompt position is newly numbered.
        def make_messages(candidate_indices: list[int]):
            choices = "\n".join(
                f"[{pos}] " + items[i]["levels"][0]
                for pos, i in enumerate(candidate_indices, 1)
            )
            return [
                {"role": "system",
                 "content": (
                     "Given a failed Java test and candidate methods, "
                     "identify the one method most likely to contain the bug. "
                     "Return only one candidate number, such as 2."
                 )},
                {"role": "user",
                 "content": (
                     "FAILED TEST AND STACK TRACE:\n" +
                     normalized["context"] +
                     "\n\nCANDIDATE METHODS:\n" + choices +
                     "\n\nSelect only one numbered candidate."
                 )},
            ]
        current = list(indices)
        # Budget safety: shrink list BEFORE any model inference, but
        # preserve the best prefix for each condition.
        while len(current) >= 2:
            prompt = tokenizer.apply_chat_template(
                make_messages(current), tokenize=False,
                add_generation_prompt=True)
            count = _token_count(tokenizer, prompt)
            if count <= max_input_tokens:
                break
            current.pop()
        else:
            frozen[condition] = {
                "skip_reason": "no_two_methods_fit_total_input_ceiling",
                "candidate_indices": [], "prompt": None,
            }
            continue
        frozen[condition] = {
            "skip_reason": None,
            "candidate_indices": current,
            "candidate_ids": [items[i]["function_id"] for i in current],
            "candidate_positions": {
                str(pos): items[i]["function_id"]
                for pos, i in enumerate(current, 1)
            },
            "actual_input_tokens": count,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
        }
    return {
        "conditions": frozen, "ranking": failure_rerank(normalized),
        "failure_sha256": normalized["provenance"]["failure_sha256"],
        "model_input_ceiling": max_input_tokens,
        "no_gold_used": True,
    }


def score_predictions(raw: list[dict], posthoc: dict) -> dict:
    """Score class Hit@1 after freezing all raw model replies."""
    scores = {r["bug"]: set(r["gold_modified_classes"])
              for r in posthoc["results"] if not r.get("skipped")}
    output = []
    for record in raw:
        if record["bug"] not in scores:
            raise ValueError("missing original posthoc modified-class labels")
        for result in record["predictions"]:
            hit = _predicted_class(result["predicted_method"]) in scores[record["bug"]]
            output.append({
                "bug": record["bug"], "condition": result["condition"],
                "parse_valid": result["predicted_method"] is not None,
                "changed_class_hit_at_1": hit,
                "predicted_method": result["predicted_method"],
                "actual_input_tokens": result["actual_input_tokens"],
                "output_tokens": result["output_tokens"],
                "generation_seconds": result["generation_seconds"],
                "candidate_count": len(result["candidate_ids"]),
            })
    return {
        "scope": "Previously examined Math-1 and Chart-1 development-only",
        "model_inference_performed": True,
        "prediction_frozen_before_posthoc_scoring": True,
        "method_is_pre_golden_regex_source_candidate": True,
        "results": output,
        "limitations": [
            "Very small CPU Qwen model; this is not a heldout evaluation",
            "Candidate class Hit@1 is not causal root cause",
            "Reversing only one candidate order cannot establish position invariance",
            "Context and method identity format are kept constant across arms",
        ],
    }


def run(snapshots: dict[str, Path], posthoc_path: Path, output_dir: Path,
        max_input_tokens: int = 1800) -> None:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_frozen = {}
    for bug, path in snapshots.items():
        source = path.read_bytes()
        pre_frozen[bug] = build_frozen_prompts(
            json.loads(source), tokenizer, max_input_tokens=max_input_tokens)
        pre_frozen[bug]["source_sha256"] = hashlib.sha256(source).hexdigest()
    # Save all candidate ordering and full prompts BEFORE accessing labels.
    (output_dir / "frozen_prompts.json").write_text(
        json.dumps(pre_frozen, indent=2) + "\n", encoding="utf-8")

    raw = []
    for bug, spec in pre_frozen.items():
        record = {"bug": bug, "model_id": MODEL_ID, "revision": revision,
                  "predictions": []}
        for condition, row in spec["conditions"].items():
            if row["skip_reason"]:
                print(bug, condition, "SKIP", row["skip_reason"], flush=True)
                continue
            prompt = row["prompt"]
            encoded = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False)
            actual = encoded["input_ids"].shape[1]
            if actual != row["actual_input_tokens"]:
                raise ValueError("pre-frozen tokens do not match model encoding")
            started = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **encoded, max_new_tokens=24, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            elapsed = time.perf_counter() - started
            completion = out[0][actual:]
            response = tokenizer.decode(completion, skip_special_tokens=True)
            selection = {i: 1 for i in range(len(row["candidate_ids"]))}
            predicted = parse_prediction(response, selection)
            record["predictions"].append({
                "condition": condition,
                "prompt_sha256": row["prompt_sha256"],
                "candidate_ids": row["candidate_ids"],
                "response": response,
                "predicted_method": (
                    row["candidate_ids"][predicted] if predicted is not None
                    else None),
                "actual_input_tokens": actual,
                "output_tokens": len(completion),
                "generation_seconds": elapsed,
            })
            print(bug, condition, "response", repr(response),
                  "prediction", record["predictions"][-1]["predicted_method"],
                  flush=True)
        raw.append(record)
    (output_dir / "raw_predictions_before_gold.json").write_text(
        json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    # Golden labels are opened ONLY AFTER all model predictions are written.
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    scored = score_predictions(raw, posthoc)
    (output_dir / "posthoc_scores.json").write_text(
        json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print("Saved", len(scored["results"]), "real-model ranking ablation predictions")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="append", nargs=2,
                   metavar=("BUG", "PATH"), required=True)
    p.add_argument("--posthoc", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run({k: Path(v) for k, v in args.snapshot},
        args.posthoc, args.output_dir)


if __name__ == "__main__":
    main()

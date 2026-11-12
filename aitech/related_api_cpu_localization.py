"""CPU same-model localization test for source-grounded related-API retrieval.

Predeclared arms on identical pre-gold 25-method candidate pool:
- stack_L0: stack + BM25 top 8, L0 identities
- related_L0: related API class expansion, top 8, L0 identities
- related_explain_L0: same 8 methods with general cause-vs-symptom hint
- related_reverse_L0: same 8 related methods in reverse order

All arms use the SAME pinned model, 8 method identities and max total
input budget. The causality note is a hypothesis, not ground truth.
Source selections and raw responses are frozen BEFORE reading gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cpu_llm_localization import _token_count, build_messages, parse_prediction
from .failure_grounded_rerank import rerank, shortlist
from .related_api_retrieval import related_shortlist

ARMS = ("stack_L0", "related_L0", "related_explain_L0", "related_reverse_L0")
MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def freeze(snapshot: dict, tokenizer, max_input_tokens: int = 1800) -> list[dict]:
    if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
        raise ValueError("requires frozen pre-gold source candidates")
    related = related_shortlist(snapshot, count=8)
    grounded = rerank(snapshot)[:8]
    orders = {
        "stack_L0": grounded,
        "related_L0": list(related.original_indices),
        "related_explain_L0": list(related.original_indices),
        "related_reverse_L0": list(reversed(related.original_indices)),
    }
    frozen = []
    for arm in ARMS:
        local = shortlist(snapshot, orders[arm], count=8)
        messages = build_messages(local, {i: 1 for i in range(8)})
        if arm == "related_explain_L0":
            messages[0]["content"] += (
                " A stack frame may show where a failure was detected,"
                " not which component caused it. Consider both producers"
                " and consumers of data. Do not simply choose the first"
                " item or the first stack frame.")
            pairs = related.related_class_proposals
            if pairs:
                notes = "; ".join(
                    f"{a} and {b} are related API names (unverified)"
                    for a, b in pairs)
                messages[1]["content"] += (
                    "\nPOTENTIAL RELATED SOURCE CLASSES: " + notes)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tokens = _token_count(tokenizer, prompt)
        if tokens > max_input_tokens:
            raise ValueError(f"{arm} exceeds common model-input ceiling")
        frozen.append({
            "arm": arm, "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "prompt": prompt, "actual_input_tokens": tokens,
            "selected_method_ids": [
                item["function_id"] for item in local["candidates"]],
            "related_api_proposals": [
                list(x) for x in related.related_class_proposals],
            "original_candidate_indices": orders[arm],
        })
    return frozen


def score(raw: list[dict], labels: dict[str, set[str]]) -> dict:
    output = []
    for record in raw:
        gold = labels[record["bug"]]
        arms = {}
        for item in record["arms"]:
            prediction = item["predicted_method"]
            owner = prediction.split("(", 1)[0].rsplit(".", 1)[0] if prediction else None
            arms[item["arm"]] = {
                "class_hit_at_1": owner in gold,
                "predicted_class": owner,
                "predicted_method": prediction,
                "actual_input_tokens": item["actual_input_tokens"],
                "generation_seconds": item["generation_seconds"],
                "parse_valid": prediction is not None,
            }
        output.append({
            "bug": record["bug"],
            "changed_class_covered_by_related_8": bool(
                {x.split("(", 1)[0].rsplit(".", 1)[0]
                 for x in record["arms"][1]["selected_method_ids"]} & gold),
            "arms": arms,
        })
    return {
        "scope": "Previously examined Codec/Compress development bugs",
        "ground_truth": "Defects4J modified classes, not proven fault causes",
        "no_independent_test_claim": True,
        "raw_responses_saved_before_labels": True,
        "results": output,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", nargs=2, action="append", required=True,
                   metavar=("BUG", "SOURCE_JSON"))
    p.add_argument("--gold", nargs=2, action="append", required=True,
                   metavar=("BUG", "MODIFIED_CLASSES"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-input-tokens", type=int, default=1800)
    args = p.parse_args()
    snapshots = dict(args.snapshot)
    gold_files = dict(args.gold)
    if len(snapshots) != len(args.snapshot) or set(snapshots) != set(gold_files):
        raise ValueError("mismatched or duplicate bugs")

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(42)
    revision = HfApi().model_info(args.model).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.float32)
    model.eval()

    frozen = {}
    for bug, source_path in snapshots.items():
        source = json.loads(Path(source_path).read_text())
        frozen[bug] = freeze(source, tokenizer, args.max_input_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_prompts_before_gold.json").write_text(
        json.dumps(frozen, indent=2) + "\n")

    raw = []
    for bug, items in frozen.items():
        report = {"bug": bug, "model_id": args.model,
                  "model_revision": revision, "arms": []}
        for arm in items:
            inputs = tokenizer(
                arm["prompt"], return_tensors="pt",
                add_special_tokens=False)
            input_len = int(inputs["input_ids"].shape[1])
            if input_len > args.max_input_tokens:
                raise ValueError("model prompt exceeded common budget")
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=24,
                    pad_token_id=tokenizer.eos_token_id)
            secs = time.perf_counter() - started
            response = tokenizer.decode(
                generated[0][input_len:], skip_special_tokens=True)
            chosen = parse_prediction(response, {i: 1 for i in range(8)})
            item = {k: v for k, v in arm.items() if k != "prompt"}
            item.update({
                "raw_response": response,
                "predicted_method": (
                    arm["selected_method_ids"][chosen]
                    if chosen is not None else None),
                "generation_seconds": secs,
                "actual_input_tokens": input_len,
                "output_tokens": int(len(generated[0]) - input_len),
            })
            report["arms"].append(item)
            print(bug, arm["arm"], "reply", repr(response),
                  "method", item["predicted_method"], flush=True)
        raw.append(report)

    (args.output_dir / "raw_predictions_before_gold.json").write_text(
        json.dumps(raw, indent=2) + "\n")
    labels = {
        bug: set(Path(file).read_text().split())
        for bug, file in gold_files.items()
    }
    scored = score(raw, labels)
    scored["model_id"] = args.model
    scored["model_revision"] = revision
    (args.output_dir / "score_after_gold.json").write_text(
        json.dumps(scored, indent=2) + "\n")
    print("Saved audited same-model related API source recall and localization")


if __name__ == "__main__":
    main()

"""Matched CPU ablation of failure-grounded retrieval and order robustness.

Four frozen evidence conditions on the SAME buggy project:
  original_first8: existing ranked first 8 L0 signatures
  grounded_first8: stack identity + BM25 lexical reranking, 8 L0
  grounded_reversed: SAME eight signatures in reverse order (position test)
  grounded_one_L1: same eight methods, upgrade one when budget allows

The model and total input-token ceiling are fixed across arms. Actual
token counts, latency and raw replies are logged BEFORE gold labels.
Gold is the official changed-class proxy from a PREVIOUS run, used
strictly for *posthoc* scoring. These are studied development bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .cpu_llm_localization import (
    _token_count, build_messages, parse_prediction, text_for_candidate
)
from .failure_grounded_rerank import rerank, shortlist
from .grounded_stop_guard import decide_pre_gold

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ARMS = ("original_first8", "grounded_first8",
        "grounded_reversed", "grounded_one_L1")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_arms(snapshot: dict, tokenizer, *, count: int = 8,
                max_input_tokens: int = 1800) -> list[dict]:
    items = snapshot["candidates"]
    if len(items) < count:
        raise ValueError("not enough candidates for matched comparison")
    normal = list(range(len(items)))
    grounded = rerank(snapshot)
    selections = {
        "original_first8": normal[:count],
        "grounded_first8": grounded[:count],
        "grounded_reversed": list(reversed(grounded[:count])),
        "grounded_one_L1": grounded[:count],
    }
    frozen = []
    for arm in ARMS:
        local = shortlist(snapshot, selections[arm], count=count)
        tiers = {i: 1 for i in range(count)}
        if arm == "grounded_one_L1":
            # Choose candidate 0 by pre-gold reranking. If it does not
            # fit, try the remaining ranked targets. If no upgrade
            # fits, record skipped, NEVER call it an outcome.
            eligible = []
            for i in range(count):
                levels = dict(tiers)
                levels[i] = 2
                prompt = tokenizer.apply_chat_template(
                    build_messages(local, levels), tokenize=False,
                    add_generation_prompt=True)
                if _token_count(tokenizer, prompt) <= max_input_tokens:
                    l0 = _token_count(tokenizer, text_for_candidate(
                        i, local["candidates"][i], 1))
                    l1 = _token_count(tokenizer, text_for_candidate(
                        i, local["candidates"][i], 2))
                    if l1 > l0:
                        eligible.append(i)
            if eligible:
                tiers[eligible[0]] = 2
            else:
                frozen.append({
                    "arm": arm, "skip_reason": "no_L1_upgrade_fits",
                    "selected_method_ids": [
                        c["function_id"] for c in local["candidates"]
                    ],
                })
                continue
        prompt = tokenizer.apply_chat_template(
            build_messages(local, tiers), tokenize=False,
            add_generation_prompt=True)
        tokens = _token_count(tokenizer, prompt)
        if tokens > max_input_tokens:
            raise ValueError(f"full prompt exceeds ceiling for {arm}: {tokens}")
        frozen.append({
            "arm": arm,
            "skip_reason": None,
            "prompt": prompt,
            "prompt_sha256": sha(prompt),
            "actual_input_tokens": tokens,
            "selected_levels": tiers,
            "selected_method_ids": [
                item["function_id"] for item in local["candidates"]],
            "original_candidate_indices": selections[arm],
            "source_sha256": snapshot["provenance"]["failure_sha256"],
        })
    return frozen


def score_arms(raw: list[dict], changed_classes: dict[str, set[str]]) -> dict:
    results = []
    for item in raw:
        gold = changed_classes[item["bug"]]
        records = {x["arm"]: x for x in item["arms"]}
        scored = {}
        for arm in ARMS:
            prediction = records[arm]
            method = prediction.get("predicted_method")
            cls = (method.split("(", 1)[0].rsplit(".", 1)[0]
                   if method and "." in method else None)
            scored[arm] = {
                "predicted_method": method, "predicted_class": cls,
                "hit_at_1_modified_class": (cls in gold if cls else False),
                "parse_valid": method is not None,
                "actual_input_tokens": prediction.get("actual_input_tokens"),
                "generation_seconds": prediction.get("generation_seconds"),
                "skipped": bool(prediction.get("skip_reason")),
            }
        a = scored["grounded_first8"]
        b = scored["grounded_reversed"]
        scored["position_agreement"] = (
            a["predicted_method"] is not None and
            a["predicted_method"] == b["predicted_method"])
        scored["grounded_has_gold_in_pool"] = any(
            x.split("(", 1)[0].rsplit(".", 1)[0] in gold
            for x in records["grounded_first8"]["selected_method_ids"]
        )
        scored["original_has_gold_in_pool"] = any(
            x.split("(", 1)[0].rsplit(".", 1)[0] in gold
            for x in records["original_first8"]["selected_method_ids"]
        )
        results.append({"bug": item["bug"], "arms": scored})
    return {
        "scope": "Two previously studied bugs, not independent evaluation",
        "gold_definition": "Defects4J changed class membership, not causal root cause",
        "retrieval_labels_used_before_inference": False,
        "llm_accuracies_claimed_independent": False,
        "results": results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", nargs=2, action="append",
                   metavar=("BUG", "PRE_GOLD_JSON"), required=True)
    p.add_argument("--gold-report", type=Path,
                   help="Previously frozen development-only labels")
    p.add_argument("--checkout", nargs=2, action="append",
                   metavar=("BUG", "BUGGY_CHECKOUT"),
                   help="For unseen projects, only export labels after raw predictions")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-input-tokens", type=int, default=1800)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if len({bug for bug, _ in args.snapshot}) != len(args.snapshot):
        raise ValueError("duplicate bugs")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    revision = HfApi().model_info(args.model).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.float32)
    model.eval()

    frozen = {}
    pre_gold_decisions = {}
    for bug, path in args.snapshot:
        snap = json.loads(Path(path).read_text())
        if snap.get("provenance", {}).get("gold_patches_loaded") is not False:
            raise ValueError("candidate source may not contain gold")
        frozen[bug] = freeze_arms(
            snap, tokenizer, max_input_tokens=args.max_input_tokens)
        pre_gold_decisions[bug] = decide_pre_gold(snap).__dict__

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_prompts.json").write_text(
        json.dumps(frozen, indent=2) + "\n")
    (args.output_dir / "pre_gold_guard_decisions.json").write_text(
        json.dumps(pre_gold_decisions, indent=2) + "\n")
    raw = []
    for bug, arms in frozen.items():
        item = {"bug": bug, "model_id": args.model,
                "model_revision": revision, "arms": []}
        for arm in arms:
            output = {k: v for k, v in arm.items() if k != "prompt"}
            if arm["skip_reason"] is not None:
                item["arms"].append(output)
                continue
            inputs = tokenizer(arm["prompt"], return_tensors="pt",
                               add_special_tokens=False)
            input_len = int(inputs["input_ids"].shape[1])
            if input_len > args.max_input_tokens:
                raise ValueError("prompt exceeded matched max")
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=24,
                    pad_token_id=tokenizer.eos_token_id)
            seconds = time.perf_counter() - started
            completion = generated[0][input_len:]
            answer = tokenizer.decode(completion, skip_special_tokens=True)
            # Number in each arm refers to that arm's local ordering.
            chosen = parse_prediction(answer, arm["selected_levels"])
            output.update({
                "raw_response": answer,
                "predicted_method": (arm["selected_method_ids"][chosen]
                                     if chosen is not None else None),
                "generation_seconds": seconds,
                "actual_input_tokens": input_len,
                "output_tokens": int(len(completion)),
            })
            print(bug, arm["arm"], "reply", repr(answer),
                  "method", output["predicted_method"],
                  "input", input_len, flush=True)
            item["arms"].append(output)
        raw.append(item)

    # Hard gold firewall: persist all model responses before opening labels.
    (args.output_dir / "raw_predictions_before_gold.json").write_text(
        json.dumps(raw, indent=2) + "\n")
    if (args.gold_report is None) == (args.checkout is None):
        raise ValueError("Choose exactly one posthoc label source")
    if args.checkout:
        # This import and subprocess run occur *after* raw predictions
        # are persisted to disk, enforcing the pre-gold firewall.
        from .cpu_live_defects4j_pairs import _read_gold_modified_classes
        gold = {bug: _read_gold_modified_classes(Path(checkout))
                for bug, checkout in args.checkout}
    else:
        posthoc = json.loads(args.gold_report.read_text())
        gold = {
            row["bug"]: set(row["gold_modified_classes"])
            for row in posthoc["results"] if not row.get("skipped")
        }
    if set(gold) != {x[0] for x in args.snapshot}:
        raise ValueError("gold report has different bug set")
    scores = score_arms(raw, gold)
    for row in scores["results"]:
        bug = row["bug"]
        decision = pre_gold_decisions[bug]
        selected = ("grounded_one_L1"
                    if decision["action"] == "UPGRADE_L1" and
                    not row["arms"]["grounded_one_L1"]["skipped"]
                    else "grounded_first8")
        arm = row["arms"][selected]
        extra = row["arms"]["grounded_one_L1"]
        base = row["arms"]["grounded_first8"]
        row["guarded_policy"] = {
            "action": ("UPGRADE_L1" if selected == "grounded_one_L1"
                       else "STOP"),
            "pre_gold_reason": decision["reason"],
            "hit_at_1_modified_class": arm["hit_at_1_modified_class"],
            "actual_input_tokens": arm["actual_input_tokens"],
            "generation_seconds": arm["generation_seconds"],
            "extra_upgrade_tokens_avoided": (
                max(0, (extra["actual_input_tokens"] or 0)
                    - (base["actual_input_tokens"] or 0))
                if selected == "grounded_first8" and not extra["skipped"]
                else 0),
            "harm_avoided_vs_forced_upgrade": (
                selected == "grounded_first8" and
                not extra["skipped"] and
                base["hit_at_1_modified_class"] and
                not extra["hit_at_1_modified_class"]),
            "benefit_missed_vs_forced_upgrade": (
                selected == "grounded_first8" and
                not extra["skipped"] and
                not base["hit_at_1_modified_class"] and
                extra["hit_at_1_modified_class"]),
        }
    scores["pre_gold_guard_decisions_frozen"] = True
    scores["model_id"] = args.model
    scores["model_revision"] = revision
    (args.output_dir / "scored_after_gold.json").write_text(
        json.dumps(scores, indent=2) + "\n")
    print("Saved real matched-model reranking and order-control results")


if __name__ == "__main__":
    main()

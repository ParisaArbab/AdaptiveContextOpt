"""Run Qwen paired STOP vs upgrade on independent real Defects4J projects.

Requires pre-gold snapshot JSONs made from *buggy checkouts* by
aitech.live_defects4j_snapshot. No model prompts or candidate rankings
use modified-file labels. Only after the raw prediction JSON is flushed
do we call Defects4J 'classes.modified' for posthoc FILE-level scoring.

This is a feasibility experiment, not sufficient data to train a
cross-project adaptive controller or establish localization novelty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from .cpu_action_counterfactual import freeze_upgrade_pair
from .cpu_llm_localization import MODEL_ID, parse_prediction


def _read_gold_modified_classes(checkout: Path) -> set[str]:
    proc = subprocess.run(
        ["defects4j", "export", "-p", "classes.modified"],
        cwd=checkout, capture_output=True, text=True,
        timeout=60, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"posthoc modified classes failed: {proc.stderr[-500:]}")
    classes = {x.strip() for x in proc.stdout.splitlines() if x.strip()}
    if not classes:
        raise RuntimeError("no modified class labels exported")
    return classes


def _predicted_class(method_id: str | None) -> str | None:
    if not method_id or "." not in method_id.split("(", 1)[0]:
        return None
    return method_id.split("(", 1)[0].rsplit(".", 1)[0]


def score_file_outcomes(records: list[dict],
                        gold_by_bug: dict[str, set[str]]) -> list[dict]:
    scored = []
    for record in records:
        if record["skip_reason"]:
            scored.append({"bug": record["bug"], "skipped": True,
                           "reason": record["skip_reason"]})
            continue
        predictions = record["predictions"]
        if [x["condition"] for x in predictions] != ["STOP", "UPGRADE_L1"]:
            raise ValueError("paired conditions mismatched")
        gold = gold_by_bug[record["bug"]]
        before = _predicted_class(predictions[0]["predicted_method"]) in gold
        after = _predicted_class(predictions[1]["predicted_method"]) in gold
        scored.append({
            "bug": record["bug"], "skipped": False,
            "gold_modified_classes": sorted(gold),
            "stop_file_hit": before,
            "upgrade_file_hit": after,
            "outcome": int(after) - int(before),
            "stop_actual_input_tokens":
                predictions[0]["actual_total_input_tokens"],
            "upgrade_actual_input_tokens":
                predictions[1]["actual_total_input_tokens"],
            "stop_generation_seconds": predictions[0]["generation_seconds"],
            "upgrade_generation_seconds": predictions[1]["generation_seconds"],
            "ground_truth_scope": "official Defects4J changed class, not root-cause proof",
        })
    return scored


def run(bug_inputs: dict[str, tuple[Path, Path]], *,
        output_dir: Path, evidence_budget: int = 650,
        max_input_tokens: int = 1800,
        max_new_tokens: int = 24) -> None:
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
    for bug, (snapshot_path, checkout) in bug_inputs.items():
        source_bytes = snapshot_path.read_bytes()
        snapshot = json.loads(source_bytes)
        if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
            raise ValueError("input does not certify pre-gold source extraction")
        spec = freeze_upgrade_pair(
            snapshot, tokenizer,
            evidence_budget=evidence_budget,
            max_total_input=max_input_tokens)
        spec["snapshot_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        spec["checkout"] = str(checkout)
        frozen[bug] = spec

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_prompts_before_gold.json").write_text(
        json.dumps({
            bug: {k: v for k, v in spec.items() if k != "checkout"}
            for bug, spec in frozen.items()
        }, indent=2) + "\n")

    records = []
    for bug, spec in frozen.items():
        row = {
            "bug": bug, "skip_reason": spec["skip_reason"],
            "snapshot_sha256": spec["snapshot_sha256"],
            "model_id": MODEL_ID, "model_revision": revision,
            "target_upgrade": spec.get("target_method"),
            "predictions": [],
        }
        if not spec["skip_reason"]:
            for condition in spec["prompts"]:
                prompt = condition["prompt"]
                inputs = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False)
                prompt_tokens = int(inputs["input_ids"].shape[1])
                if prompt_tokens > max_input_tokens:
                    raise ValueError("input exceeded common ceiling")
                started = time.perf_counter()
                with torch.no_grad():
                    result = model.generate(
                        **inputs, do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id)
                duration = time.perf_counter() - started
                completion = result[0][prompt_tokens:]
                raw_reply = tokenizer.decode(
                    completion, skip_special_tokens=True)
                selected = {
                    int(k): int(v)
                    for k, v in condition["selected_levels"].items()
                }
                predicted = parse_prediction(raw_reply, selected)
                row["predictions"].append({
                    "condition": condition["condition"],
                    "prompt_sha256": condition["prompt_sha256"],
                    "actual_total_input_tokens": prompt_tokens,
                    "raw_response": raw_reply,
                    "predicted_method": (
                        spec["candidate_id_by_index"][str(predicted)]
                        if predicted is not None else None),
                    "generation_seconds": duration,
                    "output_tokens": int(len(completion)),
                })
                print("REAL MODEL", bug, condition["condition"],
                      row["predictions"][-1]["predicted_method"], flush=True)
        records.append(row)

    # Strict posthoc boundary: persist before ANY modified-class query.
    (output_dir / "raw_predictions_before_gold.json").write_text(
        json.dumps(records, indent=2) + "\n")
    modified = {
        bug: _read_gold_modified_classes(spec["checkout"] and
              Path(spec["checkout"]))
        for bug, spec in frozen.items()
    }
    results = score_file_outcomes(records, modified)
    (output_dir / "file_level_outcomes_posthoc.json").write_text(
        json.dumps({
            "scope": "Real buggy Defects4J project, development feasibility only",
            "frozen_before_gold": True,
            "not_independently_heldout_for_controller_training": True,
            "model_id": MODEL_ID, "model_revision": revision,
            "results": results,
        }, indent=2) + "\n")
    print("Saved raw predictions before gold and posthoc file-level outcomes")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bug", action="append", nargs=3,
                   metavar=("BUG", "SNAPSHOT", "CHECKOUT"),
                   required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--evidence-budget", type=int, default=650)
    p.add_argument("--max-input-tokens", type=int, default=1800)
    args = p.parse_args()
    data = {}
    for bug, snapshot, checkout in args.bug:
        if bug in data:
            raise ValueError("duplicate bug input")
        data[bug] = (Path(snapshot), Path(checkout))
    run(data, output_dir=args.output_dir,
        evidence_budget=args.evidence_budget,
        max_input_tokens=args.max_input_tokens)


if __name__ == "__main__":
    main()

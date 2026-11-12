"""Score *frozen* real LLM predictions against official Time-1 patch labels.

This module is separate from context construction and inference, so gold
files/methods cannot enter model prompts or the evidence selection policy.
It reports both evidence availability and actual model prediction validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .time1_posthoc_scoring import PATCH_LABELS


def evaluate(predictions: dict, labels: dict) -> dict:
    if predictions.get("gold_patch_loaded") is not False:
        raise ValueError("Cannot score predictions without explicit no-gold run provenance")
    if labels.get("scoring_only") is not True:
        raise ValueError("Patch labels are not frozen as scoring-only")
    gold = set(labels["changed_method_candidate_ids"])
    if not gold:
        raise ValueError("No changed methods")
    rows = []
    for row in predictions["rows"]:
        selected = set(row["selected_candidate_ids"])
        predicted = row.get("predicted_candidate_id")
        if predicted is not None and predicted not in selected:
            raise ValueError("Model prediction outside eligible evidence methods")
        rows.append({
            "condition": row["condition"],
            "budget_model_tokens": row["max_input_tokens"],
            "actual_prompt_model_tokens": row["input_token_count"],
            "source_helper_included": row["source_helper_included"],
            "gold_methods_in_context": len(selected.intersection(gold)),
            "gold_methods_total": len(gold),
            "model_prediction": predicted,
            "prediction_valid": predicted is not None,
            "model_hit_any_patched_method": predicted in gold,
            "inference_seconds": row["inference_seconds"],
            "reused_identical_prompt": row["reused_identical_prompt"],
        })
    paired = {}
    for row in rows:
        paired.setdefault(row["budget_model_tokens"], {})[row["condition"]] = row
    comparisons = []
    for budget, group in sorted(paired.items()):
        if not {"anchor_first", "force_verified_source",
                "gate_unverified_binding"} <= group.keys():
            raise ValueError("Missing policy condition at a budget")
        base, forced, gate = (
            group[x] for x in ("anchor_first", "force_verified_source",
                               "gate_unverified_binding")
        )
        if base["model_prediction"] != gate["model_prediction"]:
            raise ValueError("Abstain gate differs from identical baseline")
        comparisons.append({
            "budget_model_tokens": budget,
            "model_hit_delta_forced_minus_anchor": (
                int(forced["model_hit_any_patched_method"]) -
                int(base["model_hit_any_patched_method"])
            ),
            "valid_prediction_delta": (
                int(forced["prediction_valid"]) - int(base["prediction_valid"])
            ),
            "gold_context_method_delta": (
                forced["gold_methods_in_context"] -
                base["gold_methods_in_context"]
            ),
        })
    return {
        "scope": "single previously seen Defects4J Time-1 bug; exploratory small CPU model",
        "model_id": predictions.get("model_id"),
        "model_revision_sha": predictions.get("model_revision_sha"),
        "gold_source": labels["label_source"],
        "selected_before_gold_scoring": True,
        "rows": rows,
        "paired": comparisons,
        "limitations": [
            "Only one previously seen bug, no independent test split",
            "0.5B CPU model accuracy does not represent larger code models",
            "Gold patch changed methods are not necessarily unique causal locations",
            "Single deterministic model output is not statistical evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(PATCH_LABELS.read_text(encoding="utf-8"))
    result = evaluate(raw, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(f"{row['budget_model_tokens']} {row['condition']}: "
              f"valid={row['prediction_valid']} "
              f"patch_hit={row['model_hit_any_patched_method']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

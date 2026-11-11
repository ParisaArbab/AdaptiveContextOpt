"""Validate paired LLM action outcomes before training a marginal-value model.

Each row compares STOP vs one evidence action on one frozen bug. True
outcomes must be scored only AFTER raw model predictions have been
written and the two prompts are fixed. This validator does not assume
a label is true just because a record says it is.

Expected JSONL rows:
{
  "project": "Lang", "bug": "Lang-1", "split": "train",
  "model_id": "Qwen/...", "seed": 42,
  "decoding_hash": "...", "failure_sha256": "...",
  "baseline_prompt_sha256": "...", "action_prompt_sha256": "...",
  "baseline_prediction_sha256": "...", "action_prediction_sha256": "...",
  "actual_baseline_input_tokens": 300, "actual_action_input_tokens": 350,
  "common_input_token_ceiling": 1800,
  "action": {... EvidenceAction fields ...}, "outcome": -1 | 0 | 1,
  "gold_scoring_only": true, "predictions_frozen_before_gold": true
}

Positive outcome = action localization was correct when STOP was not.
Negative = STOP was correct and action was not.
Zero = both correct or both incorrect. Invalid predictions count as wrong.
All rows must represent SAME MODEL and matched decoding protocol, and the
same input-token ceiling; differences in realized token counts are logged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json

from .marginal_action_policy import EvidenceAction, FEATURE_NAMES, action_feature_vector


REQUIRED = {
    "project", "bug", "split", "model_id", "seed", "decoding_hash",
    "failure_sha256", "baseline_prompt_sha256", "action_prompt_sha256",
    "baseline_prediction_sha256", "action_prediction_sha256",
    "actual_baseline_input_tokens", "actual_action_input_tokens",
    "common_input_token_ceiling", "action", "outcome",
    "gold_scoring_only", "predictions_frozen_before_gold",
}
SPLITS = {"train", "calibration", "test"}


def validate_pairs(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("need actual paired LLM outcome rows")
    project_splits = {}
    seen = set()
    labels = Counter()
    model_ids = set()
    ceilings = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != REQUIRED:
            raise ValueError(f"row {i}: schema mismatch")
        if row["split"] not in SPLITS:
            raise ValueError(f"row {i}: invalid split")
        for key in ("project", "bug", "model_id", "decoding_hash",
                    "failure_sha256", "baseline_prompt_sha256",
                    "action_prompt_sha256", "baseline_prediction_sha256",
                    "action_prediction_sha256"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"row {i}: invalid {key}")
        if type(row["seed"]) is not int:
            raise ValueError(f"row {i}: invalid seed")
        for key in ("actual_baseline_input_tokens",
                    "actual_action_input_tokens", "common_input_token_ceiling"):
            if type(row[key]) is not int or row[key] <= 0:
                raise ValueError(f"row {i}: invalid {key}")
        if (row["actual_baseline_input_tokens"] > row["common_input_token_ceiling"]
                or row["actual_action_input_tokens"] > row["common_input_token_ceiling"]):
            raise ValueError(f"row {i}: prompt exceeds common token ceiling")
        if row["baseline_prompt_sha256"] == row["action_prompt_sha256"]:
            raise ValueError(f"row {i}: action and STOP prompt are identical")
        if (row["gold_scoring_only"] is not True
                or row["predictions_frozen_before_gold"] is not True):
            raise ValueError(f"row {i}: no verified freeze-before-scoring contract")
        if type(row["outcome"]) is not int or row["outcome"] not in (-1, 0, 1):
            raise ValueError(f"row {i}: outcome must be -1, 0 or 1")
        if not isinstance(row["action"], dict):
            raise ValueError(f"row {i}: invalid action")
        expected_fields = {f.name for f in fields(EvidenceAction)}
        if not set(row["action"]).issubset(expected_fields):
            raise ValueError(f"row {i}: unknown action feature")
        try:
            action = EvidenceAction(**row["action"])
            action_feature_vector(action)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"row {i}: invalid pre-action evidence: {exc}") from exc
        if (action.requires_call_binding and not action.compiler_binding_verified):
            raise ValueError(f"row {i}: experiment used unverified graph binding")
        key = (row["bug"], row["seed"], action.action_id)
        if key in seen:
            raise ValueError(f"row {i}: duplicate bug/action/seed")
        seen.add(key)
        project = row["project"]
        if project in project_splits and project_splits[project] != row["split"]:
            raise ValueError(f"row {i}: cross-project split leakage")
        project_splits[project] = row["split"]
        labels[(row["split"], row["outcome"])] += 1
        model_ids.add(row["model_id"])
        ceilings.add(row["common_input_token_ceiling"])
    if len(model_ids) != 1 or len(ceilings) != 1:
        raise ValueError("model and input ceiling must be matched")
    return {
        "schema": 1,
        "count": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "projects": len(project_splits),
        "project_split": dict(sorted(project_splits.items())),
        "label_counts": {
            split: {str(label): labels[split, label] for label in (-1, 0, 1)}
            for split in sorted(SPLITS)
        },
        "model_id": next(iter(model_ids)),
        "common_input_token_ceiling": next(iter(ceilings)),
        "sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "limits": [
            "Cannot independently verify that raw predictions were frozen before scoring",
            "Same input ceiling does not imply equal actual input tokens or computation",
            "Changed patch methods are imperfect labels for root cause",
            "Mixed project train/calibration/test data is rejected"
        ],
    }

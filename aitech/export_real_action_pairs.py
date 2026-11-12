"""Export independently frozen real Defects4J pairs to action-learning JSONL.

Reads pre-gold raw Qwen predictions, paired prompt manifests, original
pre-gold source snapshots, and POSTHOC scored modified-class labels.
Records only completed pairs. Skipped pairs are NOT training labels.

The explicit split is supplied by the experimenter. Data from the
same project must never appear in different train/calibration/test
splits. The validator enforces project grouping and a consistent
changed-class Hit@1 outcome. This does not pretend the labels are
causal root-cause or that the model is calibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .paired_action_outcomes import validate_pairs


def _sha(data: object) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def export_rows(
    raw: list[dict], frozen: dict, scored: dict,
    snapshots: dict[str, dict], project_split: dict[str, str],
) -> tuple[list[dict], dict]:
    if not scored.get("frozen_before_gold") is True:
        raise ValueError("posthoc report lacks freeze-before-gold declaration")
    scores_by_bug = {x["bug"]: x for x in scored.get("results", [])}
    if len(scores_by_bug) != len(scored.get("results", [])):
        raise ValueError("duplicate posthoc bug")
    rows = []
    skipped = []
    for result in raw:
        bug = result["bug"]
        project = bug.split("-", 1)[0]
        if project not in project_split:
            raise ValueError(f"no declared split for project {project}")
        if bug not in frozen or bug not in scores_by_bug or bug not in snapshots:
            raise ValueError(f"missing paired data for {bug}")
        spec = frozen[bug]
        score = scores_by_bug[bug]
        if result.get("skip_reason") or score.get("skipped"):
            skipped.append({"bug": bug, "reason": result.get("skip_reason") or
                            score.get("reason")})
            continue
        predictions = result["predictions"]
        prompts = spec["prompts"]
        if ([p["condition"] for p in predictions] != ["STOP", "UPGRADE_L1"]
                or [p["condition"] for p in prompts] != ["STOP", "UPGRADE_L1"]):
            raise ValueError("wrong frozen before/after condition order")
        if any(p["prompt_sha256"] != f["prompt_sha256"]
               for p, f in zip(predictions, prompts)):
            raise ValueError("raw prediction hashes differ from frozen prompts")
        snapshot = snapshots[bug]
        if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
            raise ValueError("candidate pool not demonstrably pre-gold")
        target = spec["target_method"]
        index = spec["upgrade_index"]
        candidates = snapshot["candidates"]
        if not (0 <= index < len(candidates)
                and candidates[index]["function_id"] == target):
            raise ValueError("upgrade target not in frozen candidate list")
        before = predictions[0]["actual_total_input_tokens"]
        after = predictions[1]["actual_total_input_tokens"]
        delta = after - before
        if delta <= 0:
            raise ValueError("action did not add input tokens")
        if score["outcome"] != (
            int(score["upgrade_file_hit"]) - int(score["stop_file_hit"])
        ):
            raise ValueError("posthoc outcome and file labels disagree")
        if (score["stop_actual_input_tokens"] != before
                or score["upgrade_actual_input_tokens"] != after):
            raise ValueError("scoring changed actual token accounting")
        candidate = candidates[index]
        name = target.split("(", 1)[0].rsplit(".", 1)[-1]
        failure = snapshot["context"]
        row = {
            "project": project, "bug": bug,
            "split": project_split[project],
            "model_id": result["model_id"] + "@" + result["model_revision"],
            "seed": 42,
            "decoding_hash": _sha({
                "mode": "greedy", "max_new_tokens": 24, "seed": 42
            }),
            "failure_sha256": snapshot["provenance"]["failure_sha256"],
            "baseline_prompt_sha256": predictions[0]["prompt_sha256"],
            "action_prompt_sha256": predictions[1]["prompt_sha256"],
            "baseline_prediction_sha256": _sha({
                "response": predictions[0]["raw_response"],
                "method": predictions[0]["predicted_method"],
            }),
            "action_prediction_sha256": _sha({
                "response": predictions[1]["raw_response"],
                "method": predictions[1]["predicted_method"],
            }),
            "actual_baseline_input_tokens": before,
            "actual_action_input_tokens": after,
            "common_input_token_ceiling": spec["common_total_input_ceiling"],
            "action": {
                "action_id": bug + ":UPGRADE_L1",
                "kind": "UPGRADE_L1", "target_id": target,
                "incremental_input_tokens": delta,
                "incremental_latency_ms": 0.0,
                "source_revision": spec["snapshot_sha256"],
                "provenance": (
                    "frozen Defects4J buggy snapshot; method candidate "
                    "from regex source extraction, not compiler-verified"),
                "compiler_binding_verified": False,
                "requires_call_binding": False,
                "support_sufficiency_audited": False,
                "failure_relevance": candidate["features"]["failure_relevance"],
                "semantic_similarity": 0.0,
                "rank_fraction": index / max(1, len(candidates) - 1),
                "failure_anchor": name + "(" in failure,
                "context_fill_fraction": before / spec["common_total_input_ceiling"],
            },
            "outcome": score["outcome"],
            "outcome_definition": "defects4j_changed_class_hit_at_1",
            "gold_scoring_only": True,
            "predictions_frozen_before_gold": True,
        }
        rows.append(row)
    validation = validate_pairs(rows) if rows else {
        "status": "NO_VALID_PAIRED_ROWS",
        "reason": "all comparisons skipped",
    }
    return rows, {
        "exported_pairs": len(rows),
        "skipped_pairs": skipped,
        "validation": validation,
        "limitation": (
            "Only method upgrade observations, not independent "
            "trained-controller accuracy; no semantic embedding features."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--snapshot", nargs=2, action="append",
                   metavar=("BUG", "SNAPSHOT"), required=True)
    p.add_argument("--split", action="append", nargs=2,
                   metavar=("PROJECT", "SPLIT"), required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    args = p.parse_args()
    input_dir = args.results_dir
    snapshots = {
        bug: json.loads(Path(path).read_text())
        for bug, path in args.snapshot
    }
    rows, summary = export_rows(
        json.loads((input_dir / "raw_predictions_before_gold.json").read_text()),
        json.loads((input_dir / "frozen_prompts_before_gold.json").read_text()),
        json.loads((input_dir / "file_level_outcomes_posthoc.json").read_text()),
        snapshots,
        dict(args.split),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print("Validated real LLM outcome rows:", len(rows),
          "skipped:", len(summary["skipped_pairs"]))


if __name__ == "__main__":
    main()

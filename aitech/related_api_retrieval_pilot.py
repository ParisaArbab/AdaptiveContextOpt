"""Replay frozen source candidate selection before reading patch classes.

Pure CPU/no-LLM retrieval-recall pilot. This evaluates whether the
changed class is available among eight presented method identities,
NOT whether a model can correctly localize it or whether that class
is a proven cause. Gold is opened only after frozen selections are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .failure_grounded_rerank import rerank
from .related_api_retrieval import related_shortlist, owner_of


def freeze(snapshot: dict, *, count: int = 8) -> dict:
    if snapshot.get("provenance", {}).get("gold_patches_loaded") is not False:
        raise ValueError("snapshot must originate from pre-gold source")
    candidates = snapshot["candidates"]
    if len(candidates) < count:
        raise ValueError("not enough pre-gold candidates")
    grounded = rerank(snapshot)[:count]
    related = related_shortlist(snapshot, count=count)
    choices = {
        "original_top8": list(range(count)),
        "stack_bm25_top8": grounded,
        "related_api_top8": list(related.original_indices),
    }
    return {
        "source_failure_sha256": snapshot["provenance"]["failure_sha256"],
        "source_candidate_count": len(candidates),
        "count_per_policy": count,
        "related_heuristic_proposals": [
            list(p) for p in related.related_class_proposals
        ],
        "test_class_hints": list(related.test_name_class_hints),
        "compiler_binding_claimed": False,
        "selections": {
            policy: {
                "indices": selected,
                "methods": [candidates[i]["function_id"] for i in selected],
                "classes": sorted({
                    owner_of(candidates[i]["function_id"]) for i in selected
                }),
            } for policy, selected in choices.items()
        },
    }


def score(frozen: dict, modified_classes: set[str]) -> dict:
    if not modified_classes:
        raise ValueError("missing official changed-class labels")
    scores = {}
    for method, choices in frozen["selections"].items():
        selected_classes = set(choices["classes"])
        found = selected_classes.intersection(modified_classes)
        scores[method] = {
            "modified_class_present": bool(found),
            "retrieved_modified_class_count": len(found),
            "retrieved_class_count": len(selected_classes),
            "modified_class_recall": len(found) / len(modified_classes),
            "not_a_model_hit_at_1": True,
        }
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", nargs=2, action="append", required=True,
                   metavar=("BUG", "PRE_GOLD_SNAPSHOT"))
    p.add_argument("--gold", nargs=2, action="append", required=True,
                   metavar=("BUG", "OFFICIAL_MODIFIED_CLASSES"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    snapshots = dict(args.snapshot)
    paths = dict(args.gold)
    if len(snapshots) != len(args.snapshot) or set(paths) != set(snapshots):
        raise ValueError("duplicate bugs or unmatched gold paths")
    frozen = {}
    for bug, source_path in snapshots.items():
        source = json.loads(Path(source_path).read_text())
        frozen[bug] = freeze(source)
        frozen[bug]["snapshot_sha256"] = hashlib.sha256(
            Path(source_path).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_before_gold.json").write_text(
        json.dumps(frozen, indent=2) + "\n")

    scored = {}
    for bug, file in paths.items():
        labels = set(Path(file).read_text().split())
        scored[bug] = score(frozen[bug], labels)
    report = {
        "scope": "Studied development bugs, changed-class candidate recall",
        "not_llm_localization": True,
        "labels_loaded_after_freeze": True,
        "bugs": scored,
    }
    (args.output_dir / "retrieval_recall_after_gold.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for bug, r in scored.items():
        print(bug, {k: x["modified_class_present"]
                    for k, x in r.items()}, flush=True)


if __name__ == "__main__":
    main()

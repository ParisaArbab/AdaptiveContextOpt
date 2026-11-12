"""Post-selection patch-method retention analysis for Time-1.

This separate scorer loads frozen retrieval selections THEN patch labels.
No labels are available to candidate retrieval, rank or budget packing.
Changed-method retention does not measure causal localization or LLM quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .time1_verified_retrieval_study import SNAPSHOT, SOURCE_RECORD, run

PATCH_LABELS = Path("data/aitech/time1_posthoc_patch_labels.json")


def score(selection: dict, patch: dict) -> dict:
    if patch.get("scoring_only") is not True:
        raise ValueError("patch record not marked scoring-only")
    gold = set(patch["changed_method_candidate_ids"])
    if not gold or not selection.get("rows"):
        raise ValueError("missing frozen selections or changed-method labels")
    rows = []
    for original in selection["rows"]:
        chosen = set(original["selected_ids"])
        hits = gold & chosen
        rows.append({
            "condition": original["condition"],
            "budget_regex_proxy": original["budget_regex_proxy"],
            "retained_patch_methods": sorted(hits),
            "changed_method_hits": len(hits),
            "changed_method_total": len(gold),
            "changed_method_recall": len(hits) / len(gold),
            "used_regex_proxy_tokens": original["selected_regex_proxy"],
            "retrieved_helper_included": original["retrieved_helper_included"],
        })
    baseline = {r["budget_regex_proxy"]: r for r in rows
                if r["condition"] == "anchor_first"}
    for r in rows:
        r["changed_method_recall_delta_vs_anchor"] = (
            r["changed_method_recall"] -
            baseline[r["budget_regex_proxy"]]["changed_method_recall"]
        )
    return {
        "bug": "Time-1",
        "scope": "Post-selection changed-method retention on a previously studied bug",
        "patch_source": patch["label_source"],
        "patch_source_blob_sha": patch["label_source_blob_sha"],
        "selection_frozen_before_scoring": True,
        "llm_ran": False,
        "causal_fault_localization_measured": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/aitech/time1_posthoc_recall.json"))
    args = parser.parse_args()
    selection = run(
        json.loads(SNAPSHOT.read_text(encoding="utf-8")),
        json.loads(SOURCE_RECORD.read_text(encoding="utf-8")),
    )
    patch = json.loads(PATCH_LABELS.read_text(encoding="utf-8"))
    result = score(selection, patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(f"{row['budget_regex_proxy']:4} {row['condition']:25} "
              f"patch method hits={row['changed_method_hits']}/{row['changed_method_total']} "
              f"delta={row['changed_method_recall_delta_vs_anchor']:+.2f} "
              f"helper={row['retrieved_helper_included']}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

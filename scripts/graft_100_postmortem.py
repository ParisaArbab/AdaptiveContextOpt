"""Offline, non-destructive audit of the frozen GRAFT-FL 100-bug run.

Reads archived real artifacts; never calls Ollama or Defects4J. Does not alter
the frozen cohort, source evidence, model cache, or original aggregate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

POLICIES = ("l1_anchor", "l1_graph_focused",
            "l1_graph_test_aware_nostop", "l1_graph_test_aware")
FUSIONS = ("rrf", "suspicion_rank_blend")


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def diagnose(root: Path):
    aggregate = load(root / "aggregate" / "aggregate.json")
    if not aggregate or "instances" not in aggregate:
        raise ValueError("Frozen aggregate.json missing or invalid")
    selection = aggregate["cohort_selection_sha256"]
    rows = aggregate["instances"]
    if len(rows) != 200:
        raise ValueError(f"Expected exactly 200 bug/fusion rows, got {len(rows)}")
    status_by_bug = defaultdict(dict)
    for row in rows:
        status_by_bug[row["bug"]][row["fusion"]] = row
    if len(status_by_bug) != 100 or any(set(v) != set(FUSIONS) for v in status_by_bug.values()):
        raise ValueError("Missing/duplicate bug-fusion rows, refusing analysis")
    groups = []
    paired = []
    incomplete = []
    for batch in range(20):
        folder = root / "batches" / f"batch_{batch:02d}"
        prep = load(folder / "prepared" / "preparation_report.json") or {}
        preflight = load(folder / "experiment" / "preflight.json") or {}
        gate = load(folder / "experiment" / "completion_gate.json") or {}
        audit = load(folder / "experiment" / "paired_audit.json") or {}
        batch_bugs = [r["bug"] for r in rows
                      if r["batch"] == batch and r["fusion"] == "rrf"]
        prep_errors = {e.get("bug"): e.get("error") for e in prep.get("errors", [])}
        candidate_errors = {e.get("instance_id"): e.get("error")
                            for e in preflight.get("candidate_preparations", [])
                            if e.get("status") != "ready"}
        incomplete_arms = gate.get("incomplete", [])
        audit_index = {(r.get("bug"), r.get("fusion"), r.get("policy")): r
                       for r in audit.get("records", [])}
        batch_entry = {"batch": batch, "bugs": batch_bugs,
                       "gate_completed": gate.get("completed"),
                       "preparation_status": prep.get("status"),
                       "preparation_errors": prep_errors,
                       "candidate_errors": candidate_errors,
                       "incomplete_arms": incomplete_arms}
        groups.append(batch_entry)
        for bug in batch_bugs:
            observations = status_by_bug[bug]
            valid = all(r.get("status") == "complete_with_graph"
                        for r in observations.values())
            if not valid:
                reason = ("preparation" if bug in prep_errors else
                          "candidate_pool" if bug in candidate_errors else
                          "inference_or_pair" if any(a[0] == bug for a in incomplete_arms) else
                          "not_scored_or_missing")
                incomplete.append({"bug": bug, "batch": batch,
                                   "split": observations["rrf"]["split"],
                                   "reason": reason,
                                   "detail": (prep_errors.get(bug) or
                                              candidate_errors.get(bug) or
                                              "Check preparation, preflight and per-bug run status")})
                continue
            for fusion in FUSIONS:
                row = observations[fusion]
                m = row.get("file_mrr", {})
                counts = row.get("graph_calls", {})
                if not all(isinstance(m.get(p), (int, float)) for p in POLICIES):
                    raise ValueError(f"Invalid MRR in valid pair: {bug}/{fusion}")
                audits = {p: audit_index.get((bug, fusion, p), {}) for p in POLICIES}
                focused_ids = audits["l1_graph_focused"].get("graph_candidate_ids")
                test_aware_ids = audits["l1_graph_test_aware"].get("graph_candidate_ids")
                paired.append({
                    "bug": bug, "batch": batch, "split": row["split"],
                    "fusion": fusion,
                    "mrr": m,
                    "mrr_delta_focused": m["l1_graph_focused"] - m["l1_anchor"],
                    "mrr_delta_test_aware": m["l1_graph_test_aware"] - m["l1_anchor"],
                    "graph_calls": counts,
                    "focused_graph_targets": focused_ids,
                    "test_aware_graph_targets": test_aware_ids,
                    "graph_target_overlap": (
                        len(set(focused_ids) & set(test_aware_ids))
                        if isinstance(focused_ids, list) and isinstance(test_aware_ids, list)
                        else None),
                    "stop_gate_candidates": audits["l1_graph_test_aware"].get("stop_gate_candidates"),
                    "logical_tokens": {p: audits[p].get("logical_tokens") for p in POLICIES},
                    "graph_downgrade_guards": {p: audits[p].get("graph_downgrade_guard_count")
                                               for p in POLICIES},
                })
    result = {
        "schema": "graft-100-offline-postmortem-v1",
        "selection_sha256": selection,
        "source": "frozen archived outputs; no inference executed",
        "fully_paired_unique_bugs": len({r["bug"] for r in paired}),
        "fully_paired_evaluation_bugs": len({r["bug"] for r in paired if r["split"] == "evaluation"}),
        "incomplete_unique_bugs": len(incomplete),
        "incomplete_by_reason": dict(Counter(r["reason"] for r in incomplete)),
        "paired": paired, "incomplete": incomplete, "batches": groups,
    }
    return result


def report(data: dict) -> str:
    pairs = data["paired"]
    lines = [
        "# GRAFT-FL recovery and graph-effect audit",
        "",
        f"Frozen selection SHA256: `{data['selection_sha256']}`",
        "",
        f"Fully paired unique bugs: **{data['fully_paired_unique_bugs']} / 100**.",
        f"Fully paired held-out evaluation: **{data['fully_paired_evaluation_bugs']} / 80**.",
        f"Incomplete unique bugs: **{data['incomplete_unique_bugs']}**.",
        "",
        "## Reason groups (bug-level, not per-arm)",
        "",
    ]
    for k, v in sorted(data["incomplete_by_reason"].items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Graph impact on completed pairs", ""]
    for split in ("development", "evaluation"):
        for fusion in FUSIONS:
            chosen = [r for r in pairs if r["split"] == split and r["fusion"] == fusion]
            if not chosen:
                continue
            improved = sum(r["mrr_delta_focused"] > 0 for r in chosen)
            worsened = sum(r["mrr_delta_focused"] < 0 for r in chosen)
            graph_saved = sum(
                (r["graph_calls"].get("l1_graph_test_aware") or 0)
                < (r["graph_calls"].get("l1_graph_test_aware_nostop") or 0)
                for r in chosen
            )
            lines.append(f"- {split}, {fusion}: n={len(chosen)}, focused MRR higher={improved}, "
                         f"lower={worsened}, unchanged={len(chosen)-improved-worsened}; "
                         f"test-aware saved graph calls in {graph_saved} cases.")
    lines += [
        "",
        "Identical MRR does **not** establish identical method rankings or LLM responses.",
        "The audit compares the recorded graph target IDs and graph calls when archived.",
        "Inspect original request ledgers and result traces before claiming why the",
        "rankings did not change. Physical provider tokens are shared across policies.",
        "",
        "## Incomplete bugs and recovery action",
        "",
        "| Batch | Bug | Cause | Action |",
        "|---:|---|---|---|",
    ]
    actions = {
        "candidate_pool": "Inspect BM25 candidate enumeration and deduplication; do not lower frozen 24-candidate requirement",
        "preparation": "Inspect exact test parser and source bound; do not truncate or replace authentic test",
        "inference_or_pair": "Inspect per-bug status, provider ledger and resume safety",
        "not_scored_or_missing": "Inspect archived preparation and completion gate",
    }
    for r in data["incomplete"]:
        detail = str(r["detail"]).replace("|", "/").replace("\n", " ")[:150]
        lines.append(f"| {r['batch']:02d} | {r['bug']} | {r['reason']}: {detail} | "
                     f"{actions[r['reason']]} |")
    lines += [
        "",
        "## Recovery rules",
        "",
        "1. Keep the original frozen 100 IDs, split, evidence hash and provider settings.",
        "2. Preserve completed result files, caches, raw ledgers, and checkpoints.",
        "3. Fix verified software defects only; record code version and affected bug IDs.",
        "4. Run a fresh preflight on each affected batch after a documented correction.",
        "5. Retry incomplete evaluations only after confirming no new cohort substitution.",
        "6. Re-aggregate, report before/after attrition, and do not overwrite the original run.",
        "",
        "## Random Forest gate",
        "",
        "Train only on development bugs and only with features available before graph",
        "inspection. A positive label is measured graph MRR gain over L1, not",
        "graph-call completion. If the training split has only one label class,",
        "a valid supervised gain predictor cannot be trained.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/graft_100_cohort_20261115"))
    parser.add_argument("--out", type=Path, default=Path("results/graft_100_recovery_audit"))
    args = parser.parse_args()
    data = diagnose(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "postmortem.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (args.out / "postmortem.md").write_text(report(data))
    print(f"Audited {data['fully_paired_unique_bugs']} completed and "
          f"{data['incomplete_unique_bugs']} incomplete bugs. No inference executed.")
    print(args.out / "postmortem.md")


if __name__ == "__main__":
    main()

"""Audit BM25 candidate-pool recall at K=24,48,96 without ANY LLM calls.

Uses the same immutable GRAFT checkouts and observed-failure query as the
frozen EGAV study. Gold is read only after candidate retrieval, for the audit.
Does not modify the recorded manifests, policies, or previous evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rave_fl.eval.adapters import load_instances
from rave_fl.eval.candidates import provide
from rave_fl.eval.runner import validate_checkout

SIZES = (24, 48, 96)
RESULTS = ROOT / "results/egav_validation_10bugs"
CONFIGS = ROOT / "data/graft_100_cohort_20261115/batches"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(path):
    return str(path).replace("\\", "/").lstrip("./")


def evaluate_batch(batch):
    config = CONFIGS / ("batch_" + batch) / "experiment/config.json"
    opts = dict(load(config)["benchmark"])
    if opts["candidate_provider"] != "bm25":
        raise ValueError("Not a BM25 candidate provider: " + batch)
    opts["candidate_count"] = SIZES[-1]
    opts["allow_downloads"] = False
    opts["materialize_checkouts"] = False
    instances, _ = load_instances(opts)
    output = RESULTS / ("batch_" + batch)
    previous = {
        bug["id"]: bug for bug in load(output / "manifest.json")["bugs"]
    }
    target_ids = set(previous)
    if len(previous) != 5:
        raise ValueError("Expected exactly 5 prepared validation cases")
    by_id = {inst.instance_id: inst for inst in instances if inst.instance_id in target_ids}
    if set(by_id) != target_ids:
        raise ValueError("Unable to load identical original GRAFT cases")
    # Gold must not influence retrieval. We deliberately delay loading labels
    # until every candidate pool is generated.
    rows = {}
    for bid in sorted(target_ids):
        instance = validate_checkout(by_id[bid])
        if instance.evidence.kind != "observed_failure":
            raise ValueError("Not the frozen observed failure evidence: " + bid)
        entities, provenance = provide(instance, opts)
        if not entities:
            raise ValueError("Empty candidate retrieval: " + bid)
        original = previous[bid]["candidates"]
        expected = [c["id"] for c in original]
        observed = [e.entity_id for e in entities[:len(expected)]]
        if expected != observed:
            raise ValueError(
                "BM25 prefix no longer matches saved pilot candidate order "
                "for " + bid + ". Do not combine these pools.")
        rows[bid] = {
            "bug": bid, "retrieved_unique_methods": len(entities),
            "indexed_entities": provenance.get("indexed_entities"),
            "ordered_files": [normalize(e.path) for e in entities],
        }
    gold = load(output / "gold_offline.json")["bugs"]
    for bid in rows:
        files = {normalize(f) for f in gold[bid]["files"]}
        ordered_files = rows[bid].pop("ordered_files")
        first_rank = next(
            (i for i, f in enumerate(ordered_files, 1) if f in files), None)
        rows[bid].update(
            gold_files=sorted(files), first_method_position_for_gold_file=first_rank,
            coverage_by_k={str(k): bool(first_rank and first_rank <= k) for k in SIZES}
        )
    return rows


def main():
    combined = {}
    for batch in ("01", "02"):
        combined.update(evaluate_batch(batch))
    if len(combined) != 10:
        raise ValueError("Expected 10 non-overlapping cases")
    print("BM25 CANDIDATE RECALL, NO MODEL CALLS")
    print(f"{'Bug':15} {'First method':>12} {'Top24':>6} {'Top48':>6} {'Top96':>6}")
    for bid, row in sorted(combined.items()):
        rank = row["first_method_position_for_gold_file"]
        bits = ["Y" if row["coverage_by_k"][str(k)] else "N" for k in SIZES]
        print(f"{bid:15} {str(rank or 'MISS'):>12} {bits[0]:>6} {bits[1]:>6} {bits[2]:>6}")
    rates = {str(k): sum(x["coverage_by_k"][str(k)] for x in combined.values())
             for k in SIZES}
    print("\nRecall counts (10 bugs):", rates)
    print("NOTE: file presence only, not final MRR or code localization.")
    payload = {
        "schema": "bm25-pool-recall-audit-v1",
        "cohort": "egav_validation_10bugs",
        "no_inference": True,
        "candidate_count": list(SIZES),
        "file_hit_count": rates,
        "rows": combined,
    }
    path = RESULTS / "retrieval_recall_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("Saved:", path)


if __name__ == "__main__":
    main()

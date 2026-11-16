"""Read-only BM25 source-pool diagnostic for the six failed preflight bugs.

Never runs Defects4J tests, writes source files, contacts Ollama, or changes
candidate_count. Uses the same instance loader, source index and BM25 provider
as the frozen run. It reports only metadata and errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from rave_fl.eval.adapters import load_instances
from rave_fl.eval.candidates import provide
from rave_fl.eval.representations import index_source

BATCHES = (2, 6, 7, 11, 14, 19)
FAILED = {"Closure-32", "Closure-106", "Closure-170",
          "Closure-85", "Closure-161", "Math-58"}


def inspect(root: Path):
    records = []
    for n in BATCHES:
        path = root / "batches" / f"batch_{n:02d}" / "experiment" / "config.json"
        if not path.is_file():
            records.append({"batch": n, "error": "Missing archived config.json"})
            continue
        cfg = json.loads(path.read_text())["benchmark"]
        # Source indexing and retrieval only. Nothing writes to the checkouts.
        instances, _ = load_instances(cfg)
        for instance in instances:
            if instance.instance_id not in FAILED:
                continue
            row = {"batch": n, "bug": instance.instance_id,
                   "required_candidates": cfg["candidate_count"],
                   "checkout": instance.checkout}
            try:
                pool = index_source(instance, cfg["granularity"],
                                    cfg["max_source_files"], cfg["max_source_bytes"])
                ids = [e.entity_id for e in pool]
                duplicates = {name: c for name, c in Counter(ids).items() if c > 1}
                row.update(indexed_entities=len(pool),
                           indexed_unique_ids=len(set(ids)),
                           duplicate_ids=duplicates,
                           eligible_pool_at_least_24=len(set(ids)) >= cfg["candidate_count"])
                entities, provenance = provide(instance, cfg)
                selected = [e.entity_id for e in entities]
                row.update(selected_count=len(selected), selected_unique=len(set(selected)),
                           provider=provenance.get("provider"),
                           provider_indexed_entities=provenance.get("indexed_entities"),
                           retrieval_first_ids=selected[:5])
                if len(set(selected)) < cfg["candidate_count"]:
                    row["diagnosis"] = "BM25 did not return 24 unique indexed entity IDs"
                else:
                    row["diagnosis"] = ("BM25 returned 24 unique IDs; failure occurred "
                                        "later in the preparation/preflight pipeline")
            except Exception as exc:
                row["diagnosis"] = "Exception during source indexing or retrieval"
                row["exception_type"] = type(exc).__name__
                row["exception_message"] = str(exc)
            records.append(row)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/graft_100_cohort_20261115"))
    p.add_argument("--out", type=Path, default=Path("results/graft_100_recovery_audit/bm25_pool_diagnostic.json"))
    args = p.parse_args()
    records = inspect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "graft-bm25-pool-diagnostic-v1",
                                    "records": records}, indent=2) + "\n")
    for r in records:
        print(f"Batch {r['batch']:02d}, {r.get('bug', '?')}: "
              f"indexed={r.get('indexed_entities', 'n/a')}, "
              f"unique={r.get('indexed_unique_ids', 'n/a')}, "
              f"retrieved={r.get('selected_count', 'n/a')}, "
              f"diagnosis={r.get('diagnosis', r.get('error'))}")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

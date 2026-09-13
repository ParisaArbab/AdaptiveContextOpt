from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from wp1.swebench_ochiai import (
    collect_entities,
    entity_covered,
)


def load_coverage(path: Path, repo: Path):
    data = json.loads(path.read_text())

    covered = {}

    for name, info in data.get("files", {}).items():
        normalized = name.replace("\\", "/")

        try:
            p = Path(name)

            if p.is_absolute():
                normalized = (
                    p.resolve()
                    .relative_to(repo.resolve())
                    .as_posix()
                )
        except Exception:
            pass

        covered[normalized] = {
            int(x)
            for x in info.get("executed_lines", [])
        }

    return covered


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--instance", required=True)
    ap.add_argument("--source-package", required=True)

    args = ap.parse_args()

    root = Path.home() / "AdaptiveContextOpt"

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    repo = work / "repo"
    out = work / "outputs" / "ochiai"

    failing_files = sorted(
        out.glob("coverage_failing_*.json")
    )

    passing_files = sorted(
        out.glob("coverage_passing_*.json")
    )

    print("Failing spectra:", len(failing_files))
    print("Passing spectra:", len(passing_files))

    if not failing_files:
        raise SystemExit("No failing coverage JSON files.")

    if not passing_files:
        raise SystemExit("No passing coverage JSON files.")

    entities = collect_entities(
        repo,
        args.source_package,
    )

    print("Production entities:", len(entities))

    # Accumulate counts without keeping 22 huge coverage JSONs in memory.
    failed_count = {
        entity.ref: 0
        for entity in entities
    }

    passed_count = {
        entity.ref: 0
        for entity in entities
    }

    entity_by_ref = {
        entity.ref: entity
        for entity in entities
    }

    for i, path in enumerate(failing_files, 1):
        print(
            f"Reading failing spectrum {i}/{len(failing_files)}: "
            f"{path.name}",
            flush=True,
        )

        coverage = load_coverage(path, repo)

        for entity in entities:
            if entity_covered(entity, coverage):
                failed_count[entity.ref] += 1

        del coverage

    for i, path in enumerate(passing_files, 1):
        print(
            f"Reading passing spectrum {i}/{len(passing_files)}: "
            f"{path.name}",
            flush=True,
        )

        coverage = load_coverage(path, repo)

        for entity in entities:
            if entity_covered(entity, coverage):
                passed_count[entity.ref] += 1

        del coverage

    total_failed = len(failing_files)
    total_passed = len(passing_files)

    rows = []

    for ref, entity in entity_by_ref.items():
        ef = failed_count[ref]
        ep = passed_count[ref]

        if ef == 0:
            continue

        denominator = math.sqrt(
            total_failed * (ef + ep)
        )

        score = ef / denominator if denominator else 0.0

        rows.append(
            {
                "entity": ref,
                "kind": entity.kind,
                "file": entity.file,
                "start": entity.start,
                "end": entity.end,
                "failed_covered": ef,
                "passed_covered": ep,
                "score": score,
                "runtime_line_count": len(
                    entity.executable_lines
                ),
            }
        )

    rows.sort(
        key=lambda x: (
            -x["score"],
            -x["failed_covered"],
            x["passed_covered"],
            x["entity"],
        )
    )

    csv_path = out / "ochiai.csv"
    json_path = out / "ochiai.json"

    fields = [
        "rank",
        "entity",
        "kind",
        "file",
        "start",
        "end",
        "failed_covered",
        "passed_covered",
        "score",
        "runtime_line_count",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for rank, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                }
            )

    result = {
        "instance_id": args.instance,
        "method": "Ochiai",
        "adaptation": (
            "Python coverage.py spectrum, "
            "runtime-body entity mapping"
        ),
        "total_failed": total_failed,
        "total_passed": total_passed,
        "candidate_count": len(rows),
        "top20": rows[:20],
        "top5": [
            row["entity"]
            for row in rows[:5]
        ],
    }

    json_path.write_text(
        json.dumps(result, indent=2)
    )

    unique_scores = len({
        row["score"]
        for row in rows
    })

    print()
    print("==================================================")
    print("RESCORE SUMMARY")
    print("==================================================")
    print("Candidates    :", len(rows))
    print("Unique scores :", unique_scores)

    print()
    print("==================================================")
    print("OCHIAI TOP-20")
    print("==================================================")

    for i, row in enumerate(rows[:20], 1):
        print(
            f"{i:2}. {row['entity']}"
            f"  score={row['score']:.6f}"
            f"  fail={row['failed_covered']}"
            f"  pass={row['passed_covered']}"
        )

    gold = "sympy/core/_print_helpers.py::Printable"

    gold_rows = [
        (i, row)
        for i, row in enumerate(rows, 1)
        if row["entity"] == gold
    ]

    print()
    print("==================================================")
    print("DIAGNOSTIC GOLD CHECK, EVALUATION ONLY")
    print("==================================================")

    if gold_rows:
        rank, row = gold_rows[0]

        print("Gold entity rank:", rank)
        print("Gold score      :", row["score"])
        print(
            "Gold fail/pass  :",
            row["failed_covered"],
            row["passed_covered"],
        )
    else:
        print("Gold entity not dynamically covered.")

    print()
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()

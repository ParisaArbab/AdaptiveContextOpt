from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from wp1.swebench_blues import (
    load_stopwords,
    collect_corpus,
)


def normalize_path(name: str, repo: Path) -> str:
    name = name.replace("\\", "/")

    try:
        p = Path(name)

        if p.is_absolute():
            return (
                p.resolve()
                .relative_to(repo.resolve())
                .as_posix()
            )
    except Exception:
        pass

    return name


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

    ochiai_out = (
        work
        / "outputs"
        / "ochiai"
    )

    ochiai_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    failing_files = sorted(
        ochiai_out.glob(
            "coverage_failing_*.json"
        )
    )

    passing_files = sorted(
        ochiai_out.glob(
            "coverage_passing_*.json"
        )
    )

    if not failing_files:
        raise SystemExit(
            "No failing coverage JSON files."
        )

    if not passing_files:
        raise SystemExit(
            "No passing coverage JSON files."
        )

    print(
        "Failing spectra:",
        len(failing_files),
    )

    print(
        "Passing spectra:",
        len(passing_files),
    )

    # Use the exact same Python statement universe as Blues.
    stopwords = load_stopwords(root)

    _, _, statements = collect_corpus(
        repo,
        args.source_package,
        stopwords,
    )

    print(
        "Statement universe:",
        len(statements),
    )

    # One statement per source line in our Blues adaptation.
    statement_index = {
        (s.file, int(s.line)): s
        for s in statements
    }

    failed_covered = {
        key: 0
        for key in statement_index
    }

    passed_covered = {
        key: 0
        for key in statement_index
    }

    def process_spectrum(
        path: Path,
        target,
    ):
        data = json.loads(
            path.read_text()
        )

        covered_once = set()

        for name, info in data.get(
            "files",
            {},
        ).items():

            rel = normalize_path(
                name,
                repo,
            )

            # Production package only.
            prefix = (
                args.source_package.rstrip("/")
                + "/"
            )

            if not (
                rel == args.source_package
                or rel.startswith(prefix)
            ):
                continue

            if "/tests/" in f"/{rel}/":
                continue

            for line in info.get(
                "executed_lines",
                [],
            ):
                key = (
                    rel,
                    int(line),
                )

                if key in statement_index:
                    covered_once.add(key)

        for key in covered_once:
            target[key] += 1

    for i, path in enumerate(
        failing_files,
        1,
    ):
        print(
            f"Reading failing "
            f"{i}/{len(failing_files)}: "
            f"{path.name}",
            flush=True,
        )

        process_spectrum(
            path,
            failed_covered,
        )

    for i, path in enumerate(
        passing_files,
        1,
    ):
        print(
            f"Reading passing "
            f"{i}/{len(passing_files)}: "
            f"{path.name}",
            flush=True,
        )

        process_spectrum(
            path,
            passed_covered,
        )

    total_failed = len(
        failing_files
    )

    rows = []

    for key, statement in statement_index.items():

        ef = failed_covered[key]
        ep = passed_covered[key]

        if ef == 0:
            continue

        denominator = math.sqrt(
            total_failed * (ef + ep)
        )

        score = (
            ef / denominator
            if denominator
            else 0.0
        )

        rows.append(
            {
                "statement": statement.ref,
                "file": statement.file,
                "line": statement.line,
                "entity": statement.entity,
                "failed_covered": ef,
                "passed_covered": ep,
                "score": score,
            }
        )

    rows.sort(
        key=lambda x: (
            -x["score"],
            -x["failed_covered"],
            x["passed_covered"],
            x["statement"],
        )
    )

    for rank, row in enumerate(
        rows,
        1,
    ):
        row["rank"] = rank

    result = {
        "instance_id": args.instance,
        "method": "Ochiai",
        "level": "statement",
        "total_failed": len(
            failing_files
        ),
        "total_passed": len(
            passing_files
        ),
        "count": len(rows),
        "ranking": rows,
    }

    output = (
        ochiai_out
        / "ochiai_statements.json"
    )

    output.write_text(
        json.dumps(result, indent=2)
    )

    print()
    print("============================================")
    print("OCHIAI STATEMENT TOP-20")
    print("============================================")

    for row in rows[:20]:

        print(
            f"{row['rank']:2}. "
            f"{row['statement']} "
            f"score={row['score']:.6f} "
            f"fail={row['failed_covered']} "
            f"pass={row['passed_covered']} "
            f"entity={row['entity']}"
        )

    print()
    print(
        "Statement candidates:",
        len(rows),
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def load_statement_ranking(path: Path):
    x = json.loads(path.read_text())

    rows = []

    for original_rank, row in enumerate(
        x.get("ranking", []),
        1,
    ):
        statement = row.get("statement")

        if not statement:
            continue

        # RAFL must receive production-code statements only.
        # Test files and pytest configuration are evidence,
        # not fault-localization candidates.
        file_path = row.get("file")

        if not file_path and "#" in statement:
            file_path = statement.split("#", 1)[0]

        if file_path:
            normalized = file_path.replace("\\", "/")
            filename = Path(normalized).name

            if "/tests/" in f"/{normalized}/":
                continue

            if filename == "conftest.py":
                continue

            if filename.startswith("test_"):
                continue

        score = float(
            row.get("score", 0.0)
        )

        rank = int(
            row.get("rank", original_rank)
        )

        rows.append(
            {
                "statement": statement,
                "score": score,
                "rank": rank,
                "entity": row.get("entity"),
                "file": row.get("file"),
                "line": row.get("line"),
            }
        )

    # Original RAFL first keeps the best score for
    # duplicate statements.
    best = {}

    for row in rows:
        statement = row["statement"]

        previous = best.get(statement)

        if (
            previous is None
            or row["score"] > previous["score"]
        ):
            best[statement] = row

    # Descending suspiciousness score.
    # Preserve original rank for deterministic tie handling.
    ranked = sorted(
        best.values(),
        key=lambda r: (
            -r["score"],
            r["rank"],
            r["statement"],
        ),
    )

    return ranked


def production_entity(entity: str | None) -> bool:
    if not entity:
        return False

    path = entity.split("::", 1)[0]

    if "/tests/" in f"/{path}/":
        return False

    if path.endswith("/conftest.py"):
        return False

    if Path(path).name.startswith("test_"):
        return False

    return True


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--instance",
        required=True,
    )

    args = ap.parse_args()

    root = Path.home() / "AdaptiveContextOpt"

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    blues_path = (
        work
        / "outputs"
        / "blues"
        / "blues_statements.json"
    )

    ochiai_path = (
        work
        / "outputs"
        / "ochiai"
        / "ochiai_statements.json"
    )

    out = work / "outputs" / "sbir"
    out.mkdir(parents=True, exist_ok=True)

    if not blues_path.exists():
        raise SystemExit(
            f"Missing Blues ranking: {blues_path}"
        )

    if not ochiai_path.exists():
        raise SystemExit(
            f"Missing Ochiai ranking: {ochiai_path}"
        )

    blues = load_statement_ranking(
        blues_path
    )

    ochiai = load_statement_ranking(
        ochiai_path
    )

    print("Blues statements :", len(blues))
    print("Ochiai statements:", len(ochiai))

    # --------------------------------------------------
    # Original RAFL behavior:
    #
    # N = minimum length across FL lists
    # k = min(100, N)
    # --------------------------------------------------

    N = min(
        len(blues),
        len(ochiai),
    )

    k = min(100, N)

    if k == 0:
        raise SystemExit(
            "Cannot aggregate empty FL lists."
        )

    print("Minimum list size:", N)
    print("RAFL k           :", k)

    blues_top = blues[:k]
    ochiai_top = ochiai[:k]

    # Statement -> entity lookup used only AFTER RAFL.
    statement_entity = {}

    for row in blues + ochiai:
        if row.get("entity"):
            statement_entity.setdefault(
                row["statement"],
                row["entity"],
            )

    # --------------------------------------------------
    # Write exact two-list RAFL input
    # --------------------------------------------------

    input_path = out / "rafl_input.tsv"

    with input_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "technique",
                "rank",
                "statement",
                "score",
            ]
        )

        for name, ranking in (
            ("Blues", blues_top),
            ("Ochiai", ochiai_top),
        ):
            for i, row in enumerate(
                ranking,
                1,
            ):
                writer.writerow(
                    [
                        name,
                        i,
                        row["statement"],
                        row["score"],
                    ]
                )

    # --------------------------------------------------
    # R RankAggreg script
    #
    # Parameters match original RAFL:
    #
    # k=100
    # seed=1
    # method=CE
    # convIn=7
    # popSize=100
    # CP=0.4
    # MP=0.01
    # maxIter=1000
    # N=10000
    # distance=Spearman
    # rho=0.01
    # --------------------------------------------------

    r_script = out / "run_rafl.R"

    optimal_path = out / "rafl_optimal_list.txt"

    r_script.write_text(
r'''
library(RankAggreg)

args <- commandArgs(trailingOnly=TRUE)

input_file <- args[1]
output_file <- args[2]

d <- read.delim(
    input_file,
    sep="\t",
    stringsAsFactors=FALSE,
    quote="",
    check.names=FALSE
)

techniques <- unique(d$technique)

lists <- lapply(
    techniques,
    function(name) {
        x <- d[d$technique == name, ]
        x <- x[order(x$rank), ]
        x$statement
    }
)

names(lists) <- techniques

k <- min(
    100,
    min(sapply(lists, length))
)

cat("Techniques:", paste(techniques, collapse=", "), "\n")
cat("k:", k, "\n")

data <- do.call(
    rbind,
    lapply(
        lists,
        function(x) x[1:k]
    )
)

cat("RAFL matrix dimensions:",
    nrow(data),
    "x",
    ncol(data),
    "\n"
)

set.seed(1)

result <- RankAggreg(
    data,
    k,
    seed=1,
    method="CE",
    convIn=7,
    popSize=100,
    CP=0.4,
    MP=0.01,
    maxIter=1000,
    N=10000,
    distance="Spearman",
    rho=0.01,
    verbose=TRUE
)

cat("\n============================================\n")
cat("RAFL RESULT\n")
cat("============================================\n")

print(result)

writeLines(
    as.character(result$top.list),
    output_file
)

cat("\nOptimal list saved:", output_file, "\n")
'''
    )

    # --------------------------------------------------
    # Run actual R RankAggreg
    # --------------------------------------------------

    log_path = out / "rafl.log"

    print()
    print("Running RankAggreg...")
    print(
        "This uses the original RAFL CE/Spearman settings."
    )

    proc = subprocess.run(
        [
            "Rscript",
            str(r_script),
            str(input_path),
            str(optimal_path),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(
        proc.stdout
    )

    print(proc.stdout)

    if proc.returncode != 0:
        raise SystemExit(
            f"RAFL failed. See {log_path}"
        )

    if not optimal_path.exists():
        raise SystemExit(
            "RAFL did not produce an optimal list."
        )

    optimal = [
        line.strip()
        for line in optimal_path.read_text().splitlines()
        if line.strip()
    ]

    print()
    print("Aggregated statements:", len(optimal))

    # --------------------------------------------------
    # Match original RAFL output suspiciousness:
    #
    # score = (maxLen - i) / maxLen
    # --------------------------------------------------

    statement_rows = []

    max_len = len(optimal)

    for i, statement in enumerate(optimal):
        score = (
            (max_len - i) / max_len
            if max_len
            else 0.0
        )

        statement_rows.append(
            {
                "rank": i + 1,
                "statement": statement,
                "score": score,
                "entity": statement_entity.get(
                    statement
                ),
            }
        )

    # --------------------------------------------------
    # Statement -> Python entity
    #
    # Analogous to FlexFL's statement -> method conversion.
    # --------------------------------------------------

    entity_rows = []
    seen_entities = set()

    for row in statement_rows:
        entity = row.get("entity")

        if not production_entity(entity):
            continue

        if entity in seen_entities:
            continue

        seen_entities.add(entity)

        entity_rows.append(
            {
                "rank": len(entity_rows) + 1,
                "entity": entity,
                "source_statement": row["statement"],
                "rafl_statement_rank": row["rank"],
                "score": row["score"],
            }
        )

    result = {
        "instance_id": args.instance,
        "method": "SBIR",
        "aggregation": "RAFL",
        "gold_used": False,
        "inputs": {
            "Blues": str(blues_path),
            "Ochiai": str(ochiai_path),
        },
        "rafl_config": {
            "k": k,
            "seed": 1,
            "method": "CE",
            "distance": "Spearman",
            "N": 10000,
            "maxIter": 1000,
            "convIn": 7,
            "rho": 0.01,
            "popSize": 100,
            "CP": 0.4,
            "MP": 0.01,
        },
        "statement_count": len(
            statement_rows
        ),
        "statement_ranking": statement_rows,
        "entity_count": len(
            entity_rows
        ),
        "top20": entity_rows[:20],
        "top5": [
            row["entity"]
            for row in entity_rows[:5]
        ],
    }

    result_path = out / "sbir.json"

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "============================================"
    )
    print("SBIR STATEMENT TOP-20")
    print(
        "============================================"
    )

    for row in statement_rows[:20]:
        print(
            f"{row['rank']:2}. "
            f"{row['statement']} "
            f"-> {row['entity']}"
        )

    print()
    print(
        "============================================"
    )
    print("SBIR ENTITY TOP-20")
    print(
        "============================================"
    )

    for row in entity_rows[:20]:
        print(
            f"{row['rank']:2}. "
            f"{row['entity']} "
            f"(statement rank "
            f"{row['rafl_statement_rank']})"
        )

    print()
    print(
        "============================================"
    )
    print("SBIR TOP-5")
    print(
        "============================================"
    )

    for i, entity in enumerate(
        result["top5"],
        1,
    ):
        print(
            f"Top-{i}: {entity}"
        )

    print()
    print("Saved:", result_path)
    print("RAFL log:", log_path)


if __name__ == "__main__":
    main()

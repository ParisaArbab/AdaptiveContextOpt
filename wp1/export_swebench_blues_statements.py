from __future__ import annotations

import argparse
import json
from pathlib import Path

from wp1.swebench_blues import (
    load_stopwords,
    tokenize,
    collect_corpus,
    rank_documents,
    create_configuration,
    combine_configurations,
)


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
    out = work / "outputs" / "blues"
    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (work / "metadata.json").read_text()
    )

    # Problem statement only.
    # No gold patch or gold file is used.
    query_text = metadata["problem_statement"]

    stopwords = load_stopwords(root)

    query_tokens = tokenize(
        query_text,
        stopwords,
    )

    print("INSTANCE:", args.instance)
    print("QUERY TOKENS:", len(query_tokens))

    entities, files, statements = collect_corpus(
        repo,
        args.source_package,
        stopwords,
    )

    print("FILES:", len(files))
    print("ENTITIES:", len(entities))
    print("STATEMENTS:", len(statements))

    # --------------------------------------------------
    # File ranking
    # --------------------------------------------------

    file_scores = rank_documents(
        query_tokens,
        [x["tokens"] for x in files],
        k1=1.0,
        b=0.3,
    )

    file_ranking = [
        (
            files[index]["file"],
            score,
        )
        for index, score in file_scores[:50]
    ]

    # --------------------------------------------------
    # Statement ranking
    # --------------------------------------------------

    statement_docs = [
        tokenize(
            statement.text,
            stopwords,
        )
        for statement in statements
    ]

    statement_scores = rank_documents(
        query_tokens,
        statement_docs,
        k1=1.0,
        b=0.3,
    )

    statement_ranking = [
        (
            statements[index].ref,
            score,
        )
        for index, score
        in statement_scores[:10000]
    ]

    statements_by_ref = {
        s.ref: s
        for s in statements
    }

    # --------------------------------------------------
    # Six Blues configurations
    # --------------------------------------------------

    configurations = []

    for m in [1, 25, 50, 100, 10000]:

        configurations.append(
            create_configuration(
                file_ranking,
                statement_ranking,
                statements_by_ref,
                m=m,
                weighted=False,
            )
        )

    configurations.append(
        create_configuration(
            file_ranking,
            statement_ranking,
            statements_by_ref,
            m=10000,
            weighted=True,
        )
    )

    ensemble = combine_configurations(
        configurations
    )

    rows = []

    for rank, (
        statement_ref,
        score,
        votes,
    ) in enumerate(ensemble, 1):

        statement = statements_by_ref.get(
            statement_ref
        )

        if statement is None:
            continue

        rows.append(
            {
                "rank": rank,
                "statement": statement_ref,
                "file": statement.file,
                "line": statement.line,
                "entity": statement.entity,
                "score": score,
                "votes": votes,
            }
        )

    result = {
        "instance_id": args.instance,
        "method": "Blues",
        "level": "statement",
        "query_source": "problem_statement only",
        "gold_used": False,
        "count": len(rows),
        "ranking": rows,
    }

    output = out / "blues_statements.json"

    output.write_text(
        json.dumps(result, indent=2)
    )

    print()
    print("============================================")
    print("BLUES STATEMENT TOP-20")
    print("============================================")

    for row in rows[:20]:
        print(
            f"{row['rank']:2}. "
            f"{row['statement']} "
            f"score={row['score']:.6f} "
            f"votes={row['votes']} "
            f"entity={row['entity']}"
        )

    print()
    print("Statement candidates:", len(rows))
    print("Saved:", output)


if __name__ == "__main__":
    main()

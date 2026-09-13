from __future__ import annotations

import argparse
import ast
import json
import keyword
import math
import re

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.stem import PorterStemmer


@dataclass
class MethodDoc:
    file: str
    entity: str
    start: int
    end: int
    raw_source: str
    tokens: list[str]


# ------------------------------------------------------------
# Original BoostN-style preprocessing
# ------------------------------------------------------------

STEMMER = PorterStemmer()


def load_stopwords(root: Path) -> set[str]:

    candidates = [
        root
        / "references"
        / "FlexFL_OriginalReplication"
        / "prepare"
        / "non-LLM-based_FL"
        / "BoostN"
        / "BoostN-main"
        / "scripts"
        / "StopwordsPlusJava.txt",

        Path.home()
        / "FlexFL_OriginalReplication"
        / "prepare"
        / "non-LLM-based_FL"
        / "BoostN"
        / "BoostN-main"
        / "scripts"
        / "StopwordsPlusJava.txt",
    ]

    stopwords = set()

    for path in candidates:
        if path.exists():

            stopwords.update(
                line.strip().lower()
                for line in path.read_text(
                    errors="replace"
                ).splitlines()
                if line.strip()
            )

            print(
                "Using original BoostN stopwords:",
                path,
            )

            break

    # Python adaptation:
    # remove Python language keywords in the same spirit as
    # the original Java-specific stopword list.
    stopwords.update(
        x.lower()
        for x in keyword.kwlist
    )

    stopwords.update(
        {
            "the", "and", "for", "are", "with",
            "this", "that", "from", "into",
            "not", "but", "you", "your", "has",
            "have", "was", "were", "can", "will",
            "should", "would", "could", "its",
            "our", "their",
        }
    )

    return stopwords


def split_identifier(word: str) -> list[str]:

    word = word.replace("_", " ")

    word = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        word,
    )

    word = re.sub(
        r"(?<=[A-Z])(?=[A-Z][a-z])",
        " ",
        word,
    )

    return word.split()


def preprocess(
    text: str,
    stopwords: set[str],
) -> list[str]:

    # Match original preprocessing:
    # eliminate non literals except underscore.
    text = re.sub(
        r"[^a-zA-Z_]",
        " ",
        text,
    )

    output = []

    for raw in text.split():

        for token in split_identifier(raw):

            token = token.lower().strip()

            if not token:
                continue

            # Original implementation removes words
            # of length <= 1.
            if len(token) <= 1:
                continue

            if token in stopwords:
                continue

            output.append(
                STEMMER.stem(token)
            )

    return output


# ------------------------------------------------------------
# Python method corpus
# ------------------------------------------------------------

def collect_methods(
    repo: Path,
    source_package: str,
    stopwords: set[str],
) -> list[MethodDoc]:

    methods = []

    prefix = (
        source_package.rstrip("/")
        + "/"
    )

    python_files = []

    for path in repo.rglob("*.py"):

        rel = path.relative_to(
            repo
        ).as_posix()

        if not (
            rel == source_package
            or rel.startswith(prefix)
        ):
            continue

        # Original BoostN uses production source.
        if "/tests/" in f"/{rel}/":
            continue

        if Path(rel).name == "conftest.py":
            continue

        if Path(rel).name.startswith(
            "test_"
        ):
            continue

        python_files.append(
            (path, rel)
        )

    python_files.sort(
        key=lambda x: x[1]
    )

    print(
        "PRODUCTION PYTHON FILES:",
        len(python_files),
        flush=True,
    )

    for index, (path, rel) in enumerate(
        python_files,
        1,
    ):

        if (
            index == 1
            or index % 100 == 0
            or index == len(python_files)
        ):

            print(
                f"Parsing "
                f"{index}/{len(python_files)}: "
                f"{rel}",
                flush=True,
            )

        try:
            source = path.read_text(
                errors="replace"
            )

            tree = ast.parse(
                source
            )

        except Exception:
            continue

        lines = source.splitlines()

        def walk(node, stack):

            if isinstance(
                node,
                ast.ClassDef,
            ):

                for child in node.body:
                    walk(
                        child,
                        stack + [node.name],
                    )

                return

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                qualname = ".".join(
                    stack + [node.name]
                )

                start = int(
                    node.lineno
                )

                end = int(
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    )
                )

                raw_source = "\n".join(
                    lines[
                        start - 1:end
                    ]
                )

                # Same idea as original BoostN's
                # method_contents.length() < 11 filter.
                if len(
                    raw_source.strip()
                ) >= 11:

                    entity = (
                        f"{rel}::{qualname}"
                    )

                    methods.append(
                        MethodDoc(
                            file=rel,
                            entity=entity,
                            start=start,
                            end=end,
                            raw_source=raw_source,
                            tokens=preprocess(
                                raw_source,
                                stopwords,
                            ),
                        )
                    )

                # Preserve nested Python functions as
                # separate method-level candidates.
                for child in node.body:
                    walk(
                        child,
                        stack + [node.name],
                    )

                return

            for child in ast.iter_child_nodes(
                node
            ):
                walk(
                    child,
                    stack,
                )

        walk(
            tree,
            [],
        )

    return methods


# ------------------------------------------------------------
# Lucene-style BM25
# ------------------------------------------------------------

def bm25_rank(
    query: list[str],
    docs: list[MethodDoc],
):

    n = len(docs)

    if n == 0:
        return [], None, None

    # Original BoostN:
    #
    # if counter > 3000:
    #     k1 = 1.0
    # else:
    #     k1 = 0.0
    #
    # b = 0.3
    k1 = (
        1.0
        if n > 3000
        else 0.0
    )

    b = 0.3

    lengths = [
        len(doc.tokens)
        for doc in docs
    ]

    avgdl = (
        sum(lengths) / n
        if n
        else 1.0
    )

    df = Counter()

    for doc in docs:

        for token in set(
            doc.tokens
        ):
            df[token] += 1

    qtf = Counter(query)

    ranking = []

    for doc_index, doc in enumerate(
        docs
    ):

        frequencies = Counter(
            doc.tokens
        )

        dl = len(
            doc.tokens
        )

        score = 0.0

        for token, query_frequency in qtf.items():

            tf = frequencies.get(
                token,
                0,
            )

            if tf == 0:
                continue

            document_frequency = (
                df[token]
            )

            # Lucene BM25-style positive IDF.
            idf = math.log(
                1.0
                + (
                    n
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b
                    * dl
                    / max(
                        avgdl,
                        1e-12,
                    )
                )
            )

            if denominator == 0:
                continue

            tf_component = (
                tf
                * (k1 + 1.0)
                / denominator
            )

            score += (
                idf
                * tf_component
                * query_frequency
            )

        if score > 0.0:

            ranking.append(
                (
                    doc_index,
                    score,
                )
            )

    ranking.sort(
        key=lambda x: (
            -x[1],
            docs[x[0]].entity,
        )
    )

    return ranking, k1, b


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--instance",
        required=True,
    )

    ap.add_argument(
        "--source-package",
        required=True,
    )

    args = ap.parse_args()

    root = (
        Path.home()
        / "AdaptiveContextOpt"
    )

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    repo = (
        work / "repo"
    )

    out = (
        work
        / "outputs"
        / "boostn"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = json.loads(
        (
            work
            / "metadata.json"
        ).read_text()
    )

    # SWE-bench provides one combined issue/problem
    # statement rather than separate title/description
    # files. Use that as the BoostN bug-report query.
    #
    # No patch / gold location is read.
    query_text = metadata.get(
        "problem_statement",
        "",
    )

    if not query_text.strip():

        raise SystemExit(
            "Missing problem_statement."
        )

    stopwords = load_stopwords(
        root
    )

    query = preprocess(
        query_text,
        stopwords,
    )

    print(
        "INSTANCE:",
        args.instance,
    )

    print(
        "QUERY TOKENS:",
        len(query),
    )

    methods = collect_methods(
        repo,
        args.source_package,
        stopwords,
    )

    print()
    print(
        "METHOD CORPUS:",
        len(methods),
    )

    ranking, k1, b = bm25_rank(
        query,
        methods,
    )

    print(
        "BM25 k1:",
        k1,
    )

    print(
        "BM25 b:",
        b,
    )

    print(
        "RETRIEVED METHODS:",
        len(ranking),
    )

    top_score = (
        ranking[0][1]
        if ranking
        else 0.0
    )

    rows = []

    for rank, (
        doc_index,
        raw_score,
    ) in enumerate(
        ranking,
        1,
    ):

        doc = methods[
            doc_index
        ]

        normalized = (
            raw_score
            / top_score
            if top_score > 0
            else 0.0
        )

        rows.append(
            {
                "rank": rank,
                "entity": doc.entity,
                "file": doc.file,
                "start": doc.start,
                "end": doc.end,
                "raw_score": raw_score,
                "suspiciousness": normalized,
            }
        )

    result = {
        "instance_id":
            args.instance,

        "method":
            "BoostN",

        "level":
            "method",

        "adaptation":
            "Python SWE-bench method-level BM25 adaptation",

        "query_source":
            "problem_statement only",

        "gold_used":
            False,

        "config": {
            "method_count":
                len(methods),

            "k1":
                k1,

            "b":
                b,

            "score_normalization":
                "score / highest_score",

            "preprocessing":
                [
                    "remove non alphabetic literals",
                    "split compound identifiers",
                    "lowercase",
                    "remove stopwords",
                    "Porter stemming",
                ],
        },

        "candidate_count":
            len(rows),

        "top20":
            rows[:20],

        "top5": [
            row["entity"]
            for row in rows[:5]
        ],

        "ranking":
            rows,
    }

    output = (
        out / "boostn.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "============================================"
    )
    print(
        "BOOSTN TOP-20"
    )
    print(
        "============================================"
    )

    for row in rows[:20]:

        print(
            f"{row['rank']:2}. "
            f"{row['entity']} "
            f"score="
            f"{row['suspiciousness']:.6f}"
        )

    print()
    print(
        "============================================"
    )
    print(
        "BOOSTN TOP-5"
    )
    print(
        "============================================"
    )

    for index, entity in enumerate(
        result["top5"],
        1,
    ):

        print(
            f"Top-{index}: {entity}"
        )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()

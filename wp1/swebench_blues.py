from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------

@dataclass
class Entity:
    file: str
    name: str
    kind: str
    start: int
    end: int

    @property
    def ref(self):
        return f"{self.file}::{self.name}"


@dataclass
class Statement:
    file: str
    line: int
    text: str
    entity: str | None

    @property
    def ref(self):
        return f"{self.file}#{self.line}"


# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------

def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace(".", " ")
    return text


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    text = split_identifier(text)

    tokens = re.findall(
        r"[A-Za-z][A-Za-z0-9]*",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if len(token) > 1
        and token not in stopwords
    ]


def load_stopwords(root: Path) -> set[str]:
    candidates = [
        root / "references" / "Blues" / "stopwords",
        root
        / "references"
        / "SBIR-ReplicationPackage"
        / "FaultLocalization"
        / "src"
        / "blues"
        / "stopwords",
    ]

    for path in candidates:
        if path.exists():
            return {
                line.strip().lower()
                for line in path.read_text(
                    errors="replace"
                ).splitlines()
                if line.strip()
            }

    # minimal fallback only if original file unavailable
    return {
        "the", "a", "an", "and", "or", "of",
        "to", "in", "is", "it", "for", "on",
        "with", "as", "be", "this", "that",
    }


# ------------------------------------------------------------
# BM25-style scorer
#
# Original Blues uses Indri with:
# method:tfidf, k1=1.0, b=0.3
#
# This is a Python IR adaptation preserving those parameters.
# ------------------------------------------------------------

def rank_documents(
    query_tokens: list[str],
    docs: list[list[str]],
    *,
    k1: float = 1.0,
    b: float = 0.3,
) -> list[tuple[int, float]]:

    n_docs = len(docs)

    if n_docs == 0:
        return []

    avg_len = (
        sum(len(doc) for doc in docs) / n_docs
        if n_docs
        else 1.0
    )

    df = Counter()

    for doc in docs:
        for token in set(doc):
            df[token] += 1

    query_counts = Counter(query_tokens)

    results = []

    for index, doc in enumerate(docs):
        frequencies = Counter(doc)

        dl = len(doc)

        score = 0.0

        for token, qtf in query_counts.items():

            tf = frequencies.get(token, 0)

            if tf == 0:
                continue

            document_frequency = df.get(token, 0)

            # Standard positive BM25 IDF.
            idf = math.log(
                1.0
                + (
                    n_docs
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
                    + b * dl / max(avg_len, 1e-9)
                )
            )

            score += (
                idf
                * (
                    tf
                    * (k1 + 1.0)
                    / denominator
                )
                * qtf
            )

        if score > 0:
            results.append((index, score))

    results.sort(
        key=lambda x: (-x[1], x[0])
    )

    return results


# ------------------------------------------------------------
# Python AST corpus
# ------------------------------------------------------------

def collect_entities(
    repo: Path,
    source_package: str,
) -> list[Entity]:

    entities = []

    prefix = source_package.rstrip("/") + "/"

    for path in repo.rglob("*.py"):

        rel = path.relative_to(repo).as_posix()

        if not (
            rel == source_package
            or rel.startswith(prefix)
        ):
            continue

        if "/tests/" in f"/{rel}/":
            continue

        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue

        def walk(node, stack):

            if isinstance(node, ast.ClassDef):

                name = ".".join(
                    stack + [node.name]
                )

                entities.append(
                    Entity(
                        file=rel,
                        name=name,
                        kind="class",
                        start=int(node.lineno),
                        end=int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                    )
                )

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

                name = ".".join(
                    stack + [node.name]
                )

                entities.append(
                    Entity(
                        file=rel,
                        name=name,
                        kind="function",
                        start=int(node.lineno),
                        end=int(
                            getattr(
                                node,
                                "end_lineno",
                                node.lineno,
                            )
                        ),
                    )
                )

                for child in node.body:
                    walk(
                        child,
                        stack + [node.name],
                    )

                return

            for child in ast.iter_child_nodes(node):
                walk(child, stack)

        walk(tree, [])

    return entities


def collect_corpus(
    repo: Path,
    source_package: str,
    stopwords: set[str],
):
    """
    Fast Python adaptation of the Blues source/statement corpus.

    Important optimization:
    We assign statements to their containing class/function while
    walking the AST. We do NOT repeatedly scan every entity for
    every statement.
    """

    entities = collect_entities(
        repo,
        source_package,
    )

    file_records = []
    statements = []

    prefix = source_package.rstrip("/") + "/"

    python_files = []

    for path in repo.rglob("*.py"):

        try:
            rel = path.relative_to(repo).as_posix()
        except ValueError:
            continue

        if not (
            rel == source_package
            or rel.startswith(prefix)
        ):
            continue

        # Production code only.
        if "/tests/" in f"/{rel}/":
            continue

        python_files.append(
            (path, rel)
        )

    python_files.sort(
        key=lambda x: x[1]
    )

    total_files = len(python_files)

    print(
        "PRODUCTION PYTHON FILES TO PARSE:",
        total_files,
        flush=True,
    )

    for file_index, (path, rel) in enumerate(
        python_files,
        1,
    ):

        if (
            file_index == 1
            or file_index % 100 == 0
            or file_index == total_files
        ):
            print(
                f"Parsing file "
                f"{file_index}/{total_files}: "
                f"{rel}",
                flush=True,
            )

        try:
            source = path.read_text(
                errors="replace"
            )

            tree = ast.parse(source)

        except Exception as exc:

            print(
                f"SKIP parse error: {rel}: {exc}",
                flush=True,
            )

            continue

        source_lines = source.splitlines()

        # --------------------------------------------------
        # FILE-LEVEL DOCUMENT
        # --------------------------------------------------

        names = []
        identifiers = []
        comments = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                names.append(node.name)

                try:
                    doc = ast.get_docstring(
                        node,
                        clean=False,
                    )
                except Exception:
                    doc = None

                if doc:
                    comments.append(doc)

            elif isinstance(node, ast.Name):

                identifiers.append(node.id)

            elif isinstance(node, ast.Attribute):

                identifiers.append(node.attr)

        try:
            module_doc = ast.get_docstring(
                tree,
                clean=False,
            )
        except Exception:
            module_doc = None

        if module_doc:
            comments.append(module_doc)

        file_text = " ".join(
            [
                rel,
                " ".join(names),
                " ".join(identifiers),
                " ".join(comments),
                source,
            ]
        )

        file_records.append(
            {
                "file": rel,
                "tokens": tokenize(
                    file_text,
                    stopwords,
                ),
            }
        )

        # --------------------------------------------------
        # STATEMENT-LEVEL DOCUMENTS
        # --------------------------------------------------

        seen_lines = set()

        def statement_text(node):

            start_line = getattr(
                node,
                "lineno",
                None,
            )

            if start_line is None:
                return ""

            end_line = getattr(
                node,
                "end_lineno",
                start_line,
            )

            start_line = int(start_line)
            end_line = int(end_line or start_line)

            if start_line < 1:
                return ""

            if start_line > len(source_lines):
                return ""

            end_line = min(
                end_line,
                len(source_lines),
            )

            # Much faster than ast.get_source_segment().
            return " ".join(
                line.strip()
                for line in source_lines[
                    start_line - 1:end_line
                ]
                if line.strip()
            )


        def walk_statements(
            nodes,
            stack,
            current_entity,
        ):

            for node in nodes:

                # ------------------------------
                # Class
                # ------------------------------

                if isinstance(node, ast.ClassDef):

                    class_stack = (
                        stack + [node.name]
                    )

                    class_ref = (
                        f"{rel}::"
                        + ".".join(class_stack)
                    )

                    # Statements directly inside the class
                    # belong to the class entity.
                    walk_statements(
                        node.body,
                        class_stack,
                        class_ref,
                    )

                    continue

                # ------------------------------
                # Function / method
                # ------------------------------

                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):

                    function_stack = (
                        stack + [node.name]
                    )

                    function_ref = (
                        f"{rel}::"
                        + ".".join(function_stack)
                    )

                    walk_statements(
                        node.body,
                        function_stack,
                        function_ref,
                    )

                    continue

                # ------------------------------
                # Regular executable statement
                # ------------------------------

                if isinstance(node, ast.stmt):

                    line = getattr(
                        node,
                        "lineno",
                        None,
                    )

                    if line is not None:

                        line = int(line)

                        key = (rel, line)

                        # One statement document per source
                        # line, avoiding duplicate AST nodes.
                        if key not in seen_lines:

                            seen_lines.add(key)

                            text_value = statement_text(
                                node
                            )

                            if text_value:

                                statement_document = (
                                    " ".join(
                                        [
                                            rel,
                                            current_entity or "",
                                            text_value,
                                        ]
                                    )
                                )

                                statements.append(
                                    Statement(
                                        file=rel,
                                        line=line,
                                        text=statement_document,
                                        entity=current_entity,
                                    )
                                )

                # ------------------------------
                # Recurse into compound statement
                # children without changing entity.
                # ------------------------------

                child_statement_lists = []

                for field_name in (
                    "body",
                    "orelse",
                    "finalbody",
                ):
                    value = getattr(
                        node,
                        field_name,
                        None,
                    )

                    if isinstance(value, list):
                        child_statement_lists.append(
                            value
                        )

                handlers = getattr(
                    node,
                    "handlers",
                    None,
                )

                if isinstance(handlers, list):

                    for handler in handlers:

                        body = getattr(
                            handler,
                            "body",
                            None,
                        )

                        if isinstance(body, list):
                            child_statement_lists.append(
                                body
                            )

                for child_list in child_statement_lists:

                    walk_statements(
                        child_list,
                        stack,
                        current_entity,
                    )


        walk_statements(
            tree.body,
            [],
            None,
        )

    print(
        "CORPUS BUILD COMPLETE",
        flush=True,
    )

    return (
        entities,
        file_records,
        statements,
    )


# ------------------------------------------------------------
# Blues configurations
# ------------------------------------------------------------

def normalize_ranked_items(items):

    if not items:
        return {}

    n = len(items)

    decrement = 1.0 / n

    score = 1.0

    output = {}

    for key, _ in items:

        output[key] = score

        score -= decrement

    return output


def create_configuration(
    file_ranking,
    statement_ranking,
    statements_by_ref,
    *,
    m,
    weighted,
):

    selected = []

    seen = set()

    maxscore = (
        len(file_ranking)
        * len(statement_ranking)
    )

    indexscore = float(maxscore)

    for file_ref, file_score in file_ranking:

        used_for_file = 0

        for stmt_ref, stmt_score in statement_ranking:

            stmt = statements_by_ref[stmt_ref]

            if stmt.file != file_ref:
                continue

            if stmt_ref in seen:
                continue

            seen.add(stmt_ref)

            if weighted:
                score = file_score * stmt_score
            else:
                score = (
                    indexscore / maxscore
                    if maxscore
                    else 0.0
                )

            if used_for_file < m:
                selected.append(
                    (
                        stmt_ref,
                        score,
                    )
                )

                used_for_file += 1

            indexscore -= 1.0

    selected.sort(
        key=lambda x: (-x[1], x[0])
    )

    return selected


def combine_configurations(configurations):

    best_score = {}

    votes = Counter()

    first_seen = {}

    order_counter = 0

    for config in configurations:

        normalized = normalize_ranked_items(
            config
        )

        for statement, score in normalized.items():

            if statement not in first_seen:
                first_seen[statement] = order_counter
                order_counter += 1

            votes[statement] += 1

            if (
                statement not in best_score
                or score > best_score[statement]
            ):
                best_score[statement] = score

    ranked = sorted(
        best_score,
        key=lambda x: (
            -best_score[x],
            -votes[x],
            first_seen[x],
            x,
        ),
    )

    return [
        (
            statement,
            best_score[statement],
            votes[statement],
        )
        for statement in ranked
    ]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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

    root = Path.home() / "AdaptiveContextOpt"

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    repo = work / "repo"

    out = work / "outputs" / "blues"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = json.loads(
        (work / "metadata.json").read_text()
    )

    # IMPORTANT:
    # Problem statement only.
    # No gold patch and no changed-file metadata.
    query_text = metadata.get(
        "problem_statement",
        "",
    )

    if not query_text.strip():
        raise SystemExit(
            "Missing SWE-bench problem_statement."
        )

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

    print("PRODUCTION FILES:", len(files))
    print("PRODUCTION ENTITIES:", len(entities))
    print("STATEMENTS:", len(statements))

    # ----------------------------------------
    # File retrieval
    # ----------------------------------------

    file_docs = [
        row["tokens"]
        for row in files
    ]

    raw_file_scores = rank_documents(
        query_tokens,
        file_docs,
        k1=1.0,
        b=0.3,
    )

    file_ranking = [
        (
            files[index]["file"],
            score,
        )
        for index, score in raw_file_scores[:50]
    ]

    # ----------------------------------------
    # Statement retrieval
    # ----------------------------------------

    statement_docs = [
        tokenize(
            statement.text,
            stopwords,
        )
        for statement in statements
    ]

    raw_statement_scores = rank_documents(
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
        in raw_statement_scores[:10000]
    ]

    statements_by_ref = {
        statement.ref: statement
        for statement in statements
    }

    # ----------------------------------------
    # Original Blues-style six configs
    # ----------------------------------------

    configurations = []

    config_names = []

    for m in [1, 25, 50, 100, 10000]:

        config = create_configuration(
            file_ranking,
            statement_ranking,
            statements_by_ref,
            m=m,
            weighted=False,
        )

        configurations.append(config)

        config_names.append(
            "mAll" if m == 10000 else f"m{m}"
        )

    weighted = create_configuration(
        file_ranking,
        statement_ranking,
        statements_by_ref,
        m=10000,
        weighted=True,
    )

    configurations.append(weighted)

    config_names.append("Wted")

    # ----------------------------------------
    # Blues ensemble
    # ----------------------------------------

    ensemble = combine_configurations(
        configurations
    )

    # ----------------------------------------
    # Statement -> entity ranking
    #
    # Same purpose as FlexFL filter_SBIR.py:
    # convert statement-level FL to method/entity FL.
    # ----------------------------------------

    entity_rank = []

    seen_entities = set()

    for statement_ref, score, votes in ensemble:

        statement = statements_by_ref.get(
            statement_ref
        )

        if statement is None:
            continue

        entity = statement.entity

        if not entity:
            continue

        if entity in seen_entities:
            continue

        seen_entities.add(entity)

        entity_rank.append(
            {
                "entity": entity,
                "statement": statement_ref,
                "ensemble_score": score,
                "votes": votes,
            }
        )

    result = {
        "instance_id": args.instance,
        "method": "Blues",
        "adaptation": (
            "Python SWE-bench Blues adaptation"
        ),
        "query_source": "problem_statement only",
        "gold_used": False,
        "retrieval": {
            "k1": 1.0,
            "b": 0.3,
            "top_files": 50,
            "top_statements": 10000,
        },
        "configurations": config_names,
        "counts": {
            "files": len(files),
            "entities": len(entities),
            "statements": len(statements),
            "retrieved_files": len(file_ranking),
            "retrieved_statements": len(
                statement_ranking
            ),
            "ensemble_statements": len(
                ensemble
            ),
            "entity_candidates": len(
                entity_rank
            ),
        },
        "top50_files": [
            {
                "file": file,
                "score": score,
            }
            for file, score
            in file_ranking
        ],
        "top20": entity_rank[:20],
        "top5": [
            row["entity"]
            for row in entity_rank[:5]
        ],
    }

    (out / "blues.json").write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    # ----------------------------------------
    # Diagnostics
    # ----------------------------------------

    print()
    print(
        "=================================================="
    )
    print("BLUES TOP-20 ENTITIES")
    print(
        "=================================================="
    )

    for index, row in enumerate(
        entity_rank[:20],
        1,
    ):
        print(
            f"{index:2}. "
            f"{row['entity']} "
            f"score={row['ensemble_score']:.6f} "
            f"votes={row['votes']} "
            f"stmt={row['statement']}"
        )

    print()
    print(
        "=================================================="
    )
    print("BLUES TOP-5")
    print(
        "=================================================="
    )

    for index, entity in enumerate(
        result["top5"],
        1,
    ):
        print(
            f"Top-{index}: {entity}"
        )

    print()
    print("Saved:", out / "blues.json")


if __name__ == "__main__":
    main()

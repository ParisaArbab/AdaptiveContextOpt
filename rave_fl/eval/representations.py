"""Language-specific structural representations and gold-free source indexing."""
import ast
import os
from dataclasses import replace
from pathlib import Path
import re
import textwrap

from rave_fl.code_sketch import FEATURE_NAMES, java_sketch, words
from rave_fl.core import approx_tokens
from rave_fl.eval.schema import CodeEntity, CompatibilityError, beneath
from rave_fl.pilot_types import PilotCandidate

EXTENSIONS = {
    "python": ".py",
    "java": ".java",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "rust": ".rs",
    "c": ".c",
    "cpp": ".cpp",
    "ruby": ".rb",
    "php": ".php",
}
EXCLUDE = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    "vendor",
    "generated",
    "tests",
    "test",
    "testing",
    "__pycache__",
}


def python_sketch(ident, source, max_chars=2200, file_level=False):
    try:
        tree = ast.parse(textwrap.dedent(source))
        nodes = list(ast.walk(tree))
        definitions = [
            n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        signatures = []
        for n in definitions:
            signatures.append(
                ("class " + n.name + "(" + ",".join(ast.unparse(b) for b in n.bases) + ")")
                if isinstance(n, ast.ClassDef)
                else ("async " if isinstance(n, ast.AsyncFunctionDef) else "")
                + "def "
                + n.name
                + "("
                + ast.unparse(n.args)
                + ")"
            )
        l0 = (
            "ENTITY: "
            + ident
            + "\nSIGNATURES:\n"
            + "\n".join(signatures if file_level else signatures[:1])
        )
        selected = []
        for n in nodes:
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                value = getattr(n, "test", getattr(n, "iter", getattr(n, "type", None)))
                selected.append(type(n).__name__ + ": " + (ast.unparse(value) if value else "all"))
            elif isinstance(
                n,
                (
                    ast.Return,
                    ast.Raise,
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                selected.append(ast.unparse(n))
            elif isinstance(n, ast.Call):
                selected.append("CALL " + ast.unparse(n))
        sketch = l0 + "\nSTRUCTURE (lossy Python AST):\n" + "\n".join(dict.fromkeys(selected))
        truncated = len(sketch) > max_chars
        if truncated:
            sketch = sketch[:max_chars] + "\n[SKETCH TRUNCATED]"
        return (
            (l0, sketch, source),
            {
                "parser": "python-ast",
                "parse_error": False,
                "truncated": truncated,
                "complexity": 1
                + sum(
                    isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)) for n in nodes
                ),
                "call_count": sum(isinstance(n, ast.Call) for n in nodes),
                "literal_count": sum(isinstance(n, ast.Constant) for n in nodes),
            },
        )
    except (SyntaxError, ValueError, RecursionError):
        return (
            ("ENTITY: " + ident, "ENTITY: " + ident + "\n[PARSE ERROR; Full required]", source),
            {
                "parser": "python-ast",
                "parse_error": True,
                "truncated": False,
                "complexity": 1,
                "call_count": 0,
                "literal_count": 0,
            },
        )


def java_tree(source):
    from tree_sitter import Language, Parser
    import tree_sitter_java

    return Parser(Language(tree_sitter_java.language())).parse(source.encode()).root_node


def java_file_sketch(ident, source, max_chars):
    root = java_tree(source)
    data = source.encode()
    nodes = []
    pending = [root]
    while pending:
        n = pending.pop()
        nodes.append(n)
        pending.extend(reversed(n.named_children))
    signatures = []
    selected = []
    for n in nodes:
        if n.type in {
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
        }:
            body = n.child_by_field_name("body")
            signatures.append(
                data[n.start_byte : body.start_byte if body else n.end_byte].decode().strip()
            )
        if n.type in {
            "if_statement",
            "return_statement",
            "throw_statement",
            "method_invocation",
            "assignment_expression",
            "field_declaration",
        }:
            selected.append(re.sub(r"\s+", " ", data[n.start_byte : n.end_byte].decode()))
    l0 = "FILE: " + ident + "\nSIGNATURES:\n" + "\n".join(signatures)
    l1 = l0 + "\nSTRUCTURE (lossy Java AST):\n" + "\n".join(dict.fromkeys(selected))
    truncated = len(l1) > max_chars
    if truncated:
        l1 = l1[:max_chars] + "\n[SKETCH TRUNCATED]"
    return (
        (l0, l1, source),
        {
            "parser": "tree-sitter-java-file",
            "parse_error": root.has_error,
            "truncated": truncated,
            "complexity": 1 + sum(n.type == "if_statement" for n in nodes),
            "call_count": sum(n.type == "method_invocation" for n in nodes),
            "literal_count": sum("literal" in n.type for n in nodes),
        },
    )


def index_source(instance, granularity, max_files=3000, max_bytes=30_000_000):
    if instance.language not in EXTENSIONS:
        raise CompatibilityError("No source extension registered for language " + instance.language)
    root = Path(instance.checkout).resolve()
    source_root = root if instance.source_root == "." else beneath(root, instance.source_root)
    if not source_root.is_dir():
        raise CompatibilityError("Source root unavailable: " + str(source_root))
    files = []
    total_bytes = 0
    for directory, dirs, names in os.walk(source_root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs if d.lower() not in EXCLUDE and not (Path(directory) / d).is_symlink()
        )
        for name in sorted(names):
            if (
                not name.endswith(EXTENSIONS[instance.language])
                or name.startswith("test_")
                or name.endswith("_test.py")
            ):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                continue
            files.append(path)
            total_bytes += path.stat().st_size
            if len(files) > max_files or total_bytes > max_bytes:
                raise CompatibilityError(
                    "Source index exceeds --max-source-files/bytes; narrow --source-root"
                )
    files.sort()
    entities = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        path = beneath(root, rel)
        source = path.read_text()
        lines = source.splitlines(keepends=True)
        if not source.strip():
            continue
        if granularity == "file":
            entities.append(CodeEntity(rel, rel, 1, len(lines), source, 0.5, kind="file"))
            continue
        if instance.language == "python":
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            def walk(body, parents=()):
                for n in body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
                        name = ".".join((*parents, n.name))
                        entities.append(
                            CodeEntity(
                                rel + "::" + name,
                                rel,
                                start,
                                n.end_lineno,
                                textwrap.dedent("".join(lines[start - 1 : n.end_lineno])),
                                0.5,
                                kind="class" if isinstance(n, ast.ClassDef) else "function",
                            )
                        )
                        walk(n.body, (*parents, n.name))

            walk(tree.body)
        elif instance.language == "java":
            tree = java_tree(source)
            data = source.encode()

            def walk_java(n, parents=()):
                current = parents
                if n.type in {"class_declaration", "interface_declaration", "enum_declaration"}:
                    name = n.child_by_field_name("name")
                    current = (
                        (*parents, data[name.start_byte : name.end_byte].decode())
                        if name
                        else parents
                    )
                if n.type in {"method_declaration", "constructor_declaration"}:
                    name = n.child_by_field_name("name")
                    params = n.child_by_field_name("parameters")
                    ident = (
                        rel
                        + "::"
                        + ".".join((*parents, data[name.start_byte : name.end_byte].decode()))
                        + data[params.start_byte : params.end_byte].decode()
                    )
                    entities.append(
                        CodeEntity(
                            ident,
                            rel,
                            n.start_point.row + 1,
                            n.end_point.row + 1,
                            data[n.start_byte : n.end_byte].decode(),
                            0.5,
                            kind="method",
                        )
                    )
                for child in n.named_children:
                    walk_java(child, current)

            walk_java(tree)
        else:
            raise CompatibilityError(
                "Method indexing unsupported for "
                + instance.language
                + "; use file granularity and full representation"
            )
    return entities


def prepare_entities(instance, entities, representation="l1", max_chars=2200):
    prepared = []
    metadata = {}
    depths = {"SBIR": 10, "Ochiai": 10, "BoostN": 10}
    if not entities or len({e.entity_id for e in entities}) != len(entities):
        raise CompatibilityError("No unique candidate pool")
    for pos, e in enumerate(entities, 1):
        if representation == "full":
            levels = ("ENTITY: " + e.entity_id, e.source, e.source)
            m = {
                "parser": "full-source-no-AST",
                "parse_error": False,
                "truncated": False,
                "complexity": 1,
                "call_count": 0,
                "literal_count": 0,
            }
        elif instance.language == "java":
            levels, m = (
                java_file_sketch(e.entity_id, e.source, max_chars)
                if e.kind == "file"
                else java_sketch(e.entity_id, e.source, max_chars)
            )
        elif instance.language == "python":
            levels, m = python_sketch(e.entity_id, e.source, max_chars, e.kind == "file")
        else:
            raise CompatibilityError(
                "Structural representation unsupported for " + instance.language
            )
        terms = words(e.entity_id + "\n" + e.source)
        ranks = list(e.ranks.values())
        agreement = len(e.ranks) / 3
        f = {
            k: 0.0
            for k in FEATURE_NAMES
            if k
            not in {
                "level",
                "suspicion",
                "uncertainty",
                "valid_feedback",
                "remaining_fraction",
                "inspected_fraction",
            }
        }
        f.update(
            prior=e.prior,
            rank_fraction=pos / len(entities),
            ranking_gap=max(0, e.prior - (entities[pos].prior if pos < len(entities) else 0)),
            ranker_agreement=agreement,
            ranker_disagreement=(max(ranks) - min(ranks)) / max(10, max(ranks))
            if len(ranks) > 1
            else 1 - agreement,
            failure_relevance=len(terms & words(instance.evidence.text)) / max(1, len(terms)),
            trace_match=0.0,
            complexity=float(m["complexity"]),
            call_count=float(m["call_count"]),
            literal_count=float(m["literal_count"]),
            source_tokens=float(approx_tokens(e.source)),
            l0_tokens=float(approx_tokens(levels[0])),
            l1_tokens=float(approx_tokens(levels[1])),
            compression_ratio=min(1, approx_tokens(levels[1]) / max(1, approx_tokens(e.source))),
            parse_error=float(m["parse_error"]),
            sketch_truncated=float(m["truncated"]),
        )
        prepared.append(PilotCandidate(e.entity_id, e.source, e.prior, pos, levels, f, m))
        metadata[e.entity_id] = {"ranks": e.ranks}
        for ranker, rank in e.ranks.items():
            if ranker in depths:
                depths[ranker] = max(depths[ranker], rank)
    return prepared, metadata, depths

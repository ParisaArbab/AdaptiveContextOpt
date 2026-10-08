"""Dependency-guided static backward slicing for Python SWE-bench tests.

This module is intentionally small and deterministic. It implements an
intra-procedural backward slice over a failing Python test function:

1. choose the failing assertion/raise (or a supplied criterion line),
2. collect names used by that criterion,
3. walk earlier statements backward,
4. retain statements that define currently-needed names,
5. retain enclosing control statements and relevant imports.

It is a research prototype, not a full interprocedural PDG/SDG slicer. The
important property for our experiments is that selection is based on program
dependencies rather than token entropy.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SliceLine:
    line: int
    text: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SliceResult:
    source_file: str
    function_name: str
    criterion_line: int
    criterion_text: str
    selected_lines: list[SliceLine]
    required_names: list[str]
    mode: str = "static_backward_intra_procedural"

    @property
    def text(self) -> str:
        return "\n".join(
            f"{item.line:>5}: {item.text}" for item in self.selected_lines
        )

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "function_name": self.function_name,
            "criterion_line": self.criterion_line,
            "criterion_text": self.criterion_text,
            "selected_lines": [line.to_dict() for line in self.selected_lines],
            "required_names": self.required_names,
            "mode": self.mode,
        }


def parse_fail_to_pass(value: str) -> tuple[str, str]:
    """Return (python_file, test_function) from a full SWE-bench test id."""
    raw = (value or "").strip()
    if "::" not in raw:
        raise ValueError(f"Expected SWE-bench test id with '::': {raw!r}")
    path, tail = raw.split("::", 1)
    function = tail.split("[", 1)[0].split("::")[-1]
    if not path.endswith(".py"):
        raise ValueError(f"Only Python slicing is supported: {path}")
    if not function:
        raise ValueError(f"Could not resolve test function from: {raw!r}")
    return path, function


def resolve_test_target(repo: Path, test_id: str) -> tuple[str, str]:
    """Resolve either a full test id or a bare test function name."""
    raw = (test_id or "").strip()

    if "::" in raw:
        return parse_fail_to_pass(raw)

    function = raw.split("[", 1)[0]
    if not function:
        raise ValueError(f"Empty test id: {test_id!r}")

    matches: list[str] = []
    for path in Path(repo).rglob("*.py"):
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue

        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
            for node in ast.walk(tree)
        ):
            try:
                matches.append(path.relative_to(repo).as_posix())
            except ValueError:
                continue

    if not matches:
        raise ValueError(
            f"Could not find test function {function!r} in repository"
        )

    test_matches = [
        candidate for candidate in matches
        if "/test" in f"/{candidate}" or candidate.startswith("test")
    ]
    candidates = sorted(test_matches or matches)

    if len(candidates) != 1:
        raise ValueError(
            f"Test function {function!r} is ambiguous; matches: {candidates}"
        )

    return candidates[0], function


def _names(node: ast.AST, ctx_type: type[ast.expr_context]) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ctx_type):
            out.add(child.id)
    return out


def _defs_uses(node: ast.AST) -> tuple[set[str], set[str]]:
    defs = _names(node, ast.Store)
    uses = _names(node, ast.Load)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs.add(node.name)
    if isinstance(node, ast.Import):
        for alias in node.names:
            defs.add(alias.asname or alias.name.split(".", 1)[0])
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            defs.add(alias.asname or alias.name)

    return defs, uses


def _statement_nodes(function: ast.AST) -> list[ast.stmt]:
    statements = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.stmt)
        and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    statements.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)))
    return statements


def _choose_criterion(
    function: ast.AST,
    *,
    criterion_line: int | None,
) -> ast.stmt:
    statements = _statement_nodes(function)

    if criterion_line is not None:
        covering = [
            stmt
            for stmt in statements
            if getattr(stmt, "lineno", 0)
            <= criterion_line
            <= getattr(stmt, "end_lineno", getattr(stmt, "lineno", 0))
        ]
        if covering:
            covering.sort(
                key=lambda s: (
                    getattr(s, "end_lineno", getattr(s, "lineno", 0))
                    - getattr(s, "lineno", 0),
                    -getattr(s, "lineno", 0),
                )
            )
            return covering[0]

    preferred = [
        stmt
        for stmt in statements
        if isinstance(stmt, (ast.Assert, ast.Raise))
    ]
    if preferred:
        return preferred[-1]
    if statements:
        return statements[-1]
    raise ValueError("Failing test function contains no statements")


def _find_function(tree: ast.AST, function_name: str) -> ast.AST:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if not matches:
        raise ValueError(f"Function not found: {function_name}")
    matches.sort(key=lambda n: getattr(n, "lineno", 0))
    return matches[0]


def _covering_control_statements(
    statements: Iterable[ast.stmt],
    selected_lines: set[int],
) -> list[ast.stmt]:
    controls: list[ast.stmt] = []
    control_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)

    for stmt in statements:
        if not isinstance(stmt, control_types):
            continue
        start = getattr(stmt, "lineno", 0)
        end = getattr(stmt, "end_lineno", start)
        if any(start <= line <= end for line in selected_lines):
            controls.append(stmt)
    return controls


def backward_slice_python(
    source: str,
    *,
    source_file: str,
    function_name: str,
    criterion_line: int | None = None,
) -> SliceResult:
    """Compute a deterministic intra-procedural static backward slice."""
    tree = ast.parse(source)
    function = _find_function(tree, function_name)
    criterion = _choose_criterion(function, criterion_line=criterion_line)

    criterion_start = getattr(criterion, "lineno", 0)
    criterion_end = getattr(criterion, "end_lineno", criterion_start)

    statements = _statement_nodes(function)
    candidates = [
        stmt
        for stmt in statements
        if getattr(stmt, "lineno", 0) <= criterion_start
    ]

    _, criterion_uses = _defs_uses(criterion)
    required = set(criterion_uses)
    selected: dict[int, str] = {}

    for line in range(criterion_start, criterion_end + 1):
        selected[line] = "slicing criterion"

    for stmt in sorted(candidates, key=lambda n: getattr(n, "lineno", 0), reverse=True):
        if stmt is criterion:
            continue
        defs, uses = _defs_uses(stmt)
        if defs & required:
            start = getattr(stmt, "lineno", 0)
            end = getattr(stmt, "end_lineno", start)
            for line in range(start, end + 1):
                selected.setdefault(line, "defines value used by failure")
            required.difference_update(defs)
            required.update(uses)

    # Keep enclosing control structure headers for statements already selected.
    controls = _covering_control_statements(statements, set(selected))
    for stmt in controls:
        selected.setdefault(getattr(stmt, "lineno", 0), "controls selected statement")
        _, uses = _defs_uses(stmt)
        required.update(uses)

    # Keep module imports that provide names required by the slice.
    for stmt in getattr(tree, "body", []):
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        defs, _ = _defs_uses(stmt)
        if defs & required:
            start = getattr(stmt, "lineno", 0)
            end = getattr(stmt, "end_lineno", start)
            for line in range(start, end + 1):
                selected.setdefault(line, "imports symbol used by slice")

    lines = source.splitlines()
    rendered: list[SliceLine] = []
    for line_no in sorted(selected):
        if 1 <= line_no <= len(lines):
            rendered.append(
                SliceLine(
                    line=line_no,
                    text=lines[line_no - 1],
                    reason=selected[line_no],
                )
            )

    criterion_text = "\n".join(
        lines[i - 1]
        for i in range(criterion_start, min(criterion_end, len(lines)) + 1)
    )

    return SliceResult(
        source_file=source_file,
        function_name=function_name,
        criterion_line=criterion_start,
        criterion_text=criterion_text,
        selected_lines=rendered,
        required_names=sorted(required),
    )


def slice_test_file(
    repo: Path,
    test_id: str,
    *,
    criterion_line: int | None = None,
) -> SliceResult:
    path, function = resolve_test_target(repo, test_id)
    absolute = Path(repo) / path
    if not absolute.exists():
        raise FileNotFoundError(absolute)
    source = absolute.read_text(errors="replace")
    return backward_slice_python(
        source,
        source_file=path,
        function_name=function,
        criterion_line=criterion_line,
    )

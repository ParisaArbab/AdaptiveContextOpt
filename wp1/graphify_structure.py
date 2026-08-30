"""Graphify integration used as the repository structure layer for Agent4SR/LR.

Graphify is run once per Defects4J checkout. Both RAW and LeanCTX conditions use
exactly the same graph, so the only experimental variable is the tool output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    source_file: str
    line: int | None = None
    community: int | str | None = None
    callable: bool = False
    callable_class: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class GraphifyIndex:
    def __init__(self, repo: Path, nodes: list[GraphNode]):
        self.repo = Path(repo)
        self.nodes = nodes

    @classmethod
    def build(cls, repo: Path, force: bool = False, timeout: int = 900) -> "GraphifyIndex":
        repo = Path(repo)
        graph_json = repo / "graphify-out" / "graph.json"
        if force or not graph_json.exists():
            if not shutil.which("graphify"):
                raise RuntimeError(
                    "graphify is not installed. Install the graphifyy package first."
                )
            proc = subprocess.run(
                ["graphify", "extract", str(repo), "--code-only", "--no-viz"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"graphify extract failed:\n{proc.stdout[-10000:]}")
        if not graph_json.exists():
            raise RuntimeError(f"Graphify completed but graph is missing: {graph_json}")
        return cls.from_json(repo, graph_json)

    @classmethod
    def from_json(cls, repo: Path, graph_json: Path) -> "GraphifyIndex":
        graph = json.loads(Path(graph_json).read_text(errors="replace"))
        raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        nodes: list[GraphNode] = []
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            attrs = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
            data = {**raw, **attrs}
            node_id = str(data.get("id") or data.get("key") or "")
            label = str(
                data.get("label")
                or data.get("qualified_name")
                or data.get("qualifiedName")
                or data.get("name")
                or ""
            )
            source_file = str(
                data.get("source_file")
                or data.get("file")
                or data.get("file_path")
                or data.get("path")
                or ""
            )
            if not source_file or not label:
                continue
            loc = data.get("source_location") or data.get("line") or data.get("start_line")
            line = _parse_line(loc)
            kind = str(data.get("type") or data.get("kind") or "").lower()
            callable_flag = bool(data.get("_callable")) or any(
                word in kind for word in ("method", "function", "constructor")
            )
            nodes.append(
                GraphNode(
                    node_id=node_id,
                    label=label,
                    source_file=source_file,
                    line=line,
                    community=data.get("community"),
                    callable=callable_flag,
                    callable_class=data.get("_callable_class"),
                )
            )
        return cls(Path(repo), nodes)

    def save_compact(self, dest: Path) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps([n.to_dict() for n in self.nodes], indent=2))

    def find_paths(self, query: str, limit: int = 20) -> list[str]:
        q = query.lower().strip()
        paths = sorted({n.source_file for n in self.nodes})
        if not q:
            return paths[:limit]
        exactish = [p for p in paths if q in p.lower()]
        return exactish[:limit]

    def find_classes(self, query: str, limit: int = 20) -> list[str]:
        q = query.lower().strip()
        out: list[str] = []
        seen: set[str] = set()
        for n in self.nodes:
            candidates = [n.callable_class, n.label]
            for value in candidates:
                if not value:
                    continue
                value = str(value)
                if q and q not in value.lower():
                    continue
                classish = _class_part(value)
                if classish and classish not in seen:
                    seen.add(classish)
                    out.append(classish)
                    if len(out) >= limit:
                        return out
        return out

    def find_methods(self, query: str, limit: int = 30) -> list[str]:
        q = _norm(query)
        ranked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for n in self.nodes:
            if not n.callable and "(" not in n.label:
                continue
            label = n.label
            nl = _norm(label)
            score = 0
            if q:
                if nl == q:
                    score = 100
                elif q in nl:
                    score = 80
                elif all(piece in nl for piece in q.split(".")[-2:]):
                    score = 50
                else:
                    continue
            if label in seen:
                continue
            seen.add(label)
            ranked.append((score, label))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return [v for _, v in ranked[:limit]]

    def snippet(self, method_ref: str, radius: int = 35) -> str:
        """
        Return source code for a Graphify entity.

        Preferred SWE-bench format:

            path/to/file.py::Entity
            path/to/file.py::Class.method
            path/to/file.py::<module>

        IMPORTANT:
        When a file path is supplied, resolution is restricted to that
        exact file. We never fuzzy-match an entity into another file.
        """

        ref = (method_ref or "").strip().strip("`").strip()

        # ----------------------------------------------------------
        # Helper: render an exact file around a line
        # ----------------------------------------------------------
        def render(path: Path, rel: str, line: int) -> str:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except Exception as exc:
                return f"Could not read {rel}: {exc}"

            if not lines:
                return f"{rel}:1\n<empty file>"

            line = max(1, min(int(line or 1), len(lines)))

            start_line = max(1, line - radius)
            end_line = min(len(lines), line + radius)

            body = "\n".join(
                f"{i:>5}: {lines[i-1]}"
                for i in range(start_line, end_line + 1)
            )

            return f"{rel}:{line}\n{body}"

        # ----------------------------------------------------------
        # Helper: find a Python entity directly in an exact file.
        # This is useful when a SWE-bench test patch was applied
        # after Graphify built graph.json.
        # ----------------------------------------------------------
        def python_entity_line(path: Path, entity: str) -> int | None:
            if path.suffix != ".py":
                return None

            entity = (entity or "").strip()

            if not entity or entity == "<module>":
                return 1

            clean = entity.strip().strip("`")

            if clean.endswith("()"):
                clean = clean[:-2]

            clean = clean.lstrip(".")

            try:
                import ast

                source = path.read_text(errors="replace")
                tree = ast.parse(source)

                matches = []

                def walk(node, parents):
                    node_name = getattr(node, "name", None)

                    if isinstance(
                        node,
                        (
                            ast.ClassDef,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    ):
                        qual = ".".join(parents + [node.name])

                        # Highest priority: exact qualified name.
                        if qual == clean:
                            matches.append((100, node.lineno, qual))

                        # Exact simple name.
                        elif node.name == clean:
                            matches.append((90, node.lineno, qual))

                        # Entity such as Symbol.__new__
                        elif clean.endswith("." + qual):
                            matches.append((80, node.lineno, qual))

                        # Match final component.
                        elif node.name == clean.split(".")[-1]:
                            matches.append((70, node.lineno, qual))

                    new_parents = list(parents)

                    if isinstance(node, ast.ClassDef):
                        new_parents.append(node.name)

                    for child in ast.iter_child_nodes(node):
                        walk(child, new_parents)

                walk(tree, [])

                if matches:
                    matches.sort(key=lambda x: (-x[0], x[1]))
                    return matches[0][1]

            except Exception:
                pass

            # Simple textual fallback.
            import re

            simple = clean.split(".")[-1]

            try:
                lines = path.read_text(errors="replace").splitlines()
            except Exception:
                return None

            patterns = [
                re.compile(
                    rf"^\s*class\s+{re.escape(simple)}\b"
                ),
                re.compile(
                    rf"^\s*(?:async\s+)?def\s+{re.escape(simple)}\s*\("
                ),
            ]

            for i, line in enumerate(lines, 1):
                if any(pattern.search(line) for pattern in patterns):
                    return i

            return None

        # ==========================================================
        # EXACT file::entity resolution
        # ==========================================================
        if "::" in ref:
            raw_path, entity = ref.split("::", 1)

            rel = (
                raw_path.strip()
                .replace("\\", "/")
                .lstrip("./")
            )

            entity = entity.strip()

            path = self.repo / rel

            if not path.exists() or not path.is_file():
                return f"No source file found: {rel}"

            # Security/correctness check: do not escape repo.
            try:
                path.resolve().relative_to(self.repo.resolve())
            except ValueError:
                return f"Invalid source path outside repository: {rel}"

            # First try the exact current source file.
            # This correctly finds newly-added SWE-bench test functions too.
            fs_line = python_entity_line(path, entity)

            if fs_line is not None:
                return render(path, rel, fs_line)

            # Then search Graphify nodes, but ONLY nodes from this file.
            exact_nodes = [
                n
                for n in self.nodes
                if n.source_file.replace("\\", "/").lstrip("./") == rel
            ]

            target = _norm(entity)

            best = None
            best_score = -1

            for node in exact_nodes:
                score = _similarity_score(
                    target,
                    _norm(node.label),
                )

                if score > best_score:
                    best_score = score
                    best = node

            if best is not None and best_score > 0:
                return render(
                    path,
                    rel,
                    best.line or 1,
                )

            if entity == "<module>":
                return render(path, rel, 1)

            return (
                f"No exact entity '{entity}' found in {rel}. "
                f"Importantly, no fuzzy match outside this file was used."
            )

        # ==========================================================
        # Exact file-only request
        # ==========================================================
        normalized_ref = ref.replace("\\", "/").lstrip("./")
        exact_file = self.repo / normalized_ref

        if exact_file.exists() and exact_file.is_file():
            return render(
                exact_file,
                normalized_ref,
                1,
            )

        # ==========================================================
        # Legacy Graphify fuzzy entity lookup
        #
        # Used only when caller did NOT provide a file path.
        # ==========================================================
        target = _norm(ref)

        best: GraphNode | None = None
        best_score = -1

        for n in self.nodes:
            if not n.source_file:
                continue

            label = _norm(n.label)
            score = _similarity_score(target, label)

            if score > best_score:
                best_score = score
                best = n

        if best is None or best_score <= 0:
            return self._filesystem_fallback(ref, radius)

        path = self.repo / best.source_file

        if not path.exists():
            return self._filesystem_fallback(ref, radius)

        return render(
            path,
            best.source_file,
            best.line or 1,
        )

    def _filesystem_fallback(self, method_ref: str, radius: int) -> str:
        class_name, method_name = _class_and_method(method_ref)
        simple_class = class_name.rsplit(".", 1)[-1].split("$")[0] if class_name else ""
        candidates = list(self.repo.rglob(f"{simple_class}.java")) if simple_class else []
        method_re = re.compile(rf"\b{re.escape(method_name)}\s*\(") if method_name else None
        for path in candidates[:20]:
            lines = path.read_text(errors="replace").splitlines()
            hit = 1
            if method_re:
                for i, line in enumerate(lines, 1):
                    if method_re.search(line):
                        hit = i
                        break
            start, end = max(1, hit - radius), min(len(lines), hit + radius)
            rel = path.relative_to(self.repo)
            return f"{rel}:{hit}\n" + "\n".join(
                f"{i:>5}: {lines[i-1]}" for i in range(start, end + 1)
            )
        return f"No source snippet found for {method_ref}"

    def compact_overview(self, max_entries: int = 250) -> str:
        rows = ["# Graphify repository structure"]
        for n in self.nodes[:max_entries]:
            loc = f":{n.line}" if n.line else ""
            rows.append(f"- {n.source_file}{loc} :: {n.label}")
        if len(self.nodes) > max_entries:
            rows.append(f"... {len(self.nodes)-max_entries} more graph nodes")
        return "\n".join(rows)


def _parse_line(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"(\d+)", value)
        return int(m.group(1)) if m else None
    return None


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value or "").replace("#", ".").replace("$", ".").lower()


def _similarity_score(query: str, candidate: str) -> int:
    if not query or not candidate:
        return 0
    if query == candidate:
        return 100
    if query in candidate or candidate in query:
        return 80
    q_tail = query.split("(", 1)[0].split(".")[-1]
    c_tail = candidate.split("(", 1)[0].split(".")[-1]
    return 50 if q_tail and q_tail == c_tail else 0


def _class_part(value: str) -> str:
    head = value.split("(", 1)[0]
    if "." not in head:
        return head
    return head.rsplit(".", 1)[0]


def _class_and_method(ref: str) -> tuple[str, str]:
    head = ref.split("(", 1)[0].replace("#", ".")
    if "." not in head:
        return "", head
    return head.rsplit(".", 1)

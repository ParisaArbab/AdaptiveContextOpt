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
        target = _norm(method_ref)
        best: GraphNode | None = None
        best_score = -1
        for n in self.nodes:
            if not n.source_file:
                continue
            label = _norm(n.label)
            score = _similarity_score(target, label)
            if score > best_score:
                best_score, best = score, n
        if best is None or best_score <= 0:
            return self._filesystem_fallback(method_ref, radius)
        path = self.repo / best.source_file
        if not path.exists():
            return self._filesystem_fallback(method_ref, radius)
        text = path.read_text(errors="replace").splitlines()
        line = best.line or 1
        start = max(1, line - radius)
        end = min(len(text), line + radius)
        body = "\n".join(f"{i:>5}: {text[i-1]}" for i in range(start, end + 1))
        return f"{best.source_file}:{line}\n{body}"

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

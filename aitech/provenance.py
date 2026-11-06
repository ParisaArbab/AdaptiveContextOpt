"""Source-provenance extraction from archived Agent4SR get_code_snippet results.

No gold labels, fixed correct-file path, or external repository content is used.
We only annotate verbatim evidence with the file and line explicitly printed by
the debugging tool. This is an exploratory provenance oracle for captured traces,
not a full automatic code retrieval system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

_HEADER = re.compile(r"^(sympy/[\w./-]+\.py):(\d+)\s*$")
_LINE = re.compile(r"^\s*(\d+):\s?(.*)$")


@dataclass(frozen=True)
class SourceLocation:
    path: str
    line: int


def capture_locations(trace: str, evidence: Iterable) -> dict[str, SourceLocation]:
    """Find exact source lines within archived snippet blocks.

    Fail closed for selected items with no location rather than inventing paths.
    Snippet headers identify a path and a nominal line, but the actual source
    line is taken from the numbered source line itself.
    """
    anchors = {item.text: item.id for item in evidence}
    found: dict[str, SourceLocation] = {}
    active_path: str | None = None
    for raw in trace.splitlines():
        header = _HEADER.fullmatch(raw.strip())
        if header:
            active_path = header.group(1)
            continue
        # Tool and agent banners mark the end of a snippet, preventing
        # unrelated trace text from being treated as code.
        if raw.startswith("[Agent4SR]") or raw.startswith("\\n[Agent4SR]"):
            active_path = None
        if active_path is None:
            continue
        line = _LINE.match(raw)
        if line is None:
            continue
        code = line.group(2).strip()
        identifier = anchors.get(code)
        if identifier and identifier not in found:
            found[identifier] = SourceLocation(active_path, int(line.group(1)))
    return found


def render_with_provenance(evidence: Iterable, selected_ids: Iterable[str],
                           locations: dict[str, SourceLocation]) -> str:
    """Annotate selected statements with verified origin file/line only."""
    by_id = {item.id: item for item in evidence}
    output = []
    for identifier in selected_ids:
        location = locations.get(identifier)
        if location is None:
            raise ValueError(f"No verified snippet location for {identifier}")
        if identifier not in by_id:
            raise ValueError(f"Unknown evidence ID: {identifier}")
        output.append(f"[{identifier}] [source: {location.path}:{location.line}] {by_id[identifier].text}")
    return "\n".join(output)


def resolve_relative_import(from_file: str, module: str) -> str:
    """Resolve a Python relative import without knowing the gold bug label.

    Used ONLY for offline provenance auditing, not to inject a likely answer
    path into the LLM prompt. The import's existence is not verified here.
    """
    if not module.startswith("."):
        raise ValueError("expected relative import")
    dot_count = len(module) - len(module.lstrip("."))
    package = PurePosixPath(from_file).parent
    for _ in range(dot_count - 1):
        package = package.parent
    name = module[dot_count:].replace(".", "/")
    if not name:
        raise ValueError("relative import has no module name")
    return str(package / (name + ".py"))

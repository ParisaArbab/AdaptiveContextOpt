"""Extract exact CSV line ranges from an actual checked-out Java source tree.

FlexFL CSV paths are class names. Inner classes are represented as dotted
suffixes; their method bodies live in the nearest enclosing .java file.
Fail on missing, ambiguous, or signature-mismatched source.
"""
from __future__ import annotations

from pathlib import Path


def _find_java_file(checkout: Path, class_name: str) -> Path:
    checkout = Path(checkout).resolve()
    if not checkout.is_dir():
        raise FileNotFoundError(f"Buggy source checkout not found: {checkout}")
    parts = class_name.split(".")
    candidates = []
    # Strip nested class suffixes until an actual Java file can be found.
    for n in range(len(parts), 1, -1):
        suffix = "/".join(parts[:n]) + ".java"
        matches = [
            p for p in checkout.rglob(parts[n - 1] + ".java")
            if p.as_posix().endswith("/" + suffix)
            and p.is_file()
        ]
        if matches:
            candidates = matches
            break
    # Refuse to select an arbitrary duplicate in different source roots.
    if len(candidates) > 1:
        primary = [
            p for p in candidates
            if "/src/main/java/" in p.as_posix() or "/src/java/" in p.as_posix()
        ]
        candidates = primary or candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Java source for {class_name}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def extract_method(checkout: Path, function_id: str, info: dict) -> tuple[str, str]:
    path = _find_java_file(checkout, info["file_class"])
    lines = path.read_text(errors="replace").splitlines()
    start, end = int(info["start_line"]), int(info["end_line"])
    if start < 1 or end > len(lines) or start > end:
        raise ValueError(f"{function_id}: lines {start}-{end} outside {path} ({len(lines)} lines)")
    body = "\n".join(lines[start - 1:end])
    method_name = function_id.rsplit(".", 1)[-1].split("(", 1)[0]
    if method_name not in body:
        raise ValueError(
            f"{function_id}: method name absent at CSV lines {start}-{end} in {path}; "
            "possible version or source-line mismatch"
        )
    return str(path.relative_to(Path(checkout).resolve())), body


def extract_pool_sources(checkout: Path, metadata: dict) -> tuple[dict[str, str], dict[str, str]]:
    bodies = {}
    paths = {}
    for method, info in metadata.items():
        relative, body = extract_method(checkout, method, info)
        bodies[method] = body
        paths[method] = relative
    return bodies, paths

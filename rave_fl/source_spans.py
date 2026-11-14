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
    method_name = function_id.split("(", 1)[0].rsplit(".", 1)[-1]
    if method_name not in body:
        # The official buggy SHA can be correct while a third-party CSV span
        # has a small line offset. A correction is safe only when a single
        # nearby declaration matches the method signature, never a mere call.
        import re

        expected_params = function_id.partition("(")[2].rsplit(")", 1)[0]
        expected_arity = 0 if not expected_params.strip() else len(expected_params.split(","))

        def candidate_declaration(line):
            match = re.search(
                r"\b" + re.escape(method_name) + r"\s*\(([^()]*)\)\s*(?:throws\s+[\w.,\s]+)?\s*\{",
                line,
            )
            if not match or line.lstrip().startswith(("//", "*", "@")):
                return False
            prefix = line[:match.start()].strip()
            if not prefix or prefix.endswith((".", "return", "new", "=")):
                return False
            params = match.group(1).strip()
            return (0 if not params else len(params.split(","))) == expected_arity

        nearby = [
            i for i in range(max(0, start - 12), min(len(lines), end + 10))
            if candidate_declaration(lines[i])
        ]
        if len(nearby) != 1:
            raise ValueError(
                f"{function_id}: method name absent at CSV lines {start}-{end} in {path}; "
                f"nearby matching declarations: {len(nearby)}; possible version or source-line mismatch"
            )
        declaration = nearby[0] + 1  # one-based
        # Preserve the CSV's span length. Move only as far as needed to include
        # the verified declaration, never expand far across other methods.
        shift = declaration - end if declaration > end else declaration - start
        if abs(shift) > 10:
            raise ValueError(f"{function_id}: unsafe CSV span correction")
        corrected_start, corrected_end = start + shift, end + shift
        body = "\n".join(lines[corrected_start - 1:corrected_end])
        if method_name not in body:
            raise ValueError(f"{function_id}: method absent after span correction")
    return str(path.relative_to(Path(checkout).resolve())), body


def extract_pool_sources(checkout: Path, metadata: dict) -> tuple[dict[str, str], dict[str, str]]:
    bodies = {}
    paths = {}
    for method, info in metadata.items():
        relative, body = extract_method(checkout, method, info)
        bodies[method] = body
        paths[method] = relative
    return bodies, paths

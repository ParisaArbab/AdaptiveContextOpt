"""Compiler-attributed Java dependency graph, only on complete compilable sources.

Unlike `java_ast_bridge`, this path uses javac's semantic analysis and
`Trees.getElement`. It reports static target signatures, including overload
parameter types. Dynamic dispatch and reflection remain unresolved.

It explicitly fails when the source set is missing classpath dependencies
or has compilation errors. Do not silently convert partial analysis into
a "verified" graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

JAVA_HELPER = Path(__file__).parent / "java" / "ResolvedCallExtractor.java"


@dataclass(frozen=True)
class ResolvedCall:
    source: str
    caller_owner: str
    caller_method: str
    kind: str
    target_owner: str
    target_method: str
    parameter_types: str


@dataclass(frozen=True)
class ResolutionAudit:
    calls: tuple[ResolvedCall, ...]
    unresolved_count: int
    source_files: int
    source_hashes: dict[str, str]
    status: str = "compiler_attributed_static_targets"


def audit_source_directory(source_root: Path, *, classpath: str | None = None, timeout: int = 120) -> ResolutionAudit:
    """Parse and attribute a complete Java source tree, with no paid APIs."""
    if not shutil.which("javac") or not shutil.which("java"):
        raise RuntimeError("JDK javac and java must be available")
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("source_root must be an existing directory")
    files = sorted(source_root.rglob("*.java"))
    if not files:
        raise ValueError("source_root contains no .java files")
    hashes = {
        p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    with tempfile.TemporaryDirectory(prefix="aitech_compiler_graph_") as td:
        subprocess.run(
            ["javac", "-d", td, str(JAVA_HELPER)],
            check=True, text=True, capture_output=True, timeout=timeout,
        )
        run = subprocess.run(
            ["java", "-cp", td, "ResolvedCallExtractor", str(source_root)] + ([classpath] if classpath else []),
            text=True, capture_output=True, timeout=timeout, check=False,
        )
    if run.returncode != 0:
        raise RuntimeError(
            "Compiler attribution failed, source/classpath incomplete or compiler "
            f"errors: exit={run.returncode}; {run.stderr[-1800:]}"
        )
    calls = []
    unresolved = 0
    for line in run.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "UNRESOLVED":
            unresolved += 1
        elif fields[0] == "RESOLVED" and len(fields) == 8:
            _, path, owner, method, kind, dest_owner, dest_method, param_types = fields
            calls.append(ResolvedCall(path, owner, method, kind,
                                      dest_owner, dest_method, param_types))
        else:
            raise RuntimeError(f"Unexpected compiler output: {line[:200]}")
    return ResolutionAudit(tuple(calls), unresolved, len(files), hashes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path,
                        help="Complete Java source tree including required source dependencies")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classpath", default=None, help="Defects4J export cp.compile")
    args = parser.parse_args()
    result = audit_source_directory(args.sources, classpath=args.classpath)
    record = {
        "status": result.status,
        "source_files": result.source_files,
        "source_sha256": result.source_hashes,
        "unresolved_count": result.unresolved_count,
        "calls": [call.__dict__ for call in result.calls],
        "limitations": [
            "Requires complete compilable Java sources and classpath",
            "Static resolved declaration is not necessarily runtime dynamic-dispatch target",
            "Reflection and dynamically loaded methods are not represented",
            "Not a fault-localization experiment or learned ML model",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Compiler-attributed {len(result.calls)} calls in {result.source_files} files")
    print(f"Unresolved calls: {result.unresolved_count}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

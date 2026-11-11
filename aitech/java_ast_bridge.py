"""Java compiler AST invocation extractor for archived candidate source.

Runs a local JDK only, no network or gold labels. Parsing verifies that a
source-level call expression exists and determines its argument count, but
does *not* resolve Java overloads, inheritance, dynamic dispatch or types.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


JAVA_SOURCE = Path(__file__).parent / "java" / "CallAstExtractor.java"


@dataclass(frozen=True)
class AstCall:
    caller_index: int
    kind: str
    receiver: str
    name: str
    arity: int


@dataclass(frozen=True)
class AstAudit:
    calls: tuple[AstCall, ...]
    errors: dict[int, int]
    supported: bool


def extract_ast_calls(candidates: list[dict], *, timeout: int = 90) -> AstAudit:
    """Parse archived source method snippets into invocation expressions.

    Raises clear failures for missing javac or Java, rather than returning
    misleading empty graphs. Per-snippet parser errors are also recorded.
    """
    if not shutil.which("javac") or not shutil.which("java"):
        raise RuntimeError("JDK javac and java commands are required")
    with tempfile.TemporaryDirectory(prefix="aitech_java_ast_") as td:
        root = Path(td)
        sources = root / "sources"
        sources.mkdir()
        for i, item in enumerate(candidates):
            # One independent class per candidate prevents unrelated
            # duplicate method names from corrupting AST extraction.
            path = sources / f"Snippet{i:04d}.java"
            snippet = item["source"]
            # A constructor declaration must match its class name for
            # javac to parse it as a constructor, not a malformed method.
            function_id = item["function_id"]
            owner, method = function_id.split("(", 1)[0].rsplit(".", 1)
            owner_class = owner.rsplit(".", 1)[-1]
            wrapper_name = owner_class if method == owner_class else f"Snippet{i:04d}"
            path.write_text(
                f"class {wrapper_name} {{\n" + snippet + "\n}\n",
                encoding="utf-8",
            )
        compile_result = subprocess.run(
            ["javac", "-d", str(root), str(JAVA_SOURCE)],
            text=True, capture_output=True, timeout=timeout, check=False)
        if compile_result.returncode:
            raise RuntimeError("AST helper compilation failed: " + compile_result.stderr[:1200])
        process = subprocess.run(
            ["java", "-cp", str(root), "CallAstExtractor", str(sources)],
            text=True, capture_output=True, timeout=timeout, check=False)
        if process.returncode:
            raise RuntimeError("AST helper failed: " + process.stderr[:1200])
        calls = []
        errors = {}
        for line in process.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError("Malformed AST helper output: " + line[:200])
            filename, kind, receiver, name, arity = fields
            i = int(filename.removeprefix("Snippet").removesuffix(".java"))
            calls.append(AstCall(i, kind, receiver, name, int(arity)))
        for line in process.stderr.splitlines():
            if line.startswith("PARSE_ERROR\t"):
                fields = line.split("\t")
                if len(fields) == 3:
                    i = int(fields[1].removeprefix("Snippet").removesuffix(".java"))
                    errors[i] = int(fields[2])
        return AstAudit(tuple(calls), errors, True)


def verified_syntax_candidates(candidates: list[dict], audit: AstAudit) -> dict:
    """Make strictly scoped same-owner AST candidates, abstain on ambiguity.

    The AST confirms syntactic calls, NOT resolved callee identities.
    """
    def parts(function_id: str) -> tuple[str, str, int]:
        from .java_call_audit import parts as parse
        return parse(function_id)
    signatures = [parts(x["function_id"]) for x in candidates]
    edges, ambiguous, unresolved = set(), [], []
    for call in audit.calls:
        owner, _, _ = signatures[call.caller_index]
        if call.kind == "CONSTRUCTOR":
            # Constructor overload resolution needs argument types.
            unresolved.append(f"{call.caller_index}:new {call.name}/{call.arity}")
            continue
        if call.receiver not in ("", "this", owner.rsplit(".", 1)[-1]):
            unresolved.append(f"{call.caller_index}:{call.receiver}.{call.name}/{call.arity}")
            continue
        matches = [
            j for j, (own, name, arity) in enumerate(signatures)
            if j != call.caller_index and own == owner
            and name == call.name and arity == call.arity
        ]
        if len(matches) == 1:
            edges.add((call.caller_index, matches[0]))
        elif len(matches) > 1:
            ambiguous.append(f"{call.caller_index}:{call.name}/{call.arity}")
        else:
            unresolved.append(f"{call.caller_index}:{call.name}/{call.arity}")
    return {
        "edges": sorted(edges),
        "ambiguous": sorted(set(ambiguous)),
        "unresolved": sorted(set(unresolved)),
        "parse_error_indices": sorted(audit.errors),
        "calls_extracted": len(audit.calls),
        "claim": "JDK AST verified syntax; call targets remain heuristic",
    }

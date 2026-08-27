"""
benchmarks.py — WP1 step 4: dataset + language adapter seam.

Everything that was hardcoded to SWE-bench Lite + Python lives here now, so
the same pipeline runs on swe-bench-lite / -verified / -full / -java (or any
other SWE-bench-shaped HuggingFace dataset) without touching the localizer,
the compressor, the feedback loop, or the scorer.

Two things vary per benchmark:

  1. SCHEMA — which columns carry repo / base_commit / gold patch / trigger
     tests / bug report. SWE-bench and its Lite/Verified variants share one
     schema; multi-language forks do not. `DatasetSpec.field_map` makes that
     a data question instead of a code question.
  2. LANGUAGE — how to read function/class names out of a gold patch diff,
     which file suffixes count as source, which tree-sitter grammar Graphify
     should parse with, and how to actually invoke the trigger tests.
     `LanguageAdapter` owns all four.

Adding a benchmark = adding a DATASETS entry. Adding a language = adding a
LanguageAdapter subclass and registering it in LANGUAGES.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Language adapters
# ---------------------------------------------------------------------------

class LanguageAdapter:
    """Per-language behaviour. Subclass, set the class attrs, override the
    two methods."""

    name: str = "generic"
    graphify_language: str = ""          # passed to `graphify extract --lang`
    source_suffixes: Tuple[str, ...] = ()
    # Cap on how many trigger tests get put on one command line. SWE-bench
    # FAIL_TO_PASS lists can run to hundreds of entries for parametrized
    # suites; past a couple dozen the shell command becomes the bottleneck
    # and the extra failures add nothing new to the evidence text.
    max_trigger_tests: int = 25

    def parse_patch_symbols(self, patch: str) -> Tuple[List[str], List[str]]:
        """(files_touched, ['path::SymbolName', ...]) from a unified diff."""
        raise NotImplementedError

    def build_test_command(self, repo_root: Path, fail_to_pass: Sequence[str]) -> List[str]:
        """argv that runs exactly the trigger tests and prints failure detail."""
        raise NotImplementedError

    # -- shared diff walking ------------------------------------------------

    _HUNK_RE = re.compile(r"^@@ .* @@\s*(.*)$")

    def _walk_diff(self, patch: str):
        """Yields (current_file, kind, payload) where kind is 'hunk' (payload
        is the hunk header's trailing scope text) or 'body' (payload is an
        added/removed source line, sign stripped)."""
        current_file: Optional[str] = None
        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                yield current_file, "file", current_file
                continue
            if not current_file:
                continue
            m = self._HUNK_RE.match(line)
            if m:
                if m.group(1):
                    yield current_file, "hunk", m.group(1)
                continue
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                yield current_file, "body", line[1:]

    def _collect(self, patch: str, patterns: Sequence[re.Pattern]) -> Tuple[List[str], List[str]]:
        files: List[str] = []
        functions: List[str] = []
        for current_file, kind, payload in self._walk_diff(patch):
            if kind == "file":
                if payload not in files:
                    files.append(payload)
                continue
            if self.source_suffixes and not current_file.endswith(self.source_suffixes):
                continue
            for pat in patterns:
                m = pat.search(payload)
                if m:
                    key = f"{current_file}::{m.group(1)}"
                    if key not in functions:
                        functions.append(key)
                    break
        return files, functions


class PythonAdapter(LanguageAdapter):
    name = "python"
    graphify_language = "python"
    source_suffixes = (".py",)

    _PATTERNS = (
        re.compile(r"(?:^|\s)(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"(?:^|\s)class\s+([A-Za-z_][A-Za-z0-9_]*)"),
    )

    def parse_patch_symbols(self, patch: str) -> Tuple[List[str], List[str]]:
        return self._collect(patch, self._PATTERNS)

    def build_test_command(self, repo_root: Path, fail_to_pass: Sequence[str]) -> List[str]:
        # -rA + --tb=long is what produces the assertion diffs and stack
        # frames the whole compression experiment is measuring. -p no:randomly
        # keeps repeated runs byte-comparable across conditions.
        cmd = ["python3", "-m", "pytest", "-rA", "--tb=long", "-p", "no:randomly"]
        selected = list(fail_to_pass)[: self.max_trigger_tests]
        cmd.extend(selected)
        return cmd


class JavaAdapter(LanguageAdapter):
    name = "java"
    graphify_language = "java"
    source_suffixes = (".java",)

    _PATTERNS = (
        # method declaration: modifiers, optional generics, return type, name(
        re.compile(
            r"(?:public|private|protected|static|final|abstract|synchronized|native|default)\s"
            r"[\w\s<>\[\],.?]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
        re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    )

    def parse_patch_symbols(self, patch: str) -> Tuple[List[str], List[str]]:
        return self._collect(patch, self._PATTERNS)

    def build_test_command(self, repo_root: Path, fail_to_pass: Sequence[str]) -> List[str]:
        """Java trigger-test ids arrive as `com.pkg.SomeTest.testMethod` (or
        `SomeTest#testMethod`). Maven and Gradle each want a different shape,
        so detect the build system from the checkout rather than assuming."""
        selected = list(fail_to_pass)[: self.max_trigger_tests]
        repo_root = Path(repo_root)

        if (repo_root / "pom.xml").exists():
            selectors = [self._maven_selector(t) for t in selected]
            cmd = ["mvn", "-B", "-q", "test", "-DfailIfNoTests=false",
                   "-Dsurefire.failIfNoSpecifiedTests=false"]
            if selectors:
                cmd.append("-Dtest=" + ",".join(selectors))
            return cmd

        gradlew = repo_root / "gradlew"
        launcher = ["./gradlew"] if gradlew.exists() else ["gradle"]
        cmd = launcher + ["test", "--no-daemon", "-i"]
        for t in selected:
            cmd.extend(["--tests", self._gradle_selector(t)])
        return cmd

    @staticmethod
    def _split_test_id(test_id: str) -> Tuple[str, Optional[str]]:
        if "#" in test_id:
            cls, method = test_id.split("#", 1)
            return cls, method
        parts = test_id.split(".")
        # heuristic: a segment starting uppercase is the class; anything after
        # it is the method. `com.pkg.FooTest.testBar` -> ('com.pkg.FooTest','testBar')
        for i, part in enumerate(parts):
            if part[:1].isupper() and i + 1 < len(parts):
                return ".".join(parts[: i + 1]), ".".join(parts[i + 1 :])
        return test_id, None

    @classmethod
    def _maven_selector(cls, test_id: str) -> str:
        klass, method = cls._split_test_id(test_id)
        simple = klass.rsplit(".", 1)[-1]
        return f"{simple}#{method}" if method else simple

    @classmethod
    def _gradle_selector(cls, test_id: str) -> str:
        klass, method = cls._split_test_id(test_id)
        return f"{klass}.{method}" if method else klass


LANGUAGES: Dict[str, LanguageAdapter] = {
    "python": PythonAdapter(),
    "java": JavaAdapter(),
}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DEFAULT_FIELD_MAP = {
    "instance_id": "instance_id",
    "repo": "repo",
    "base_commit": "base_commit",
    "patch": "patch",
    "test_patch": "test_patch",
    "problem_statement": "problem_statement",
    "fail_to_pass": "FAIL_TO_PASS",
    "version": "version",
}


@dataclass
class DatasetSpec:
    hf_id: str
    split: str = "test"
    language: str = "python"
    field_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_MAP))
    verified: bool = True   # False = registered for convenience, schema not confirmed here
    note: str = ""

    @property
    def adapter(self) -> LanguageAdapter:
        return LANGUAGES[self.language]

    def get(self, row: dict, logical_name: str, default=None):
        return row.get(self.field_map.get(logical_name, logical_name), default)


DATASETS: Dict[str, DatasetSpec] = {
    "swe-bench-lite": DatasetSpec("princeton-nlp/SWE-bench_Lite"),
    "swe-bench-verified": DatasetSpec("princeton-nlp/SWE-bench_Verified"),
    "swe-bench": DatasetSpec("princeton-nlp/SWE-bench"),
    # Multi-language forks: registered so `--dataset swe-bench-java` works out
    # of the box, but the schema hasn't been confirmed against the live
    # dataset from here. If a column name differs, override it with
    # --field-map fail_to_pass=<col> rather than editing this file.
    "swe-bench-java": DatasetSpec(
        "Daoguang/Multi-SWE-bench", language="java", verified=False,
        note="multi-language fork; confirm column names with --field-map if load fails",
    ),
    "swe-bench-multimodal": DatasetSpec(
        "princeton-nlp/SWE-bench_Multimodal", verified=False,
        note="schema not confirmed from here",
    ),
}


def resolve_dataset(name: str, language: Optional[str] = None,
                    field_overrides: Optional[Dict[str, str]] = None) -> DatasetSpec:
    """`name` is either a registry alias or a raw HuggingFace dataset id."""
    if name in DATASETS:
        spec = DATASETS[name]
        spec = DatasetSpec(spec.hf_id, spec.split, spec.language,
                           dict(spec.field_map), spec.verified, spec.note)
    else:
        spec = DatasetSpec(name, language=language or "python", verified=False,
                           note="unregistered dataset id, assuming SWE-bench schema")
    if language:
        spec.language = language
    if language and language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; known: {sorted(LANGUAGES)}")
    if field_overrides:
        spec.field_map.update(field_overrides)
    return spec


def coerce_list(value) -> List[str]:
    """FAIL_TO_PASS is a JSON-encoded string in some releases, a real list in
    others. Both appear in the wild across SWE-bench variants."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(v) for v in parsed] if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return [str(value)]


def parse_field_map_args(pairs: Sequence[str]) -> Dict[str, str]:
    """--field-map fail_to_pass=FAIL_TO_PASS --field-map patch=gold_patch"""
    out: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--field-map expects logical=column, got {pair!r}")
        logical, column = pair.split("=", 1)
        out[logical.strip()] = column.strip()
    return out

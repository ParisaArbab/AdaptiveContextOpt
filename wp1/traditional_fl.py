"""
traditional_fl.py — the non-LLM half of FlexFL's Stage 1.

FlexFL's Agent4SR does not work alone. Section 4.5 of the paper merges its
top-5 with top-5 from each of three traditional fault localizers before
Agent4LR ever sees a candidate list, and the replication package ships those
three as precomputed CSVs — but only for Defects4J. On SWE-bench they do not
exist, so they have to be computed, which is what this module does.

  Ochiai  — real spectrum-based fault localization (SBFL). Runs the failing
            and a bounded sample of passing tests under coverage.py with a
            per-test dynamic context, then scores every method by the
            standard Ochiai formula. This is the genuine article, not a
            stand-in.

  BoostN   — information-retrieval fault localization (IRFL). The paper uses
            BoostNSift; this is BM25 over method-level documents (qualified
            name, path segments, and the method's own source) scored against
            the bug report. Labelled a stand-in wherever it is reported,
            because BoostNSift's sifting stage is not reproduced.

  SBIR     — the paper's combined spectrum+IR ranker, reproduced here as
            reciprocal-rank fusion of the two above. Also labelled a
            stand-in.

Why bother with SBFL at all, when a stack trace is cheaper: the bug the
pilot run was pointed at, astropy__astropy-12907, fails on a boolean-matrix
assertion. Nothing in its traceback names `_cstack`, the actual buggy
function, so trace-based evidence cannot reach it by construction. Coverage
can: `_cstack` executes in the failing test and not in most passing ones.
That is precisely the gap this module closes.

Every ranker degrades to "unavailable" with a stated reason rather than to
silence. A merge missing a column is a different experiment from a merge
whose column returned nothing, and the report has to be able to tell them
apart.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")

# Test code is never the fault location. In SWE-bench the gold patch and the
# test patch are disjoint by construction — the ground truth cannot contain a
# test file — so a test entity in the candidate list is a guaranteed miss
# occupying a rank slot. It is also the entity most likely to win an IR
# ranking, because the trigger test's name usually restates the bug report
# almost verbatim: in the smoke test `test_normalize_all_zero` outranked the
# actual buggy `normalize` in both SBIR and BoostN.
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing)/"                 # tests/ or test/ directory
    r"|(^|/)test_[^/]*$"                      # test_foo.py
    r"|_test\.[a-z]+$"                        # foo_test.py / FooTest.go
    r"|(^|/)conftest\.py$"                    # pytest fixtures
    r"|(^|/)src/test/",                        # Java/Maven layout
    re.IGNORECASE,
)


def is_test_entity(key: str) -> bool:
    """True for a structure-map key that lives in test code."""
    path = key.split("::", 1)[0] if "::" in key else key
    return bool(TEST_PATH_RE.search(path.replace("\\", "/")))


def source_only(structure_map: Dict[str, dict]) -> Dict[str, dict]:
    """The candidate space: production code only.

    Returns the map unchanged if filtering would empty it, so a repository
    whose layout this heuristic misreads degrades to the old behaviour
    rather than to an empty candidate list."""
    filtered = {k: v for k, v in structure_map.items() if not is_test_entity(k)}
    return filtered or structure_map
# Terms that appear in nearly every Python bug report and carry no
# discriminating signal; keeping them lets long generic reports drown out
# the identifier that actually matters.
STOPWORDS = frozenset("""
a an the and or but if then else for while of to in on at by is are was were be been
this that these those it its as with from not no can could should would will shall may
you your we our they them he she his her i me my do does did done have has had having
error bug issue problem fail failed failure test tests expected actual result results
code python file files line lines version when what which how why use used using
""".split())


@dataclass
class RankedList:
    """One traditional FL ranker's output, plus whether it actually ran."""

    name: str
    entries: List[str] = field(default_factory=list)   # "file::Symbol", best first
    available: bool = True
    reason: str = ""                                    # why not, when unavailable
    provisional: bool = False                           # true for the stand-ins
    detail: dict = field(default_factory=dict)

    def top(self, k: int) -> List[str]:
        return self.entries[:k]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "provisional": self.provisional,
            "n_entries": len(self.entries),
            "top5": self.entries[:5],
            **({"detail": self.detail} if self.detail else {}),
        }

    @classmethod
    def unavailable(cls, name: str, reason: str) -> "RankedList":
        return cls(name=name, entries=[], available=False, reason=reason)


# ---------------------------------------------------------------------------
# Method line ranges — the bridge from covered lines back to methods
# ---------------------------------------------------------------------------

def method_line_ranges(structure_map: Dict[str, dict]) -> Dict[str, List[Tuple[str, int, int]]]:
    """{file: [(key, start_line, end_line), ...]} sorted by start line.

    Graphify records only a start line per entity. The end is taken as the
    line before the next entity in the same file, which is exact for
    sequential top-level definitions and slightly generous for nested ones.
    Being generous is the safe direction here: a method credited with a few
    lines it does not own can only blur its Ochiai score toward its
    neighbour, whereas cutting a method short would silently drop the
    coverage of its body, which is the entire signal.
    """
    by_file: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for key, meta in structure_map.items():
        line = meta.get("line")
        if meta.get("file") and isinstance(line, int):
            by_file[meta["file"]].append((key, line))

    ranges: Dict[str, List[Tuple[str, int, int]]] = {}
    for file, entries in by_file.items():
        entries.sort(key=lambda t: t[1])
        out: List[Tuple[str, int, int]] = []
        for i, (key, start) in enumerate(entries):
            end = entries[i + 1][1] - 1 if i + 1 < len(entries) else start + 400
            out.append((key, start, max(start, end)))
        ranges[file] = out
    return ranges


def _method_for_line(ranges: Sequence[Tuple[str, int, int]], line: int) -> Optional[str]:
    """Innermost enclosing entity: the last range that starts at or before
    the line and still contains it."""
    hit = None
    for key, start, end in ranges:
        if start > line:
            break
        if start <= line <= end:
            hit = key
    return hit


# ---------------------------------------------------------------------------
# Ochiai — real SBFL
# ---------------------------------------------------------------------------

def _relativize(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(Path(repo_root).resolve()))
    except (ValueError, OSError):
        return path_str.replace("\\", "/").lstrip("./")


def run_coverage(
    repo_root: Path,
    python_exe: Path,
    test_command: Sequence[str],
    timeout: int = 1800,
) -> Tuple[Optional[dict], str]:
    """Run the tests under coverage with a per-test dynamic context.

    `dynamic_context = test_function` is what makes a spectrum possible at
    all: without it coverage reports one merged set of executed lines and
    every method looks equally suspicious. With it, `coverage json
    --show-contexts` attributes each line to the individual tests that
    executed it, which is exactly the (method x test) matrix Ochiai needs.
    """
    repo_root = Path(repo_root)
    rcfile = repo_root / ".wp1coveragerc"
    rcfile.write_text(
        "[run]\n"
        "branch = False\n"
        "dynamic_context = test_function\n"
        "parallel = False\n"
        "[json]\n"
        "show_contexts = True\n"
    )
    data_file = repo_root / ".wp1coverage"
    env = {**os.environ, "COVERAGE_FILE": str(data_file)}

    cmd = list(test_command)
    # test_command starts with the interpreter; splice coverage in after it
    # so the runner (pytest, runtests.py, bin/test) is what gets measured.
    if cmd and Path(cmd[0]).name.startswith("python"):
        run_cmd = [cmd[0], "-m", "coverage", "run", f"--rcfile={rcfile}", *cmd[1:]]
    else:
        run_cmd = [str(python_exe), "-m", "coverage", "run", f"--rcfile={rcfile}", *cmd]

    try:
        subprocess.run(run_cmd, cwd=repo_root, env=env, capture_output=True,
                       text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"coverage run timed out after {timeout}s"
    except FileNotFoundError as e:
        return None, f"coverage runner not found: {e}"

    if not data_file.exists():
        return None, "coverage produced no data file (the test command likely never started)"

    json_out = repo_root / ".wp1coverage.json"
    export = subprocess.run(
        [str(python_exe), "-m", "coverage", "json", f"--rcfile={rcfile}",
         "-o", str(json_out)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    if export.returncode != 0 or not json_out.exists():
        return None, f"coverage json export failed: {export.stderr.strip()[-200:]}"
    try:
        return json.loads(json_out.read_text()), ""
    except json.JSONDecodeError as e:
        return None, f"coverage json was unparseable: {e}"


def ochiai_from_coverage(
    coverage_json: dict,
    structure_map: Dict[str, dict],
    repo_root: Path,
    failing_test_ids: Sequence[str],
    limit: int = 20,
) -> RankedList:
    """Standard Ochiai over a (method x test) spectrum.

        susp(m) = failed(m) / sqrt(total_failed * (failed(m) + passed(m)))

    A method executed only by failing tests scores 1.0; one executed by
    everything scores near zero. Methods no failing test touched are dropped
    rather than ranked last, since they cannot be responsible for a failure
    they never participated in.
    """
    ranges = method_line_ranges(source_only(structure_map))
    failing_markers = [t.split("::")[-1].split(".")[-1].lower() for t in failing_test_ids]

    def context_is_failing(context: str) -> bool:
        c = context.lower()
        return any(m and m in c for m in failing_markers)

    executed_failing: Counter = Counter()
    executed_passing: Counter = Counter()
    failing_contexts, passing_contexts = set(), set()

    for file_path, file_data in (coverage_json.get("files") or {}).items():
        rel = _relativize(file_path, repo_root)
        file_ranges = ranges.get(rel)
        if not file_ranges:
            continue
        contexts_by_line = file_data.get("contexts") or {}
        for line_str, contexts in contexts_by_line.items():
            try:
                line = int(line_str)
            except ValueError:
                continue
            key = _method_for_line(file_ranges, line)
            if not key:
                continue
            for context in contexts:
                if not context:
                    continue
                if context_is_failing(context):
                    failing_contexts.add(context)
                    executed_failing[key] += 1
                else:
                    passing_contexts.add(context)
                    executed_passing[key] += 1

    total_failed = len(failing_contexts)
    if total_failed == 0:
        return RankedList.unavailable(
            "Ochiai",
            "no failing-test context appeared in the coverage data — the trigger "
            "tests did not execute under coverage",
        )

    scored: List[Tuple[str, float]] = []
    for key, n_failed in executed_failing.items():
        n_passed = executed_passing.get(key, 0)
        denom = math.sqrt(total_failed * (n_failed + n_passed))
        if denom > 0:
            scored.append((key, n_failed / denom))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))

    return RankedList(
        name="Ochiai",
        entries=[k for k, _ in scored[:limit]],
        detail={
            "failing_contexts": total_failed,
            "passing_contexts": len(passing_contexts),
            "methods_covered_by_failing_tests": len(executed_failing),
        },
    )


# ---------------------------------------------------------------------------
# BoostN — IRFL via BM25
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Identifier-aware: camelCase and snake_case are split, so a report
    saying "separability matrix" can match `_separable_matrix`."""
    out: List[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        for part in re.split(r"_+", raw):
            if not part:
                continue
            for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", part):
                low = piece.lower()
                if len(low) > 2 and low not in STOPWORDS:
                    out.append(low)
    return out


def _method_document(key: str, meta: dict, repo_root: Path, body_lines: int = 40) -> str:
    parts = [key.replace("::", " ").replace("/", " ").replace(".", " ")]
    line = meta.get("line")
    file = meta.get("file")
    if file and isinstance(line, int):
        path = Path(repo_root) / file
        try:
            src = path.read_text(errors="replace").splitlines()
            parts.append("\n".join(src[line - 1 : line - 1 + body_lines]))
        except OSError:
            pass
    return "\n".join(parts)


def boostn_irfl(
    problem_statement: str,
    structure_map: Dict[str, dict],
    repo_root: Path,
    limit: int = 20,
    k1: float = 1.2,
    b: float = 0.75,
) -> RankedList:
    """BM25 over method-level documents. Stand-in for BoostNSift: the
    retrieval half is faithful, the sifting half is not reproduced."""
    query = _tokenize(problem_statement)
    if not query:
        return RankedList.unavailable(
            "BoostN", "bug report produced no usable query terms after tokenization"
        )

    docs: Dict[str, Counter] = {}
    lengths: Dict[str, int] = {}
    df: Counter = Counter()
    for key, meta in source_only(structure_map).items():
        tokens = _tokenize(_method_document(key, meta, repo_root))
        if not tokens:
            continue
        counts = Counter(tokens)
        docs[key] = counts
        lengths[key] = len(tokens)
        df.update(counts.keys())

    if not docs:
        return RankedList.unavailable("BoostN", "structure map yielded no readable method bodies")

    n_docs = len(docs)
    avg_len = sum(lengths.values()) / n_docs
    query_counts = Counter(query)

    scored: List[Tuple[str, float]] = []
    for key, counts in docs.items():
        length = lengths[key]
        score = 0.0
        for term, qn in query_counts.items():
            tf = counts.get(term)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += qn * idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scored.append((key, score))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))

    return RankedList(
        name="BoostN",
        entries=[k for k, _ in scored[:limit]],
        provisional=True,
        reason="BM25 stand-in for BoostNSift (retrieval reproduced, sifting not)",
        detail={"query_terms": len(query_counts), "scored_methods": len(scored)},
    )


# ---------------------------------------------------------------------------
# SBIR — spectrum + IR fusion
# ---------------------------------------------------------------------------

def sbir_fusion(ochiai: RankedList, boostn: RankedList, limit: int = 20,
                k: int = 60) -> RankedList:
    """Reciprocal-rank fusion of the spectrum and IR rankers.

    RRF rather than score averaging because Ochiai scores live in [0,1] and
    BM25 scores are unbounded; combining them numerically would let BM25
    dominate purely through scale. RRF only uses positions, so neither
    ranker can win on units.
    """
    if not ochiai.available and not boostn.available:
        return RankedList.unavailable(
            "SBIR", "both inputs unavailable — nothing to fuse"
        )
    scores: Dict[str, float] = defaultdict(float)
    for ranked in (ochiai, boostn):
        if not ranked.available:
            continue
        for rank, key in enumerate(ranked.entries, 1):
            scores[key] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return RankedList(
        name="SBIR",
        entries=[key for key, _ in fused[:limit]],
        provisional=True,
        reason="reciprocal-rank fusion of Ochiai and BoostN, standing in for SBIR",
        detail={"fused_from": [r.name for r in (ochiai, boostn) if r.available]},
    )


def compute_all(
    problem_statement: str,
    structure_map: Dict[str, dict],
    repo_root: Path,
    coverage_json: Optional[dict] = None,
    failing_test_ids: Sequence[str] = (),
    coverage_error: str = "",
) -> Dict[str, RankedList]:
    """The three traditional rankers, computed for a SWE-bench instance."""
    if coverage_json:
        ochiai = ochiai_from_coverage(coverage_json, structure_map, repo_root, failing_test_ids)
    else:
        ochiai = RankedList.unavailable(
            "Ochiai", coverage_error or "coverage was not collected for this run"
        )
    boostn = boostn_irfl(problem_statement, structure_map, repo_root)
    return {"SBIR": sbir_fusion(ochiai, boostn), "Ochiai": ochiai, "BoostN": boostn}

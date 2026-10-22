"""Real-source-shaped SWE-bench runner tests. Fixtures are not benchmark results."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from rave_fl.core import Action
from rave_fl.pilot_backend import build_prompt
from rave_fl.swebench_pilot import (python_sketch, index_python_functions,
                                   prepare_swebench, gold_from_patch, run)


def _workspace(tmp_path):
    work = tmp_path / "sympy__sympy-20590"
    repo = work / "repo"
    pkg = repo / "sympy/core"
    pkg.mkdir(parents=True)
    (pkg / "symbol.py").write_text(
        "class Symbol:\n"
        "    __slots__ = ('name',)\n"
        "    def __new__(cls, value):\n"
        "        if value == 3600000:\n"
        "            return value\n"
        "        raise ValueError(value)\n"
        "\n"
        "def helper(n):\n"
        "    return n + 42\n"
    )
    (work / "metadata.json").write_text(json.dumps({
        "instance_id": "sympy__sympy-20590",
        "problem_statement": "Symbol has a __dict__ when it should use __slots__, value 3600000",
        "FAIL_TO_PASS": ["sympy/core/tests/test_symbol.py::test_slots"],
        "patch": "diff --git a/sympy/core/symbol.py b/sympy/core/symbol.py\n"
                 "--- a/sympy/core/symbol.py\n"
                 "+++ b/sympy/core/symbol.py\n"
                 "@@ -4,1 +4,1 @@\n"
                 "-        if value == 3600000:\n"
                 "+        if value == 3600001:\n",
    }))
    (work / "outputs").mkdir()
    (work / "outputs/raw_test_output.txt").write_text(
        "FAILED test_slots Symbol __dict__ unexpectedly exists value=3600000"
    )
    return work


def test_python_ast_sketch_keeps_actual_full_code_and_condition():
    source = "def example(n):\n    if n == 3600000:\n        raise ValueError(n)\n    return n + 42\n"
    levels, meta = python_sketch("pkg/a.py::example", source)
    assert levels[2] == source
    assert "3600000" in levels[1]
    assert "ValueError" in levels[1]
    assert "def example" in levels[0]
    assert meta["parser"] == "python-ast"
    assert not meta["parse_error"]


def test_python_syntax_error_is_not_silently_accepted():
    levels, meta = python_sketch("x.py::bad", "def bad(:\n")
    assert meta["parse_error"]
    assert levels[2] == "def bad(:\n"


def test_real_python_index_and_gold_free_preparation(tmp_path):
    work = _workspace(tmp_path)
    items = index_python_functions(work / "repo", "sympy")
    assert "sympy/core/symbol.py::Symbol.__new__" in items
    assert "sympy/core/symbol.py::helper" in items
    candidates, context, indexed, provenance = prepare_swebench(work, top_k=2)
    assert provenance["gold_used_in_prompt"] is False
    assert provenance["candidate_count"] == 2
    assert candidates[0].sketch_metadata["parser"] == "python-ast"
    assert "return value" in candidates[0].source
    assert "patch" not in provenance["metadata_keys_read_before_inference"]
    truths, label_type = gold_from_patch(work / "metadata.json", indexed)
    assert "sympy/core/symbol.py::Symbol.__new__" in truths
    assert label_type == "patch_hunk_intersection_proxy"


def test_no_real_test_log_fails_loudly(tmp_path):
    work = _workspace(tmp_path)
    (work / "outputs/raw_test_output.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing observed test log"):
        prepare_swebench(work)


def test_python_verifier_prompt_and_java_unchanged():
    from rave_fl.core import Candidate
    from rave_fl.pilot_types import PilotCandidate
    c = PilotCandidate("sympy/core/symbol.py::Symbol.__new__", "def __new__(cls): pass",
                       .5, 1, ("x", "y", "def __new__(cls): pass"), {})
    en, _ = build_prompt([c], "failed", Action.FULL_VERIFY, 2, language="Java")
    py, _ = build_prompt([c], "failed", Action.FULL_VERIFY, 2, language="Python")
    assert "Java" in en and "Python" not in en
    assert "Python" in py and "Java" not in py


def test_swebench_dry_run_is_gold_free_and_uses_no_llm(tmp_path):
    work = _workspace(tmp_path)
    args = Namespace(work=work, source_package="sympy", top_k=2, evidence="raw",
                     arms="full,fixed_batch,repair_grounded", model="nonexistent",
                     base_url="http://127.0.0.1:1", max_output_tokens=160,
                     timeout=10, context_window=32768, dry_run=True,
                     full_budget=60000, policy_budget=16000, batch_size=2,
                     max_full_repairs=2, output=tmp_path / "results")
    result = run(args)
    assert result["status"] == "preflight_only"
    assert result["gold_read"] is False
    assert all(a["llm_called"] is False for a in result["arms"].values())
    assert (args.output / "input.json").exists()
    assert (args.output / "run.json").exists()
    with pytest.raises(ValueError, match="not empty"):
        run(args)

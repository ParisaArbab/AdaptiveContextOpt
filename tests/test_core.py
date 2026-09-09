import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp1"))

from evaluation import evaluate_top5, method_matches
from flexfl_pipeline import parse_top5
from leanctx_compressor import _parse_token_header, _reconstruct_from_preview
from agent_localizer import _symptom_vertices_from_trace, postprocess_topk
from metrics import score_instance
from traditional_fl import ochiai_from_coverage


def test_method_matching_ignores_qualified_parameter_type():
    assert method_matches(
        "org.joda.time.DateTime.foo(java.lang.String)",
        "org.joda.time.DateTime.foo(String)",
    )


def test_top5_parser():
    text = "Top_1 : a.A.foo()\nTop_2 : a.A.bar()\n"
    assert parse_top5(text) == ["a.A.foo()", "a.A.bar()"]


def test_leanctx_preview_reconstruction():
    original = "a\nb\nc\nd\n"
    report = """compress preview — pipeline: shell
tokens: 4 -> 3  (-1, 25.0% saved)
bytes:  8 -> 6
-- diff (original -> compressed) --
-2: b
+2: X

diff +1/-1 lines
"""
    assert _reconstruct_from_preview(original, report) == "a\nX\nc\nd"
    assert _parse_token_header(report) == (4, 3, 25.0)


def test_evaluation_rank():
    metrics = evaluate_top5(["a.A.x()", "a.A.foo()"], ["a.A.foo()"])
    assert metrics["top3"] is True
    assert metrics["first_relevant_rank"] == 2
    assert metrics["reciprocal_rank"] == 0.5


def test_file_score_requires_the_same_path():
    scored = score_instance(
        ["pkg/other.py::wrong"], ["pkg/other.py"],
        ["pkg/bug.py::bad"], ["pkg/bug.py"],
    )
    assert scored["file_level"]["top1"] == 0.0


def test_trace_localization_is_scoped_to_file_and_method():
    structure = {
        "pkg/bug.py::bad()": {"file": "pkg/bug.py", "line": 10, "end_line": 20},
        "pkg/other.py::bad()": {"file": "pkg/other.py", "line": 10, "end_line": 20},
    }
    trace = '  File "/checkout/pkg/bug.py", line 12, in bad\n'
    assert _symptom_vertices_from_trace(trace, structure) == ["pkg/bug.py::bad()"]
    assert postprocess_topk(["pkg/bug.py::bad"], structure) == ["pkg/bug.py::bad()"]


def test_ochiai_counts_distinct_tests_not_covered_lines():
    coverage = {"files": {"/repo/pkg/bug.py": {"contexts": {
        "10": ["test_fail"], "11": ["test_fail"],
        "30": ["test_fail", "test_pass"],
    }}}}
    structure = {
        "pkg/bug.py::bad()": {"file": "pkg/bug.py", "line": 10, "end_line": 20},
        "pkg/bug.py::other()": {"file": "pkg/bug.py", "line": 30, "end_line": 40},
    }
    ranked = ochiai_from_coverage(coverage, structure, Path("/repo"), ["test_fail"])
    assert ranked.available and ranked.entries[0] == "pkg/bug.py::bad()"
    assert ranked.detail['scores']['pkg/bug.py::bad()'] == 1.0


def test_missing_nested_helper_is_indexed_and_trace_resolves_ambiguous_name(tmp_path):
    from graphify_structure import _python_source_scopes
    (tmp_path / 'code.py').write_text('def first():\n    def _f():\n        return 1\n    return _f()\n'
                                     'def second():\n    def _f():\n        return 2\n    return _f()\n')
    structure = _python_source_scopes({
        'code.py::first()': {'file': 'code.py', 'line': 1, 'id': 'original'},
        'code.py::second()': {'file': 'code.py', 'line': 5, 'id': 'second'},
    }, tmp_path)
    assert structure['code.py::first._f()']['end_line'] == 3
    assert structure['code.py::first()']['id'] == 'original'
    trace = 'File "/repo/code.py", line 7, in _f'
    assert postprocess_topk(['code.py::_f'], structure, trace) == ['code.py::second._f()']
    assert postprocess_topk(['code.py:first'], structure) == ['code.py::first()']


def test_ir_document_does_not_include_the_next_method(tmp_path):
    from traditional_fl import _method_document
    (tmp_path / 'code.py').write_text('def first():\n    return 1\ndef unrelated():\n    return "distinctive"\n')
    document = _method_document('code.py::first()',
                                {'file': 'code.py', 'line': 1, 'end_line': 2}, tmp_path)
    assert 'return 1' in document
    assert 'unrelated' not in document and 'distinctive' not in document


def test_exact_method_names_precede_substring_noise():
    from agent_localizer import StructureQueryTools
    structure = {f'noise.py::unrelated_function_{i}()': {'file': 'noise.py'} for i in range(100)}
    structure['code.py::outer._f()'] = {'file': 'code.py'}
    assert StructureQueryTools(structure).find_method('_f') == ['code.py::outer._f()']


def test_nested_tool_calls_are_rejected_but_quoted_method_keys_are_valid():
    from agent_localizer import _parse_function_call
    assert _parse_function_call("get_methods_of_class(find_class('simplify'))") == (None, '')
    assert _parse_function_call("get_code_snippet_of_method('code.py::outer._f()')") == (
        'get_code_snippet_of_method', 'code.py::outer._f()')
    assert _parse_function_call('get_code_snippet_of_method(3)') == ('get_code_snippet_of_method', '3')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp1"))

from evaluation import evaluate_top5, method_matches
from flexfl_pipeline import parse_top5
from leanctx_compressor import _parse_token_header, _reconstruct_from_preview


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

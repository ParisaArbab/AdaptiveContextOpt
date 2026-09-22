from wp1.run_swebench_agent4sr_pair import finalize_top5_format, run_agent


class FakeBackend:
    def complete(self, system, user):
        assert "gold" in system.lower()
        assert "sympy/core/_print_helpers.py::Printable" in user
        return """Top_1 : sympy/core/expr.py::Expr
Top_2 : sympy/core/basic.py::Basic
Top_3 : sympy/core/evalf.py::EvalfMixin
Top_4 : sympy/core/symbol.py::Symbol
Top_5 : sympy/core/_print_helpers.py::Printable"""


def test_format_only_finalization_uses_seen_entities():
    final_response = """Top_1 : sympy/core/expr.py::Expr
Top_2 : sympy/core/basic.py::Basic
Top_3 : sympy/core/evalf.py::EvalfMixin
Top_4 : sympy/core/symbol.py::Symbol"""

    history = [
        {
            "assistant": 'get_code_snippet("sympy/core/_print_helpers.py::Printable")',
            "tool": "sympy/core/_print_helpers.py::Printable",
        }
    ]

    response, predictions = finalize_top5_format(
        FakeBackend(),
        final_response,
        history,
    )

    assert len(predictions) == 5
    assert predictions[-1] == "sympy/core/_print_helpers.py::Printable"
    assert "Top_5" in response



class StateBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        assert "PREVIOUS LOCALIZATION STATE (NOT GROUND TRUTH)" in user
        assert "sympy/core/_print_helpers.py::Printable" in user
        if self.calls == 1:
            return 'find_path("sympy/core/basic.py")'
        if self.calls == 2:
            return 'get_code_snippet("sympy/core/basic.py::Basic")'
        return """Top_1 : sympy/core/_print_helpers.py::Printable
Top_2 : sympy/core/basic.py::Basic
Top_3 : sympy/core/symbol.py::Symbol
Top_4 : sympy/core/expr.py::AtomicExpr
Top_5 : sympy/core/basic.py::Atom"""


class StateGraph:
    nodes = []

    def find_paths(self, query, limit):
        return ["sympy/core/basic.py"]

    def snippet(self, ref):
        return "class Basic(Printable): pass"


def test_run_agent_carries_previous_ranking_into_prompt():
    previous = [
        "sympy/core/_print_helpers.py::Printable",
        "sympy/core/basic.py::Basic",
    ]
    result = run_agent(
        StateBackend(),
        StateGraph(),
        "problem",
        "test_immutable",
        "runtime",
        max_steps=3,
        previous_predictions=previous,
    )

    assert result["predictions"][0] == "sympy/core/_print_helpers.py::Printable"
    assert result["previous_predictions_supplied"] == previous

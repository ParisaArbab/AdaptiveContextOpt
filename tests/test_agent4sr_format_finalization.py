from wp1.run_swebench_agent4sr_pair import finalize_top5_format


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

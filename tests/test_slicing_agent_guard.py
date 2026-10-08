from wp1.run_swebench_agent4sr_pair import _previously_seen_entities, run_agent


class _FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, prompt):
        self.prompts.append((system, prompt))
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)


class _FakeGraph:
    nodes = []

    def snippet(self, ref):
        return f"SOURCE FOR {ref}"

    def find_paths(self, query, limit=30):
        return []

    def find_methods(self, query, limit=40):
        return []


def test_slice_regression_guard_requests_reconsideration():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            (
                "Top_1 : sympy/core/basic.py::Basic.__eq__\n"
                "Top_2 : sympy/core/basic.py::Basic.compare\n"
                "Top_3 : sympy/core/basic.py::Basic.__hash__\n"
                "Top_4 : sympy/core/basic.py::Basic._hashable_content\n"
                "Top_5 : sympy/core/basic.py::Basic.sort_key"
            ),
            (
                f"Top_1 : {printable}\n"
                f"Top_2 : {basic}\n"
                "Top_3 : sympy/core/basic.py::Atom\n"
                "Top_4 : sympy/core/expr.py::AtomicExpr\n"
                "Top_5 : sympy/core/symbol.py::Symbol"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=6,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
    )

    assert result["regression_guard_used"] is True
    assert printable in result["predictions"]
    assert set(result["inspected_persistent_hypotheses"]) == {basic, printable}


def test_slice_hypotheses_are_added_to_prompt():
    printable = "sympy/core/_print_helpers.py::Printable"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            'get_code_snippet("sympy/core/basic.py::Basic")',
            (
                f"Top_1 : {printable}\n"
                "Top_2 : sympy/core/basic.py::Basic\n"
                "Top_3 : sympy/core/basic.py::Atom\n"
                "Top_4 : sympy/core/expr.py::AtomicExpr\n"
                "Top_5 : sympy/core/symbol.py::Symbol"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="failure",
        failing_test="test_immutable",
        runtime_output="assertion",
        max_steps=5,
        persistent_hypotheses=[printable],
        structural_focus="focus on object layout",
    )

    first_prompt = backend.prompts[0][1]
    assert "DEPENDENCY-DERIVED PERSISTENT HYPOTHESES" in first_prompt
    assert printable in first_prompt
    assert "focus on object layout" in first_prompt
    assert result["regression_guard_used"] is False



def test_slice_agent_can_finalize_early_after_hypotheses_inspected():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            (
                f"Top_1 : {printable}\n"
                f"Top_2 : {basic}\n"
                "Top_3 : sympy/core/basic.py::Atom\n"
                "Top_4 : sympy/core/expr.py::AtomicExpr\n"
                "Top_5 : sympy/core/symbol.py::Symbol"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=20,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
    )

    assert result["steps"] == 3
    assert result["regression_guard_used"] is False
    assert result["predictions"][0] == printable


def test_final_step_regression_guard_can_reconsider():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            (
                "Top_1 : sympy/core/function.py::Derivative._eval_derivative\n"
                "Top_2 : sympy/core/function.py::Derivative.doit\n"
                "Top_3 : sympy/core/function.py::Derivative.__new__\n"
                "Top_4 : sympy/core/function.py::_derivative_dispatch\n"
                "Top_5 : sympy/core/basic.py::Basic.diff"
            ),
            (
                f"Top_1 : {printable}\n"
                f"Top_2 : {basic}\n"
                "Top_3 : sympy/core/basic.py::Atom\n"
                "Top_4 : sympy/core/expr.py::AtomicExpr\n"
                "Top_5 : sympy/core/symbol.py::Symbol"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=3,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
    )

    assert result["regression_guard_used"] is True
    assert result["predictions"][0] == printable



def test_post_slice_tool_budget_forces_evidence_only_finalizer():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            'get_code_snippet("sympy/core/basic.py::Basic.__slots__")',
            (
                f"Top_1 : {printable}\n"
                f"Top_2 : {basic}\n"
                "Top_3 : sympy/core/basic.py::Atom\n"
                "Top_4 : sympy/core/expr.py::AtomicExpr\n"
                "Top_5 : sympy/core/symbol.py::Symbol"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=20,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
        post_slice_extra_tool_budget=1,
    )

    assert result["tool_calls"] == 3
    assert result["evidence_only_finalizer_used"] is True
    assert result["post_slice_extra_tool_budget"] == 1
    assert result["predictions"][0] == printable



def test_evidence_only_finalizer_rejects_duplicate_top5():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            'get_code_snippet("sympy/core/basic.py::Basic.__slots__")',
            (
                f"Top_1 : {basic}\n"
                f"Top_2 : {printable}\n"
                f"Top_3 : {basic}\n"
                f"Top_4 : {printable}\n"
                f"Top_5 : {basic}"
            ),
            (
                f"Top_1 : {printable}\n"
                f"Top_2 : {basic}\n"
                "Top_3 : sympy/core/basic.py::Basic.__slots__\n"
                "Top_4 : sympy/core/basic.py::<module>\n"
                "Top_5 : sympy/core/_print_helpers.py::<module>"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=20,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
        post_slice_extra_tool_budget=1,
    )

    assert result["evidence_only_finalizer_used"] is True
    assert len(result["predictions"]) == 5
    assert len(set(result["predictions"])) == 5



def test_bounded_finalization_never_resumes_tools_after_invalid_llm_top5():
    printable = "sympy/core/_print_helpers.py::Printable"
    basic = "sympy/core/basic.py::Basic"

    backend = _FakeBackend(
        [
            f'get_code_snippet("{printable}")',
            f'get_code_snippet("{basic}")',
            'get_code_snippet("sympy/core/basic.py::Basic.__slots__")',
            (
                f"Top_1 : {basic}\n"
                f"Top_2 : {printable}\n"
                f"Top_3 : {basic}\n"
                f"Top_4 : {printable}\n"
                f"Top_5 : {basic}"
            ),
        ]
    )

    result = run_agent(
        backend,
        _FakeGraph(),
        problem="object unexpectedly has __dict__",
        failing_test="test_immutable",
        runtime_output="assert not hasattr(b1, '__dict__')",
        max_steps=20,
        persistent_hypotheses=[basic, printable],
        structural_focus="prioritize __slots__ and inheritance",
        post_slice_extra_tool_budget=1,
    )

    assert result["tool_calls"] == 3
    assert result["evidence_only_finalizer_used"] is True
    assert len(result["predictions"]) == 5
    assert len(set(result["predictions"])) == 5
    assert len(backend.responses) == 0



def test_seen_entity_regex_captures_method_and_module_candidates():
    history = [
        {
            "assistant": 'get_code_snippet("sympy/core/basic.py::Basic.__slots__")',
            "tool": "sympy/core/basic.py::Basic.__slots__\nsource",
        },
        {
            "assistant": 'get_code_snippet("sympy/core/_print_helpers.py::Printable")',
            "tool": "sympy/core/_print_helpers.py::Printable\nsource",
        },
    ]

    seen = _previously_seen_entities("", history)

    assert "sympy/core/basic.py::Basic.__slots__" in seen
    assert "sympy/core/_print_helpers.py::Printable" in seen

from pathlib import Path

from wp1.adaptive_feedback import (
    next_density,
    normalize_density_schedule,
    parse_feedback_decision,
    retrieve_targeted_evidence,
)


def test_density_schedule_is_sorted_and_unique():
    assert normalize_density_schedule([0.7, 0.3, 0.5, 0.3, 1.0]) == [
        0.3,
        0.5,
        0.7,
        1.0,
    ]


def test_next_density_moves_toward_more_context():
    schedule = [0.3, 0.5, 0.7, 1.0]
    assert next_density(0.3, schedule) == 0.5
    assert next_density(0.5, schedule) == 0.7
    assert next_density(0.7, schedule) == 1.0
    assert next_density(1.0, schedule) is None


def test_feedback_parser_stop_json():
    decision = parse_feedback_decision(
        '{"decision":"STOP","reason":"evidence is sufficient","missing_evidence":[]}'
    )
    assert decision.decision == "STOP"
    assert decision.expand is False
    assert decision.targeted is False


def test_feedback_parser_targeted_json():
    decision = parse_feedback_decision(
        """{
          "decision": "TARGETED_EXPAND",
          "reason": "Need the parent definition.",
          "missing_evidence": [
            {
              "evidence_type": "inheritance_chain",
              "anchor_entity": "sympy/core/basic.py::Basic",
              "question": "Which direct parent can introduce __dict__?",
              "why_needed": "The failure is consistent with an unslotted parent."
            }
          ]
        }"""
    )
    assert decision.decision == "TARGETED_EXPAND"
    assert decision.targeted is True
    assert decision.missing_evidence[0]["anchor_entity"] == (
        "sympy/core/basic.py::Basic"
    )


def test_vague_targeted_feedback_falls_back_to_density():
    decision = parse_feedback_decision(
        '{"decision":"TARGETED_EXPAND","reason":"need more","missing_evidence":[]}'
    )
    assert decision.decision == "EXPAND_DENSITY"
    assert decision.source == "validation_fallback"


def test_legacy_expand_is_supported():
    decision = parse_feedback_decision(
        "Decision: EXPAND\nReason: more runtime evidence is needed"
    )
    assert decision.decision == "EXPAND_DENSITY"
    assert decision.expand is True


def test_unparseable_feedback_expands_safely():
    decision = parse_feedback_decision("maybe")
    assert decision.decision == "EXPAND_DENSITY"
    assert decision.source == "parse_fallback"


class _Node:
    def __init__(self, label, source_file, callable_class=None):
        self.label = label
        self.source_file = source_file
        self.callable_class = callable_class


class _FakeGraph:
    def __init__(self, repo):
        self.repo = Path(repo)
        self.nodes = [
            _Node("Basic", "sympy/core/basic.py"),
            _Node("Printable", "sympy/core/_print_helpers.py"),
        ]

    def snippet(self, ref):
        path, entity = ref.split("::", 1)
        source = (self.repo / path).read_text()
        return f"{ref}\n{source}"


def test_inheritance_targeted_retrieval_inspects_direct_parent(tmp_path):
    basic = tmp_path / "sympy/core/basic.py"
    printable = tmp_path / "sympy/core/_print_helpers.py"
    basic.parent.mkdir(parents=True)
    basic.write_text(
        "from ._print_helpers import Printable\n"
        "class Basic(Printable):\n"
        "    __slots__ = ('x',)\n"
    )
    printable.write_text(
        "class Printable:\n"
        "    def __str__(self):\n"
        "        return 'x'\n"
    )

    graph = _FakeGraph(tmp_path)
    records = retrieve_targeted_evidence(
        graph,
        [
            {
                "evidence_type": "inheritance_chain",
                "anchor_entity": "sympy/core/basic.py::Basic",
                "question": "Which parent class may introduce __dict__?",
                "why_needed": "Need to verify the inheritance hypothesis.",
            }
        ],
        raw_runtime_output="",
    )

    assert len(records) == 1
    content = records[0]["content"]
    assert "Basic(Printable)" in content
    assert "DIRECT PARENT Printable" in content
    assert "class Printable" in content


def test_runtime_targeted_retrieval_extracts_local_failure_block(tmp_path):
    graph = _FakeGraph(tmp_path)
    records = retrieve_targeted_evidence(
        graph,
        [
            {
                "evidence_type": "runtime_detail",
                "anchor_entity": "test_immutable",
                "question": "What assertion failed?",
                "why_needed": "Need the exact runtime symptom.",
            }
        ],
        raw_runtime_output=(
            "test_structure ok\n"
            "test_immutable F\n"
            "Traceback\n"
            "File test_basic.py, line 38, in test_immutable\n"
            "assert not hasattr(b1, '__dict__')\n"
            "AssertionError\n"
        ),
    )

    assert len(records) == 1
    assert "assert not hasattr" in records[0]["content"]

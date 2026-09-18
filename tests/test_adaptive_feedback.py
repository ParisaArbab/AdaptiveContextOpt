from wp1.adaptive_feedback import (
    next_density,
    normalize_density_schedule,
    parse_feedback_decision,
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


def test_feedback_parser():
    stop = parse_feedback_decision("Decision: STOP\nReason: evidence is sufficient")
    assert stop.decision == "STOP"
    assert stop.expand is False

    expand = parse_feedback_decision(
        "Decision: EXPAND\nReason: more runtime evidence is needed"
    )
    assert expand.decision == "EXPAND"
    assert expand.expand is True


def test_unparseable_feedback_expands_safely():
    decision = parse_feedback_decision("maybe")
    assert decision.decision == "EXPAND"
    assert decision.source == "parse_fallback"

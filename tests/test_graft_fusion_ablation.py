"""Tests for the opt-in GRAFT fusion treatment, no changes to old baselines."""
from rave_fl.eval.fusion import Fusion
from rave_fl.pilot_types import CandidateState, Observation, PilotCandidate


def example(rank, suspicion=None, valid=True):
    candidate = PilotCandidate(
        function_id=f"example.F.method{rank}()",
        source="return 1;",
        prior=0.5,
        rank=rank,
        levels=("one", "two", "three"),
        features={},
    )
    observation = (
        None if suspicion is None else Observation(
            function_id=candidate.function_id,
            suspicion=suspicion,
            uncertainty=0.1,
            reason="fixture evidence",
            valid=valid,
        )
    )
    return candidate, CandidateState(level=1 if observation else -1, observation=observation)


def test_fixed_suspicion_blend_can_override_initial_rank():
    examples = [example(1, 0.1), example(2, 0.3), example(3, 0.9)]
    candidates = [a for a, _ in examples]
    states = {a.function_id: b for a, b in examples}
    f = Fusion("suspicion_rank_blend")
    scores = f.scores(candidates, states)
    ranked = f.order(candidates, scores)
    assert [c.rank for c in ranked] == [3, 1, 2]
    assert scores[candidates[2].function_id] == 0.70 * 0.9
    assert f.config()["calibrated"] is False


def test_invalid_and_missing_observations_use_original_rank():
    examples = [example(1, 1.0, valid=False), example(2, None), example(3, 0.5)]
    candidates = [a for a, _ in examples]
    states = {a.function_id: b for a, b in examples}
    score = Fusion("suspicion_rank_blend").scores(candidates, states)
    assert score[candidates[0].function_id] == 1.0
    assert score[candidates[1].function_id] == 0.5
    assert score[candidates[2].function_id] == 0.35


def test_all_original_fusions_are_unchanged():
    examples = [example(1, 0.1), example(2, 0.9)]
    candidates = [a for a, _ in examples]
    states = {a.function_id: b for a, b in examples}
    rrf = Fusion("rrf", rrf_k=60).scores(candidates, states)
    assert abs(rrf[candidates[0].function_id] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(rrf[candidates[1].function_id] - (1 / 62 + 1 / 61)) < 1e-12
    assert Fusion("consistent_score").scores(candidates, states) == {
        candidates[0].function_id: 0.1,
        candidates[1].function_id: 0.9,
    }
    assert Fusion("prior_aware").scores(candidates, states)


def test_blend_deterministic_when_scores_are_equal():
    examples = [example(1, None), example(2, None)]
    candidates = [a for a, _ in examples]
    states = {a.function_id: b for a, b in examples}
    f = Fusion("suspicion_rank_blend", tie="original")
    assert [c.rank for c in f.order(candidates, f.scores(candidates, states))] == [1, 2]


def test_offline_audit_uses_frozen_responses_and_evaluation_only_labels(tmp_path):
    from dataclasses import asdict
    import json
    from scripts.audit_graft_fusion_offline import inspect_instance, main

    folder = tmp_path / "run" / "instances" / "Time-4_fixture"
    folder.mkdir(parents=True)
    examples = [example(1, 0.1), example(2, 0.9)]
    candidates = [c for c, _ in examples]
    states = {c.function_id: state for c, state in examples}
    historical = Fusion("rrf")
    old = [c.function_id for c in historical.order(
        candidates, historical.scores(candidates, states)
    )]
    (folder / "input.json").write_text(json.dumps({
        "candidates": [asdict(c) for c in candidates],
    }))
    (folder / "l1_anchor__rrf_fixture.json").write_text(json.dumps({
        "instance_id": "Time-4",
        "status": "completed",
        "arm": {
            "states": {key: asdict(value) for key, value in states.items()},
            "ranker": historical.config(),
            "ranked_functions": old,
            "failure_name_anchors": {},
            "graft_config": {"failure_name_anchor": False},
        },
    }))
    (folder / "evaluation_only.json").write_text(json.dumps({
        "functions": [candidates[1].function_id],
    }))
    audit = inspect_instance(folder)
    assert audit["rrf_reproduced"]
    assert audit["rankings"]["suspicion_rank_blend"]["first_relevant_rank"] == 1
    path = tmp_path / "report.json"
    assert main([str(tmp_path / "run"), "--output", str(path)]) == 0
    result = json.loads(path.read_text())
    assert result["new_model_calls"] == 0
    assert result["gold_used_for_scoring"] is False

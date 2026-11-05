from copy import deepcopy
from dataclasses import replace
import json
import pytest
from test_graft_graph import fixture_graph
from rave_fl.eval.backend import Provider, Store, ExecutionLimit, UnknownUsage
from rave_fl.eval.cli import DEFAULTS
from rave_fl.eval.fusion import Fusion
from rave_fl.eval.representations import prepare_entities
from rave_fl.eval.schema import CompatibilityError
from rave_fl.graft.controller import (
    GraftConfig,
    Session,
    Router,
    EvidenceStage,
    select_evidence,
    GraftProvider,
)


def setup(tmp_path, config=None, budget=16000, provider_class=GraftProvider):
    instance, entities, graph = fixture_graph()
    candidates, _, _ = prepare_entities(instance, entities, "l1", 2200)
    opts = {
        **deepcopy(DEFAULTS),
        "backend": "fake",
        "admission": "strict",
        "context_window": 4096,
        "batch_size": 2,
        "output_tokens": 160,
        "physical_budget": 80000,
        "per_bug_physical_budget": 80000,
    }
    provider = provider_class(opts)
    provider.preflight()
    store = Store(tmp_path, opts)
    session = Session(
        instance, entities, provider, store, opts, budget, graph, config or GraftConfig()
    )
    return candidates, session, store


def test_distinct_stages_and_resume_reuse_without_new_paid_calls(tmp_path):
    candidates, session, store = setup(tmp_path)
    first = Router(candidates, session, Fusion("rrf")).execute()
    assert {"sketch", "graph", "method_full"} <= {c["stage"] for c in first["calls"]}
    assert not first["budget_exceeded"] and first["actual_tokens"] == sum(
        e["actual_tokens"] for e in session.events
    )
    assert first["actual_tokens"] == sum(
        c["input_tokens"] + c["output_tokens"] for c in first["calls"]
    )
    assert not first["graph_is_method_full_superset"]
    old = store.snapshot()
    second_session = Session(
        session.legacy.instance,
        session.legacy.entities,
        session.legacy.provider,
        store,
        session.legacy.opts,
        16000,
        session.graph,
        session.config,
    )
    second = Router(candidates, second_session, Fusion("rrf")).execute()
    assert first["ranked_functions"] == second["ranked_functions"]
    assert first["actual_tokens"] == second["actual_tokens"]
    assert store.snapshot() == old and all(e["reuse"] == "journal" for e in second_session.events)
    assert old["physical_inference_tokens"] == 0 and old["synthetic"]


def test_graph_is_not_projected_as_full_in_historical_ranker(tmp_path):
    candidates, session, _ = setup(tmp_path, GraftConfig(max_graph=1, max_full=0))
    result = Router(candidates, session, Fusion(), "l1_graph").execute()
    assert len([c for c in result["calls"] if c["stage"] == "graph"]) == 1
    assert all(s["level"] == 1 for s in result["states"].values())
    assert all(c["stage"] != "method_full" for c in result["calls"])


def test_hard_logical_and_physical_admission(tmp_path):
    candidates, session, store = setup(tmp_path, budget=0)
    result = Router(candidates, session, Fusion()).execute()
    assert result["actual_tokens"] == 0 and store.snapshot()["new_calls"] == 0
    session.budget = session.legacy.budget = session.graph_backend.budget = 10000
    store.opts["max_new_calls"] = 0
    limited = Router(candidates, session, Fusion()).execute()
    assert limited["stop_reason"].startswith("execution_limit")
    assert not store.snapshot()["new_calls"]


def test_altered_graph_source_or_prompt_cannot_reuse_a_response(tmp_path):
    candidates, session, _ = setup(tmp_path)
    selection = select_evidence(session.graph, candidates[0].function_id, session.config)
    with pytest.raises(CompatibilityError, match="altered"):
        session.planning(
            [candidates[0]], EvidenceStage.GRAPH, replace(selection, text="different prompt")
        )
    session.graph = replace(session.graph, revision="a" * 40)
    with pytest.raises(CompatibilityError, match="revision"):
        session.planning([candidates[0]], EvidenceStage.GRAPH, selection)


class InvalidProvider(Provider):
    def execute(self, payload, ids, action):
        value = super().execute(payload, ids, action)
        value["message"]["content"] = "malformed paid response"
        return value


def test_invalid_paid_responses_remain_charged_and_uncertain(tmp_path):
    candidates, session, _ = setup(tmp_path, provider_class=InvalidProvider)
    result = Router(candidates, session, Fusion()).execute()
    assert result["actual_tokens"] > 0 and all(
        not s["observation"]["valid"] for s in result["states"].values()
    )
    assert all(c["error"] for c in result["calls"])
    assert not result["budget_exceeded"]


class UnknownProvider(Provider):
    def execute(self, payload, ids, action):
        return {"done": True, "message": {"content": "{}"}}


def test_unknown_usage_stops_and_is_not_silently_retried(tmp_path):
    candidates, session, store = setup(tmp_path, provider_class=UnknownProvider)
    with pytest.raises(ValueError, match="counters"):
        Router(candidates, session, Fusion()).execute()
    assert store.snapshot()["usage_unknown"]
    with pytest.raises(UnknownUsage):
        Router(candidates, session, Fusion()).execute()


def test_new_package_fingerprint_and_no_future_outcomes_in_options(tmp_path):
    candidates, session, store = setup(tmp_path, GraftConfig(max_full=0))
    result = Router(candidates, session, Fusion()).execute()
    routes = [t for t in result["trace"] if t["phase"] == "route"]
    assert routes and all("actual_tokens" not in row for t in routes for row in t["options"])
    assert all(
        "gold" not in row and "marginal_mrr" not in row for t in routes for row in t["options"]
    )
    graph_events = [e for e in session.events if e["stage"] == "graph"]
    assert graph_events
    payloads = [json.loads(r[0]) for r in store.db.execute("SELECT request FROM calls")]
    assert len({json.dumps(p["messages"]) for p in payloads}) == len(payloads)


def test_provider_bound_violation_preserves_paid_cost(tmp_path):
    class ViolatingProvider(Provider):
        def execute(self, payload, ids, action):
            result = super().execute(payload, ids, action)
            result["eval_count"] = payload["options"]["num_predict"] + 1
            return result

    candidates, session, store = setup(tmp_path, provider_class=ViolatingProvider)
    result = Router(candidates, session, Fusion()).execute()
    assert result["actual_tokens"] > 0 and store.snapshot()["usage_unknown"]
    assert session.events[-1]["reuse"] == "fresh_bound_violation"
    assert result["actual_tokens"] == store.snapshot()["known_provider_tokens"]


def test_graph_uncertainty_is_required_and_missing_fields_still_cost_tokens(tmp_path):
    # The unmodified legacy fake Full parser omits explicit uncertainty.
    candidates, session, _ = setup(
        tmp_path, GraftConfig(max_graph=1, max_full=0), provider_class=Provider
    )
    result = Router(candidates, session, Fusion(), "l1_graph").execute()
    graph = next(call for call in result["calls"] if call["stage"] == "graph")
    assert graph["input_tokens"] + graph["output_tokens"] > 0
    assert not graph["observations"][0]["valid"] and "uncertainty" in graph["error"]


def test_graph_rejects_non_string_explanations_after_charging(tmp_path):
    class ArrayProvider(GraftProvider):
        def execute(self, payload, ids, action):
            result = super().execute(payload, ids, action)
            from rave_fl.graft.selection import SYSTEM
            if payload['messages'][0]['content'] == SYSTEM:
                value = json.loads(result['message']['content'])
                value['explanation'] = ['wrong schema']
                result['message']['content'] = json.dumps(value)
            return result
    candidates, session, _ = setup(tmp_path, GraftConfig(max_graph=1, max_full=0), provider_class=ArrayProvider)
    result = Router(candidates, session, Fusion(), 'l1_graph').execute()
    call = next(c for c in result['calls'] if c['stage'] == 'graph')
    assert not call['observations'][0]['valid'] and 'strings' in call['error']
    assert call['input_tokens'] + call['output_tokens'] > 0


def test_unsupported_graph_downgrade_keeps_valid_l1_score_and_audit(tmp_path, monkeypatch):
    candidates, session, _ = setup(tmp_path, GraftConfig(max_graph=1, max_full=0))
    candidate = candidates[0]
    router = Router(candidates, session, Fusion("rrf"), "l1_graph")
    router.observe([candidate], EvidenceStage.SKETCH)
    original = router.states[candidate.function_id].observation
    assert original.valid
    # Preserve the same result type as the real backend while simulating an
    # explicitly unsupported lower graph verdict.
    from dataclasses import dataclass, replace as dc_replace

    low = dc_replace(
        original,
        suspicion=max(0.0, original.suspicion - 0.2),
        missing_evidence="No specific code fragment for failure",
    )

    @dataclass
    class StubResult:
        observations: list
        input_tokens: int = 10
        output_tokens: int = 2

    monkeypatch.setattr(session, "execute", lambda *_args: StubResult([low]))
    router.observe([candidate], EvidenceStage.GRAPH)
    state = router.states[candidate.function_id]
    assert state.level == 1
    assert state.observation.suspicion == original.suspicion
    assert state.history[-1]["suspicion"] == low.suspicion
    assert state.history[-1]["applied_to_ranking"] is False
    assert state.history[-1]["guard_reason"] == "unsupported_graph_downgrade"


def test_graph_guard_is_explicitly_ablatable(tmp_path, monkeypatch):
    from dataclasses import dataclass

    config = GraftConfig(
        max_graph=1, max_full=0, guard_unsupported_graph_downgrade=False
    )
    candidates, session, _ = setup(tmp_path, config)
    candidate = candidates[0]
    router = Router(candidates, session, Fusion("rrf"), "l1_graph")
    router.observe([candidate], EvidenceStage.SKETCH)
    original = router.states[candidate.function_id].observation
    low = replace(
        original,
        suspicion=max(0.0, original.suspicion - 0.2),
        missing_evidence="Missing supporting code",
    )

    @dataclass
    class StubResult:
        observations: list
        input_tokens: int = 10
        output_tokens: int = 2

    monkeypatch.setattr(session, "execute", lambda *_args: StubResult([low]))
    router.observe([candidate], EvidenceStage.GRAPH)
    assert router.states[candidate.function_id].observation.suspicion == low.suspicion
    assert router.states[candidate.function_id].history[-1]["applied_to_ranking"] is True


def test_failure_name_anchor_from_observed_test_names_only():
    from rave_fl.graft.controller import failing_test_name_anchors

    evidence = (
        "TEST: org.joda.time.TestDateTimeZoneCutover::test_getOffsetFromLocal_Moscow_Autumn\n"
        "FAILURE: Expected +04:00 but received +03:00\n"
        "PROJECT FRAMES: some irrelevant code\n"
    )
    names = [
        "org.joda.time.DateTimeZone.getOffsetFromLocal(long)",
        "org.joda.time.DateTimeZone.getOffset(long)",
        "org.joda.time.DateTime.toString()",
    ]
    hits = failing_test_name_anchors(evidence, names)
    assert hits == {names[0]: True, names[1]: False, names[2]: False}
    assert not any(failing_test_name_anchors("No TEST header", names).values())


def test_failure_anchor_ablation_changes_only_router_scores(tmp_path):
    from rave_fl.graft.controller import failing_test_name_anchors

    candidates, session, _ = setup(tmp_path, GraftConfig(failure_name_anchor=True))
    router = Router(candidates, session, Fusion("rrf"))
    # Same online evidence in both arms; no gold labels are consulted.
    target = candidates[0].function_id
    router.failure_anchors = {c.function_id: c.function_id == target for c in candidates}
    anchored = router.scores()
    standard = router.fusion.scores(candidates, router.states)
    assert anchored[target] > standard[target]
    assert all(
        anchored[c.function_id] == standard[c.function_id]
        for c in candidates[1:]
    )
    session.config = replace(session.config, failure_name_anchor=False)
    router.config = session.config
    assert router.scores() == standard


def test_failure_anchor_config_rejects_non_boolean():
    with pytest.raises(ValueError, match="Boolean"):
        GraftConfig(failure_name_anchor=1)

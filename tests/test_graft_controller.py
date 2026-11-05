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

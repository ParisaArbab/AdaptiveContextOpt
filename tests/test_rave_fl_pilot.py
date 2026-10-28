"""Behavioral tests: adaptive escalation, accounting, tail recall and leakage."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

from rave_fl.adaptive import AdaptivePilot, fused_score
from rave_fl.code_sketch import FEATURE_NAMES, java_sketch, prepare_candidates
from rave_fl.core import Action, Candidate
from rave_fl.pilot_backend import OllamaPilotBackend, parse_response
from rave_fl.pilot_evaluation import evaluate, risk_rows, summarize
from rave_fl.pilot_types import CallResult, CandidateState, Observation, PolicyConfig
from rave_fl.risk import BootstrapRisk, LearnedRisk
from rave_fl.train_risk import check_split, train


def pool(n=5):
    cs = [Candidate(f"p.F.m{i}()", f"m{i}()", semantic_score=1 - i / n) for i in range(n)]
    sources = {c.function_id: f"public int m{i}() {{ int value = load(); if(value < 0) return -1; return value; }}"
               for i, c in enumerate(cs)}
    metadata = {c.function_id: {"methods": ["SBIR"], "ranks": {"SBIR": i + 1}, "top_per_method": n}
                for i, c in enumerate(cs)}
    return prepare_candidates(cs, sources, metadata, "Observed negative value in loading")


class FeedbackBackend:
    def __init__(self, invalid=False, cost=20, estimate=30):
        self.calls = []
        self.invalid = invalid
        self.cost = cost
        self.planned_cost = estimate

    def estimate(self, candidates, action, level):
        if self.planned_cost == 30 and level == 1:
            return 10
        return self.planned_cost

    def execute(self, candidates, action, level):
        self.calls.append((action, level, [c.function_id for c in candidates]))
        result = []
        for c in candidates:
            tail = c.function_id.endswith("m4()")
            valid = not (self.invalid and level == 0)
            result.append(Observation(c.function_id, .9 if tail else .1,
                                      .8 if level == 0 else .05, "controlled feedback", valid,
                                      "body condition" if level == 0 else ""))
        return CallResult(result, self.cost, 5, .01, "test backend")


class LowRisk(BootstrapRisk):
    def predict(self, f):
        return .05


def test_four_actions_and_low_rank_candidate_can_become_first():
    backend = FeedbackBackend()
    output = AdaptivePilot(pool(), backend, PolicyConfig(token_budget=2000), LowRisk()).execute()
    assert output["ranked_functions"][0] == "p.F.m4()"
    assert {t["action"] for t in output["trace"]} == {a.value for a in Action}
    assert output["screened_count"] == 5
    assert output["actual_tokens"] == len(backend.calls) * 25
    assert output["actual_tokens"] == output["actual_input_tokens"] + output["actual_output_tokens"]
    assert output["tokens_remaining"] == 2000 - output["actual_tokens"]
    assert output["trace"][0]["action"] == "BATCH_SCREEN"
    assert all(s["features"]["level"] == s["level"] for s in output["risk_snapshots"])


def test_invalid_screen_is_paid_and_escalates_to_full():
    backend = FeedbackBackend(invalid=True)
    out = AdaptivePilot(pool(), backend, PolicyConfig(token_budget=1000), LowRisk()).execute()
    assert out["full_verified_count"] == 5
    assert out["actual_tokens"] == 25 * out["call_count"]
    assert all(level == 2 for a, level, ids in backend.calls if a != Action.BATCH_SCREEN)


def test_zero_budget_never_calls_backend():
    backend = FeedbackBackend()
    out = AdaptivePilot(pool(), backend, PolicyConfig(token_budget=0)).execute()
    assert not backend.calls
    assert out["stop_reason"] == "budget_exhausted"
    assert out["actual_tokens"] == 0


def test_actual_overrun_is_visible_and_stops_after_one_call():
    backend = FeedbackBackend(cost=200, estimate=10)
    out = AdaptivePilot(pool(), backend, PolicyConfig(token_budget=100)).execute()
    assert out["call_count"] == 1
    assert out["budget_exceeded"]
    assert out["tokens_remaining"] == -105
    assert out["stop_reason"] == "budget_exhausted"


def test_no_duplicate_paid_levels_and_step_limit_explained():
    out = AdaptivePilot(pool(), FeedbackBackend(), PolicyConfig(max_steps=2)).execute()
    assert out["stop_reason"] == "max_steps"
    seen = [(i, c["level"]) for c in out["calls"] for i in c["candidate_ids"]]
    assert len(seen) == len(set(seen))
    assert out["trace"][-1]["action"] == "STOP"


def test_unverified_prior_not_demoted_behind_all_inspected():
    cs = pool(2)
    status = {c.function_id: CandidateState() for c in cs}
    status[cs[1].function_id] = CandidateState(2, Observation(cs[1].function_id, .01, .01, "healthy"))
    assert fused_score(cs[0], status[cs[0].function_id]) > fused_score(cs[1], status[cs[1].function_id])


def test_shared_ranker_and_all_baseline_actions():
    for mode, levels, calls in [("full", {2}, 5), ("fixed_batch", {1}, 2),
                                ("fixed_sketch", {1}, 5), ("rank_threshold", {2}, 3), ("probe_l0", {0}, 2)]:
        out = AdaptivePilot(pool(), FeedbackBackend(), PolicyConfig(mode=mode), LowRisk()).execute()
        assert out["ranker"] == "shared_prior_evidence_fusion_v1"
        assert {c["level"] for c in out["calls"]} == levels
        assert out["call_count"] == calls


def test_confidence_and_risk_policy_diverge_on_same_feedback():
    class SureNegative(FeedbackBackend):
        def execute(self, candidates, action, level):
            return CallResult([Observation(c.function_id, .1, .05, "low suspicion") for c in candidates], 10, 2, 0)
    class HighRisk(LowRisk):
        def predict(self, f):
            return .99
    adaptive = AdaptivePilot(pool(), SureNegative(), PolicyConfig(), HighRisk()).execute()
    confidence = AdaptivePilot(pool(), SureNegative(), PolicyConfig(mode="confidence"), HighRisk()).execute()
    assert adaptive["full_verified_count"] == 5
    assert confidence["full_verified_count"] == 0


def test_sketch_preserves_operators_literals_throws_dependencies():
    levels, meta = java_sketch("F.x(int)", 'public int x(int n) { int k=offset(n); if(k<=-42) throw new Bug("boom"); return k+3; }')
    assert not meta["parse_error"]
    for text in ("<=-42", "Bug(\"boom\")", "offset(n)", "k+3"):
        assert text in levels[1]
    assert "return" not in levels[0]
    assert levels[2].startswith("public int x")


def test_parse_failure_flagged_not_silently_trusted():
    _, meta = java_sketch("F.x()", "public int x() { if (")
    assert meta["parse_error"]


def test_truncation_flag_is_visible():
    levels, meta = java_sketch("F.x()", 'public int x() {' + ''.join(f'x = next(x+{i});' for i in range(200)) + 'return x;}', 100)
    assert meta["truncated"]
    assert "TRUNCATED" in levels[1]


@pytest.mark.parametrize("raw", ["not json", '{"results":[]}',
    '{"results":[{"id":"c0","suspicion":NaN,"uncertainty":0.1}]}',
    '{"results":[{"id":"c0","suspicion":true,"uncertainty":0.1}]}'])
def test_malformed_json_yields_uncertainty(raw):
    obs, error = parse_response(raw, pool(1), Action.BATCH_SCREEN)
    assert error and not obs[0].valid and obs[0].uncertainty == 1


def test_duplicate_and_foreign_batch_ids_rejected():
    raw = json.dumps({"results": [{"id": "c0", "suspicion": .2, "uncertainty": .1}] * 2})
    obs, error = parse_response(raw, pool(2), Action.BATCH_SCREEN)
    assert error and not any(o.valid for o in obs)


def test_reversed_batch_response_maps_ids_correctly():
    raw = json.dumps({"results": [{"id": "c1", "suspicion": .9, "uncertainty": .1},
                                 {"id": "c0", "suspicion": .1, "uncertainty": .2}]})
    obs, error = parse_response(raw, pool(2), Action.BATCH_SCREEN)
    assert error is None
    assert obs[0].suspicion == .1 and obs[1].suspicion == .9


def test_screen_prompt_requires_exact_candidate_ids_and_count():
    from rave_fl.pilot_backend import build_prompt

    for n in (1, 2, 4):
        candidates = pool(n)
        system, user = build_prompt(candidates, "observed failure", Action.BATCH_SCREEN, 1)
        ids = [f"c{i}" for i in range(n)]
        assert f"THIS BATCH HAS EXACTLY {n} CANDIDATES" in user
        assert f"Return exactly {n} results" in user
        assert json.dumps(ids) in user
        assert "one object" in system.lower()
        shape = user.splitlines()[-1]
        example = json.loads(shape)
        assert [r["id"] for r in example["results"]] == ids
        assert len(example["results"]) == n


def test_incomplete_screen_batch_remains_invalid_even_with_count_prompt():
    from rave_fl.pilot_backend import build_prompt

    candidates = pool(2)
    system, user = build_prompt(candidates, "observed failure", Action.BATCH_SCREEN, 1)
    assert "Return exactly 2 results" in user
    missing_second = json.dumps({"results": [
        {"id": "c0", "suspicion": 0.3, "uncertainty": 0.6}
    ]})
    observations, error = parse_response(missing_second, candidates, Action.BATCH_SCREEN)
    assert "Wrong number of results" in error
    assert len(observations) == 2 and not any(o.valid for o in observations)


def test_full_matches_original_verifier_prompt():
    from rave_fl.pilot_backend import build_prompt
    from rave_fl.ollama_verify import SYSTEM
    c = pool(1)[0]
    system, user = build_prompt([c], "observed failure", Action.FULL_VERIFY, 2)
    assert system == SYSTEM
    assert user == f"FAILURE CONTEXT:\nobserved failure\n\nMETHOD:\n{c.function_id}\n\nSOURCE:\n{c.source}"


def test_full_pool_mrr_is_not_artificially_truncated_to_top5():
    ranking = [f"F.m{i}()" for i in range(8)]
    states = {i: {"level": -1, "observation": None} for i in ranking}
    e = evaluate(ranking, ["F.m7()"], states)
    assert e["MRR"] == 1 / 8 and not e["top5"]
    assert e["candidate_pool_recall"] == 1
    with pytest.raises(ValueError, match="Missing ground truth"):
        evaluate(ranking, [], states)


def test_one_bug_has_no_fake_confidence_interval():
    out = AdaptivePilot(pool(), FeedbackBackend(), PolicyConfig(mode="full")).execute()
    out["accuracy"] = evaluate(out["ranked_functions"], ["p.F.m4()"], out["states"])
    summary = summarize([{"arms": {"full": out, "adaptive": out}}])
    assert summary["adaptive"]["paired_delta"]["MRR"]["bootstrap_95_ci"] is None


def training_rows():
    rows = []
    for bug in ("A-1", "B-1", "C-1"):
        for i in range(30):
            f = {k: 0.0 for k in FEATURE_NAMES}
            f["uncertainty"] = i / 30
            rows.append({"bug": bug, "features": f, "label": int(i > 15),
                         "label_definition": "gold_false_negative_introduced_relative_to_full"})
    return rows


def test_risk_split_rejects_leakage_single_class_and_oracle_features():
    rows = training_rows()
    with pytest.raises(ValueError, match="disjoint"):
        check_split(rows, {"train": ["A-1"], "calibration": ["A-1"], "test": ["C-1"]})
    split = {"train": ["A-1"], "calibration": ["B-1"], "test": ["C-1"]}
    rows[0]["features"]["gold_identity"] = 1
    with pytest.raises(ValueError, match="oracle"):
        check_split(rows, split)
    rows = training_rows()
    for r in rows:
        r["label"] = 0
    with pytest.raises(ValueError, match="both harmful and safe"):
        check_split(rows, split)


def test_lightgbm_train_calibrate_save_reload_and_overlap_guard(tmp_path):
    pytest.importorskip("lightgbm")
    split = {"train": ["A-1"], "calibration": ["B-1"], "test": ["C-1"]}
    artifact = train(training_rows(), split, min_admitted=5)
    p = tmp_path / "risk.json"
    p.write_text(json.dumps(artifact))
    risk = LearnedRisk(p, "C-1")
    assert risk.calibrated
    assert 0 <= risk.predict(training_rows()[0]["features"]) <= 1
    assert artifact["held_out_test_metrics"]["AUROC"] > .8
    with pytest.raises(ValueError, match="Leakage"):
        LearnedRisk(p, "A-1")


def test_backend_records_actual_usage_even_for_invalid_json(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert payload['options']['num_predict'] == 160
            body = json.dumps({'done': True, 'message': {'content': 'bad json'},
                               'prompt_eval_count': 100, 'eval_count': 17}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / 'ledger.jsonl'
        backend = OllamaPilotBackend('fixture', 'failure', f'http://127.0.0.1:{server.server_port}', 160, ledger=path)
        result = backend.execute(pool(1), Action.BATCH_SCREEN, 0)
        assert result.total_tokens == 117
        assert not result.observations[0].valid
        events = [json.loads(s) for s in path.read_text().splitlines()]
        assert [e['event'] for e in events] == ['request', 'response']
        assert events[1]['parsed']['input_tokens'] == 100
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_early_stop_requires_calibration_and_complete_pool_screening():
    class LowCalibrated(LowRisk):
        calibrated = True
        threshold = .1
    # All candidates receive discovery, even with a calibrated low-risk model.
    out = AdaptivePilot(pool(), FeedbackBackend(), PolicyConfig(), LowCalibrated()).execute()
    assert out['screened_count'] == 5
    assert out['trace'][0]['action'] == 'BATCH_SCREEN'
    assert out['trace'][1]['action'] == 'BATCH_SCREEN'
    assert out['full_verified_count'] >= 1
    for row in out['trace']:
        if row['reason'] == 'conservative_calibrated_risk_stop':
            assert row['inspected_count'] == 5


def test_online_ranking_is_invariant_to_offline_gold():
    a = AdaptivePilot(pool(), FeedbackBackend(), PolicyConfig()).execute()
    ranking = list(a['ranked_functions'])
    evaluate(ranking, ['p.F.m0()'], a['states'])
    evaluate(ranking, ['p.F.m4()'], a['states'])
    assert a['ranked_functions'] == ranking
    assert set(a['risk_snapshots'][0]['features']) == set(FEATURE_NAMES)
    assert not any('gold' in k or 'label' in k for k in FEATURE_NAMES)


def make_manifest(tmp_path):
    import csv
    source_dir = tmp_path / 'src/main/java/p'
    source_dir.mkdir(parents=True)
    source_dir.joinpath('F.java').write_text('class F {\n' + '\n'.join(
        f'public int m{i}() {{ return {i}; }}' for i in range(5)) + '\n}')
    for method in ('SBIR', 'Ochiai', 'BoostN'):
        folder = tmp_path / 'data/FL_results' / method / 'Defects4J'
        folder.mkdir(parents=True)
        with folder.joinpath('Toy-1_method-susps.csv').open('w') as out:
            writer = csv.DictWriter(out, fieldnames=['File', 'Signature', 'StartLine', 'EndLine'])
            writer.writeheader()
            writer.writerows({'File': 'p.F', 'Signature': f'm{i}()', 'StartLine': i + 2, 'EndLine': i + 2}
                             for i in range(5))
    gold = tmp_path / 'data/input/ground_truth/Defects4J/gt.json'
    gold.parent.mkdir(parents=True)
    gold.write_text(json.dumps({'Toy-1': ['p.F.m4()']}))
    evidence = tmp_path / 'observed_failure.txt'
    evidence.write_text('Observed assertion mismatch in production output')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': [{'bug': 'Toy-1', 'flexfl_repo': str(tmp_path),
        'checkout': str(tmp_path), 'evidence_file': str(evidence)}]}))
    return manifest


def test_manifest_to_paired_report_and_risk_labels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rave_fl.pilot import run_experiment
    class FakeOllama(FeedbackBackend):
        def __init__(self, *args):
            super().__init__()
            self.model_metadata = {'name': 'fixture'}
        def preflight(self):
            return self.model_metadata
    monkeypatch.setattr('rave_fl.pilot.OllamaPilotBackend', FakeOllama)
    args = SimpleNamespace(manifest=make_manifest(tmp_path), max_bugs=5,
        arms='full,adaptive', collect_risk=True, risk_model=None, model='fixture',
        dry_run=False, token_budget=3000, batch_size=4, risk_threshold=.35,
        uncertainty_threshold=.45, no_early_stop=False, max_steps=200, top_per_method=10,
        output=tmp_path / 'outputs', base_url='fixture', seed=0,
        max_output_tokens=160, context_window=16384, timeout=10)
    out = run_experiment(args)
    assert out['status'] == 'complete'
    assert out['summary']['adaptive']['n_bugs'] == 1
    assert (args.output / 'report.md').is_file()
    rows = [json.loads(s) for s in (args.output / 'risk_dataset.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    assert {r['level'] for r in rows} == {0, 1}
    assert all(set(r['features']) == set(FEATURE_NAMES) for r in rows)
    with pytest.raises(ValueError, match='empty/new'):
        run_experiment(args)


def test_reused_risk_export_rejects_incomplete_candidate_coverage():
    from rave_fl.pilot_evaluation import risk_rows
    cs = pool()
    full = AdaptivePilot(cs, FeedbackBackend(), PolicyConfig(mode='full')).execute()
    l0 = AdaptivePilot(cs, FeedbackBackend(), PolicyConfig(mode='probe_l0')).execute()
    l1 = AdaptivePilot(cs, FeedbackBackend(), PolicyConfig(mode='fixed_batch')).execute()
    case = {'arms': {'full': full, 'confidence': l0, 'fixed_batch': l1}, 'inference_fingerprint': 'fixture'}
    rows = risk_rows('Toy-1', case, ['p.F.m4()'])
    assert len(rows) == 10
    assert {r['source_arm'] for r in rows} == {'confidence', 'fixed_batch'}
    l1['risk_snapshots'].pop()
    with pytest.raises(ValueError, match='exactly once'):
        risk_rows('Toy-1', case, ['p.F.m4()'])


def test_non_full_budget_can_be_tighter_than_full_reference(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rave_fl.pilot import run_experiment
    class FakeOllama(FeedbackBackend):
        def __init__(self, *args):
            super().__init__()
            self.model_metadata = {'name': 'fixture'}
        def preflight(self):
            return self.model_metadata
    monkeypatch.setattr('rave_fl.pilot.OllamaPilotBackend', FakeOllama)
    args = SimpleNamespace(manifest=make_manifest(tmp_path), max_bugs=5,
        arms='full,adaptive', collect_risk=False, risk_model=None, model='fixture',
        dry_run=False, token_budget=3000, policy_budget=0, batch_size=4, risk_threshold=.35,
        uncertainty_threshold=.45, no_early_stop=False, max_steps=200, top_per_method=10,
        output=tmp_path / 'outputs', base_url='fixture', seed=0,
        max_output_tokens=160, context_window=16384, timeout=10)
    out = run_experiment(args)
    arms = out['cases'][0]['arms']
    assert arms['full']['full_verified_count'] == 5
    assert arms['adaptive']['call_count'] == 0
    assert arms['adaptive']['config']['token_budget'] == 0
    assert arms['adaptive']['stop_reason'] == 'budget_exhausted'
    assert (args.output / 'code_snapshot/rave_fl/pilot.py').is_file()


def test_skip_uneconomic_expansion_is_explicit():
    class ExpensiveExpansion(FeedbackBackend):
        def estimate(self, candidates, action, level):
            return 120 if level == 1 else 30
    out = AdaptivePilot(pool(), ExpensiveExpansion(), PolicyConfig(), LowRisk()).execute()
    assert any(t['reason'] == 'skip_uneconomic_expansion' for t in out['trace'])
    assert not any(c['action'] == 'EXPAND_CONTEXT' for c in out['calls'])


def test_ranking_target_uses_versioned_offline_labels(tmp_path):
    rows = training_rows()
    for r in rows:
        r['ranking_harm_label'] = r['label']
        r['ranking_label_definition'] = 'counterfactual_cheap_context_MRR_loss_relative_to_full_action'
    split = {'train': ['A-1'], 'calibration': ['B-1'], 'test': ['C-1']}
    artifact = train(rows, split, target='ranking', min_admitted=5)
    assert artifact['target'] == 'ranking'
    assert artifact['label_definition'] == 'counterfactual_cheap_context_MRR_loss_relative_to_full_action'
    assert artifact['held_out_test_metrics']['AUROC'] > .8
    rows[0].pop('ranking_label_definition')
    with pytest.raises(ValueError, match='offline counterfactual ranking labels'):
        train(rows, split, target='ranking')

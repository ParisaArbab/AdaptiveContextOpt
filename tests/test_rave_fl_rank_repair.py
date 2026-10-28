"""Rank repair: frozen ranking, paid replay, budget, and oracle isolation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rave_fl.adaptive import AdaptivePilot
from rave_fl.code_sketch import prepare_candidates
from rave_fl.core import Action, Candidate
from rave_fl.evidence_ablation import mask_candidate_names
from rave_fl.pilot_backend import build_prompt, parse_response
from rave_fl.pilot_types import CallResult, PolicyConfig
from rave_fl.rank_repair import RankRepairPilot, RepairConfig
from rave_fl.replay import ReplayCorpus


def candidates():
    cs = [Candidate('p.C.a()', 'a()', semantic_score=.9),
          Candidate('p.C.b()', 'b()', semantic_score=.6),
          Candidate('p.C.c()', 'c()', semantic_score=.1)]
    source = {c.function_id: 'public int ' + c.signature + ' { return 1; }' for c in cs}
    metadata = {c.function_id: {'methods': ['SBIR'], 'ranks': {'SBIR': n + 1}} for n, c in enumerate(cs)}
    return prepare_candidates(cs, source, metadata, 'observed assertion failure')


class RecordedBackend:
    context_window = 16384
    def __init__(self):
        self.actions = []
    def estimate(self, selected, action, level):
        return 40 if level == 1 else 30
    def execute(self, selected, action, level):
        self.actions.append((action.value, level, tuple(c.function_id for c in selected)))
        if level == 2:
            score = .9 if selected[0].function_id == 'p.C.c()' else .1
            raw = json.dumps({'suspicion': score, 'explanation': 'complete code observation'})
        else:
            scores = {'p.C.a()': .7, 'p.C.b()': .3, 'p.C.c()': .95}
            raw = json.dumps({'results': [{'id': f'c{i}', 'suspicion': scores[c.function_id],
                                           'uncertainty': .1, 'explanation': 'cheap observation'}
                                          for i, c in enumerate(selected)]})
        obs, error = parse_response(raw, selected, action)
        return CallResult(obs, 20, 5, .01, raw, error)


def execute(mode='repair_utility', budget=1000, max_full=4):
    backend = RecordedBackend()
    out = RankRepairPilot(candidates(), backend, PolicyConfig(token_budget=budget, batch_size=2, mode=mode),
                          RepairConfig(max_full, 5)).execute()
    return out, backend


def test_l1_first_utility_repairs_with_one_full_and_explicit_stop():
    out, backend = execute()
    assert [a[1] for a in backend.actions] == [1, 1, 2]
    assert backend.actions[-1][2] == ('p.C.c()',)
    assert out['ranked_functions'][0] == 'p.C.c()'
    assert out['stop_reason'] == 'no_positive_rank_proxy'
    assert out['full_verified_count'] == 1
    assert out['actual_tokens'] == 75
    assert out['initial_l1']['tokens_used'] == 50
    assert not out['utility_is_calibrated'] and not out['risk_calibrated']
    assert out['ranker'] == 'shared_prior_evidence_fusion_v1'


def test_full_action_limit_and_no_l0_or_expand():
    out, backend = execute('repair_top', max_full=1)
    assert out['full_verified_count'] == 1
    assert out['stop_reason'] == 'full_repair_limit'
    assert all(level in (1, 2) for _, level, ids in backend.actions)
    assert all(action != 'EXPAND_CONTEXT' for action, _, ids in backend.actions)


def test_unaffordable_full_preserves_complete_l1_result():
    out, backend = execute(budget=75)
    assert out['screened_count'] == 3
    assert out['full_verified_count'] == 0
    assert out['stop_reason'] == 'no_affordable_full_repair'
    fixed = AdaptivePilot(candidates(), RecordedBackend(), PolicyConfig(token_budget=1000, batch_size=2, mode='fixed_batch')).execute()
    assert out['ranked_functions'] == fixed['ranked_functions']
    assert out['actual_tokens'] == fixed['actual_tokens']


def test_partial_discovery_does_not_change_batch_composition_to_fit():
    out, backend = execute(budget=30)
    assert not backend.actions
    assert out['stop_reason'] == 'budget_admission_before_l1_coverage'
    assert out['screened_count'] == 0


def test_zero_full_limit_is_fixed_l1():
    out, backend = execute(max_full=0)
    assert out['full_verified_count'] == 0
    assert out['stop_reason'] == 'full_repair_limit'


def test_modes_choose_distinct_first_actions_without_gold():
    selected = {}
    for mode in ('repair_top', 'repair_suspicion', 'repair_utility'):
        out, _ = execute(mode, max_full=1)
        selected[mode] = next(c['candidate_ids'][0] for c in out['calls'] if c['level'] == 2)
    assert selected['repair_top'] == 'p.C.a()'
    assert selected['repair_suspicion'] == selected['repair_utility'] == 'p.C.c()'


def test_evidence_mask_preserves_assertions_scenarios_and_non_method_frames():
    context = ('TEST: p.TestC::test_c_overlap_2007\n'
               'FAILURE: expected:<+04:00> but was:<+03:00>\n'
               'at p.TestC.doTest_c(TestC.java:123)\n')
    masked, meta = mask_candidate_names(context, ['p.C.c()'])
    assert 'FAILURE: expected:<+04:00> but was:<+03:00>' in masked
    assert 'overlap_2007' in masked and 'TestC.java:123' in masked
    assert 'test_c_' not in masked and 'doTest_c(' not in masked
    assert meta['changed_occurrences'] == {'c': 2}
    assert meta['assertion_lines_byte_preserved'] and not meta['gold_used']


def make_disk_corpus(tmp_path, monkeypatch):
    cs = candidates()
    provenance = {'sources_sha256': 'sources', 'evidence': {'sha256': 'evidence'}}
    monkeypatch.setattr('rave_fl.replay.prepare_case', lambda case, top: (cs, 'context', provenance))
    arms = {}
    for mode, name in [('full', 'full'), ('fixed_batch', 'fixed_batch')]:
        arm = AdaptivePilot(cs, RecordedBackend(), PolicyConfig(token_budget=20000, batch_size=2, mode=mode)).execute()
        arm['model_metadata'] = {'digest': 'frozen-model'}
        arms[name] = arm
        path = tmp_path / 'Toy-1' / (name + '_calls.jsonl')
        path.parent.mkdir(exist_ok=True)
        entries = []
        for n, call in enumerate(arm['calls'], 1):
            selected = [c for i in call['candidate_ids'] for c in cs if c.function_id == i]
            system, user = build_prompt(selected, 'context', Action(call['action']), call['level'])
            payload = {'model': 'fixture', 'stream': False, 'think': False, 'format': 'json',
                       'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                       'options': {'temperature': 0, 'seed': 0, 'num_ctx': 16384, 'num_predict': 160 * len(selected)}}
            entries += [{'event': 'request', 'call_id': n, 'request': payload,
                         'request_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                         'candidate_ids': call['candidate_ids'], 'action': call['action'], 'level': call['level']},
                        {'event': 'response', 'call_id': n,
                         'response': {'done': True, 'message': {'content': call['raw_response']},
                                      'prompt_eval_count': call['input_tokens'], 'eval_count': call['output_tokens']}}]
        path.write_text(''.join(json.dumps(e) + '\n' for e in entries))
    run = {'status': 'complete', 'dry_run': False, 'model': 'fixture', 'seed': 0,
           'cases': [{'bug': 'Toy-1', 'top_per_method': 10, **provenance, 'arms': arms}]}
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(run))
    return path, {'bug': 'Toy-1'}


def test_replay_exact_action_is_paid_from_recorded_counters_and_never_mutates_cache(tmp_path, monkeypatch):
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    corpus = ReplayCorpus(path, case)
    backend = corpus.backend()
    first = backend.execute(corpus.candidates[:2], Action.BATCH_SCREEN, 1)
    assert first.total_tokens == 25
    first.input_tokens = 99999
    assert backend.execute(corpus.candidates[:2], Action.BATCH_SCREEN, 1).total_tokens == 25
    with pytest.raises(ValueError, match='No exact cached action'):
        backend.execute(list(reversed(corpus.candidates[:2])), Action.BATCH_SCREEN, 1)


def test_future_full_answers_and_usage_do_not_enter_first_policy_choice(tmp_path, monkeypatch):
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    corpus = ReplayCorpus(path, case)
    config = PolicyConfig(token_budget=20000, batch_size=2, mode='repair_utility')
    a = RankRepairPilot(corpus.candidates, corpus.backend(), config, RepairConfig(1)).execute()
    estimate = corpus.backend().estimate([corpus.candidates[-1]], Action.FULL_VERIFY, 2)
    for key, value in corpus._entries.items():
        if key[0] == 'FULL_VERIFY':
            value.input_tokens = 30000
            value.observations[0] = type(value.observations[0])(value.observations[0].function_id, .01, .01, 'changed future answer')
    b = RankRepairPilot(corpus.candidates, corpus.backend(), config, RepairConfig(1)).execute()
    assert corpus.backend().estimate([corpus.candidates[-1]], Action.FULL_VERIFY, 2) == estimate
    first_a = next(c['candidate_ids'] for c in a['calls'] if c['level'] == 2)
    first_b = next(c['candidate_ids'] for c in b['calls'] if c['level'] == 2)
    assert first_a == first_b
    assert b['budget_exceeded']  # actual surprise is charged, not hidden


def test_replay_refuses_history_dependent_or_changed_prompts(tmp_path, monkeypatch):
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    ledger = tmp_path / 'Toy-1/full_calls.jsonl'
    entries = [json.loads(s) for s in ledger.read_text().splitlines()]
    entries[0]['request']['messages'].append({'role': 'assistant', 'content': 'previous hypothesis'})
    entries[0]['request_sha256'] = hashlib.sha256(json.dumps(entries[0]['request'], sort_keys=True).encode()).hexdigest()
    ledger.write_text(''.join(json.dumps(e) + '\n' for e in entries))
    with pytest.raises(ValueError, match='Prompt changed or contains action history'):
        ReplayCorpus(path, case)


def test_replay_refuses_changed_model_weights(tmp_path, monkeypatch):
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    data = json.loads(path.read_text())
    data['cases'][0]['arms']['fixed_batch']['model_metadata']['digest'] = 'different'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Model weights differ'):
        ReplayCorpus(path, case)


def test_oracle_is_diagnostic_and_detects_nonadditive_action_effects(tmp_path, monkeypatch):
    from rave_fl.replay import oracle_frontier
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    corpus = ReplayCorpus(path, case)
    oracle = oracle_frontier(corpus, ['p.C.c()'], max_full=2, budget=100)
    assert oracle['uses_gold_and_future_observed_costs']
    assert not oracle['deployable_policy']
    assert oracle['frontier'][1]['best']['MRR'] == 1
    assert abs(oracle['strongest_pair_interaction']['interaction_MRR']) > 0


def test_action_value_labels_keep_future_quality_and_actual_cost_out_of_features(tmp_path, monkeypatch):
    from rave_fl.rank_repair_labels import action_value_rows
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    corpus = ReplayCorpus(path, case)
    arm = RankRepairPilot(corpus.candidates, corpus.backend(),
        PolicyConfig(token_budget=20000, batch_size=2, mode='repair_utility'), RepairConfig(4)).execute()
    rows = action_value_rows(corpus, ['p.C.c()'], [arm], 'Toy-1', 20000)
    assert len({r['bug'] for r in rows}) == 1
    assert len({r['state_id'] for r in rows}) > 1
    assert any(r['targets']['delta_MRR'] > 0 for r in rows)
    forbidden = {'delta_MRR', 'observed_full_tokens', 'gold', 'is_gold', 'full_suspicion'}
    assert all(not (set(r['features']) & forbidden) for r in rows)
    assert all('full_cost_estimate' in r['features'] for r in rows)


def test_replay_refuses_changed_reference_ranking_even_with_same_ranker_name(tmp_path, monkeypatch):
    path, case = make_disk_corpus(tmp_path, monkeypatch)
    data = json.loads(path.read_text())
    i = 'p.C.a()'
    data['cases'][0]['arms']['fixed_batch']['ranking_scores'][i] = .999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Frozen ranker or priors differ'):
        ReplayCorpus(path, case)


def test_normalization_derives_only_observed_numeric_offset_deltas():
    from rave_fl.evidence_ablation import normalize_assertion_offsets
    context = ('TEST: masked\nFAILURE: junit.ComparisonFailure: '
               'expected:<...10-28T02:00:00.000+0[4]:00> but was:<...10-28T02:00:00.000+0[3]:00>\n')
    normalized, meta = normalize_assertion_offsets(context)
    assert normalized.endswith(context)
    assert meta['parsed_pairs'] == [{'expected_offset': '+04:00', 'observed_offset': '+03:00',
                                     'expected_minus_observed_ms': 3600000}]
    assert not meta['diagnosis_added'] and not meta['gold_used']
    assert not meta['missing_text_reconstructed']
    with pytest.raises(ValueError, match='refuse to infer missing values'):
        normalize_assertion_offsets('FAILURE: expected:<[4]> but was:<[3]>')

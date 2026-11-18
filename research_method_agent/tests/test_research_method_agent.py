import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_method_agent as m


def fixture():
    manifest = {'schema': 'research-method-agent-v1', 'bugs': []}
    observations = {'schema': 'research-agent-observations-v1', 'provenance': 'SYNTHETIC_TEST_ONLY', 'observations': []}
    gold = {'schema': 'research-agent-gold-v1', 'bugs': {}}
    for x in range(6):
        bid = f'Fixture-{x}'
        candidates = [{'id': f'c{i}', 'prior': v, 'l1': f'foo {i}',
                       'full': f'def foo{i}(x):\n return x + {i}',
                       'depends_on': ['c2'] if i == 0 else []}
                      for i, v in enumerate((0.9, 0.6, 0.4))]
        manifest['bugs'].append({'id': bid, 'evidence': 'AssertionError: expected 4 got 5',
                                 'candidates': candidates})
        gold['bugs'][bid] = {'functions': ['c2']}
        for c in candidates:
            i = int(c['id'][-1])
            for kind in ('l1', 'full', 'card', 'llmlingua2') + (('dependency',) if i == 0 else ()): 
                score = (0.7, 0.6, 0.3)[i]
                if kind == 'full' and i == 2:
                    score = 0.98
                if kind == 'dependency' and i == 0:
                    score = 0.1
                observations['observations'].append({
                    'bug': bid, 'candidate': c['id'], 'kind': kind, 'status': 'ok',
                    'score': score, 'uncertainty': 0.9 if i == 2 else 0.45,
                    'missing_dependency': bool(kind == 'l1' and i == 0),
                    'input_tokens': 60 if kind == 'full' else 30,
                    'output_tokens': 10, 'model': 'SYNTHETIC_TEST_ONLY'})
    observations['manifest_hash'] = m.sha(manifest)
    return manifest, observations, gold


class TestResearchAgent(unittest.TestCase):
    def test_adaptive_k_detects_largest_gap(self):
        c = [{'id': 'a', 'prior': 0.98}, {'id': 'b', 'prior': 0.90},
             {'id': 'c', 'prior': 0.55}, {'id': 'd', 'prior': 0.50}]
        self.assertEqual(m.adaptive_k(c, buffer=0), 2)
        self.assertEqual(m.adaptive_k(c, buffer=1), 3)

    def test_manifest_validation(self):
        manifest, _, _ = fixture()
        m.validate(manifest)
        manifest['bugs'][0]['candidates'][0]['depends_on'] = ['missing']
        with self.assertRaises(ValueError):
            m.validate(manifest)

    def test_missing_observation_incomplete_not_imputed(self):
        manifest, observation, _ = fixture()
        data = m.obs_index(observation)
        key = m.observation_key('Fixture-0', 'c0', 'dependency')
        del data[key]
        result = m.evaluate_arm(manifest['bugs'][0], data, 'dependency_repair', 20000)
        self.assertTrue(result['status'].startswith('missing_real_observation:'))

    def test_gold_changes_cannot_change_online_results(self):
        manifest, observations, gold = fixture()
        data = m.obs_index(observations)
        first = m.evaluate_arm(manifest['bugs'][0], data, 'risk_full', 20000)
        gold['bugs']['Fixture-0']['functions'] = ['c1']
        second = m.evaluate_arm(manifest['bugs'][0], data, 'risk_full', 20000)
        self.assertEqual(first, second)

    def test_real_observation_usage_and_paired_results(self):
        manifest, observations, gold = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result = m.evaluate(manifest, observations, gold, tmp, 20000, 2, 1)
            self.assertEqual(result['bugs'], 6)
            summary = m.load(Path(tmp) / 'results.json')['summary']
            l1 = next(x for x in summary if x['arm'] == 'l1')
            risk = next(x for x in summary if x['arm'] == 'risk_full')
            self.assertEqual(l1['paired_n'], 6)
            self.assertEqual(risk['paired_n'], 6)
            self.assertGreater(risk['delta_mrr_vs_l1'], 0)
            self.assertIsNotNone(risk['delta_mrr_ci95'])
            self.assertTrue((Path(tmp) / 'report.md').exists())
            self.assertTrue((Path(tmp) / 'summary.csv').exists())

    def test_budget_enforcement_reports_overrun(self):
        manifest, observation, _ = fixture()
        data = m.obs_index(observation)
        item = data[m.observation_key('Fixture-0','c0','l1')]
        item['input_tokens'] = 20000
        result = m.evaluate_arm(manifest['bugs'][0], data, 'l1', 500)
        self.assertEqual(result['status'], 'measured_budget_overrun')
        self.assertTrue(result['overrun'])

    def test_missing_dependency_not_silently_restored(self):
        manifest, observation, _ = fixture()
        data = m.obs_index(observation)
        case = dict(manifest['bugs'][0])
        case['candidates'] = [dict(c) for c in case['candidates']]
        for c in case['candidates']:
            c['depends_on'] = []
        result = m.evaluate_arm(case, data, 'dependency_repair', 20000)
        self.assertTrue(result['status'] == 'complete')
        self.assertFalse(any(a['kind'] == 'dependency' for a in result['actions']))

    def test_different_manifests_rejected(self):
        manifest, obs, gold = fixture()
        manifest['bugs'][0]['evidence'] = 'changed'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.evaluate(manifest, obs, gold, tmp, 20000, 2, 1)



class TestIntegration(unittest.TestCase):
    def test_import_prepared_run_and_separate_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run' / 'instances' / 'bug'
            root.mkdir(parents=True)
            prepared = {'online': {'instance_id':'Time-25', 'dataset': 'defects4j',
                      'evidence': {'text': 'observed failing test'}},
                      'entities': [
                          {'entity_id':'src/X.java::X.first()', 'path':'src/X.java'},
                          {'entity_id':'src/X.java::X.second()', 'path':'src/X.java'}],
                      'candidates': [
                          {'function_id':'src/X.java::X.first()', 'prior':0.9,
                           'levels':['first()', 'call second();', 'return second();']},
                          {'function_id':'src/X.java::X.second()', 'prior':0.2,
                           'levels':['second()', 'return 0;', 'return 0;']}],
                      'postfix_metadata': {'other': 4}}
            (root/'input.json').write_text(json.dumps(prepared))
            (root/'evaluation_only.json').write_text(json.dumps({
                'files':['src/X.java'], 'functions':[], 'semantics':'file_proxy'}))
            filename = Path(tmp)/'manifest.json'
            m.import_rave_inputs(root.parent.parent, filename, 5, 2)
            saved = m.load(filename)
            self.assertEqual(saved['bugs'][0]['candidates'][0]['depends_on'],
                             ['src/X.java::X.second()'])
            self.assertNotIn('gold', saved['bugs'][0])
            gold_file = Path(tmp)/'labels.json'
            m.import_rave_gold(root.parent.parent, saved, gold_file)
            self.assertEqual(m.load(gold_file)['bugs']['Time-25']['files'], ['src/X.java'])

    def test_offline_sweep_is_deterministic(self):
        manifest, observations, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result = m.run_sweep(manifest, observations, labels, tmp,
                                 budgets=[600, 1000], max_verifies=[1, 2], buffers=[0, 1])
            self.assertEqual(result['configs'], 8)
            self.assertTrue((Path(tmp) / 'sweep_report.md').exists())
            self.assertTrue((Path(tmp) / 'grid.json').exists())

    def test_conformal_small_n_is_all_candidates(self):
        manifest, observations, labels = fixture()
        calib = manifest['bugs'][:3]
        self.assertEqual(m.conformal_threshold(calib, labels['bugs'], alpha=0.10), 0.0)
        self.assertGreater(m.conformal_threshold(calib, labels['bugs'], alpha=0.4), 0.0)



class TestMalformedModelRecovery(unittest.TestCase):
    def test_unpaired_surrogate_saves_and_round_trips_without_invalid_utf8(self):
        source = {'observations': [{'raw_response': 'model \udc22 output',
                                    'reason': 'broken \udcff string'}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            m.save(path, source)
            data = path.read_bytes()
            self.assertIn(b'\\udc22', data)
            self.assertIn(b'\\udcff', data)
            self.assertEqual(m.load(path), source)
            self.assertEqual(m.sha(source), m.sha(m.load(path)))
            self.assertFalse(path.with_name('observations.json.tmp').exists())

    def test_interrupted_save_preserves_checkpoint_and_flags_usage_gap(self):
        manifest, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            saved = {'schema': 'research-agent-observations-v1',
                     'provenance': 'live_ollama', 'manifest_hash': m.sha(manifest),
                     'models': {'l1': 'qwen2.5:7b', 'full': 'qwen2.5:7b'},
                     'observations': [], 'attempts': [], 'legacy_observation_tokens': 0}
            m.save(path, saved)
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_bytes(b'{"incomplete": "unterminated')
            with patch.object(m, 'ollama_call') as spy:
                result = m.collect(manifest, path, 'http://localhost',
                                   'qwen2.5:7b', 'qwen2.5:7b', 1, 1, 0)
                spy.assert_not_called()
            self.assertTrue(result['limit_reached'])
            self.assertFalse(result['physical_accounting_complete'])
            self.assertIn('lower bound', result['accounting_warnings'][0])
            self.assertEqual(path.with_name(path.name + '.interrupted-write.bak').read_bytes(),
                             b'{"incomplete": "unterminated')
            self.assertFalse(m.load(path)['physical_accounting_complete'])
            self.assertFalse(temporary.exists())

    def test_surrogate_response_collected_with_metered_usage(self):
        manifest, _, _ = fixture()
        response = {'status': 'ok', 'score': 0.75, 'uncertainty': 0.4,
                    'missing_dependency': False, 'reason': 'model \udc22 response',
                    'raw_response': 'model \udc22 response', 'usage_known': True,
                    'input_tokens': 250, 'output_tokens': 100}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            with patch.object(m, 'ollama_call', return_value=response):
                outcome = m.collect(manifest, path, 'http://localhost',
                                    'qwen2.5:7b', 'qwen2.5:7b', 1, 1, 1)
            state = m.load(path)
            self.assertEqual(outcome['physical_tokens'], 350)
            self.assertEqual(len(state['attempts']), 1)
            self.assertEqual(len(state['observations']), 1)
            self.assertEqual(state['observations'][0]['raw_response'], 'model \udc22 response')

    def test_truncated_json_preserves_provider_usage_and_requests_schema(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({
                    'prompt_eval_count': 154, 'eval_count': 130,
                    'done_reason': 'length',
                    'message': {'content': '{"suspicion": 0.5, "uncertainty": 0.2, '
                                           '"missing_dependency": false, "explanation": "unfinished'},
                }).encode()
        requests = []
        def stub(req, timeout):
            requests.append(json.loads(req.data))
            return Response()
        with patch.object(m.urllib.request, 'urlopen', side_effect=stub):
            result = m.ollama_call('http://127.0.0.1:11434', 'qwen2.5:7b',
                                   'AssertionError', 'X::f()', 'return 1;', 'l1')
        self.assertEqual(result['status'], 'invalid_model_output')
        self.assertEqual(result['input_tokens'] + result['output_tokens'], 284)
        self.assertEqual(result['done_reason'], 'length')
        self.assertEqual(requests[0]['options']['num_predict'], 384)
        self.assertIsInstance(requests[0]['format'], dict)
        self.assertTrue(requests[0]['format']['additionalProperties'] is False)

    def test_invalid_response_is_logged_and_retried_once(self):
        manifest, _, _ = fixture()
        caps = []
        def fake_call(url, model, evidence, candidate, source, kind, max_output=384):
            caps.append(max_output)
            if len(caps) == 1:
                return {'status': 'invalid_model_output', 'error': 'truncated',
                        'usage_known': True, 'input_tokens': 100, 'output_tokens': 130,
                        'raw_response': '{"explanation":"unfinished'}
            return {'status': 'ok', 'score': 0.72, 'uncertainty': 0.2,
                    'missing_dependency': False, 'reason': 'ok',
                    'usage_known': True, 'input_tokens': 105, 'output_tokens': 32,
                    'raw_response': '{"suspicion":0.72}'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            with patch.object(m, 'ollama_call', side_effect=fake_call):
                state = m.collect(manifest, path, 'http://localhost', 'qwen2.5:7b',
                                  'qwen2.5:7b', 1, 1, 2)
            data = m.load(path)
            self.assertTrue(state['limit_reached'])
            self.assertEqual(state['physical_tokens'], 367)
            self.assertEqual([a['status'] for a in data['attempts']],
                             ['invalid_model_output', 'ok'])
            self.assertEqual(len(data['observations']), 1)
            self.assertEqual(caps, [384, 768])
            with patch.object(m, 'ollama_call', return_value={
                'status': 'ok', 'score': 0.5, 'uncertainty': 0.5,
                'missing_dependency': False, 'input_tokens': 10,
                'output_tokens': 10, 'usage_known': True}) as run:
                rest = m.collect(manifest, path, 'http://localhost', 'qwen2.5:7b',
                                 'qwen2.5:7b', 1, 1, 1)
                self.assertEqual(rest['physical_tokens'], 387)
                self.assertEqual(len(m.load(path)['observations']), 2)
                self.assertEqual(run.call_count, 1)

    def test_repeated_invalid_responses_do_not_create_observation(self):
        manifest, _, _ = fixture()
        def invalid(*args, **kwargs):
            return {'status': 'invalid_model_output', 'error': 'truncated',
                    'usage_known': True, 'input_tokens': 10, 'output_tokens': 130}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            with patch.object(m, 'ollama_call', side_effect=invalid):
                with self.assertRaisesRegex(RuntimeError, 'Two invalid JSON'):
                    m.collect(manifest, path, 'http://localhost', 'qwen2.5:7b',
                              'qwen2.5:7b', 1, 1, 10)
            data = m.load(path)
            self.assertEqual(len(data['observations']), 0)
            self.assertEqual(len(data['attempts']), 2)
            self.assertEqual(sum(x['input_tokens'] + x['output_tokens']
                                 for x in data['attempts']), 280)
            with patch.object(m, 'ollama_call') as spy:
                with self.assertRaisesRegex(RuntimeError, 'manual review required'):
                    m.collect(manifest, path, 'http://localhost', 'qwen2.5:7b',
                              'qwen2.5:7b', 1, 1, 10)
                spy.assert_not_called()

    def test_legacy_unmetered_failure_needs_new_output(self):
        manifest, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.json'
            m.save(path, {'schema': 'research-agent-observations-v1',
                          'manifest_hash': m.sha(manifest),
                          'models': {'l1': 'qwen2.5:7b', 'full': 'qwen2.5:7b'},
                          'observations': [{'bug': 'Fixture-0', 'candidate': 'c0',
                                            'kind': 'l1', 'status': 'error',
                                            'error': 'Unterminated string'}]})
            with self.assertRaisesRegex(ValueError, 'NEW output directory'):
                m.collect(manifest, path, 'http://localhost', 'qwen2.5:7b',
                          'qwen2.5:7b', 1, 1, 10)
            self.assertEqual(len(m.load(path)['observations']), 1)

if __name__ == "__main__":
    unittest.main()
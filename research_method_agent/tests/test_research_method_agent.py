import json
from pathlib import Path
import sys
import tempfile
import unittest

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

if __name__ == "__main__":
    unittest.main()
"""No-inference tests for Top-8 mixed context and BM25 retrieval filters."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research_method_agent'))

import mixed_top8_validation as mixed
import audit_high_recall_filter as filt
import research_method_agent as agent


class TestMixedFrozenPolicy(unittest.TestCase):
    def fixture(self):
        bug = {'id': 'Synthetic-1', 'evidence': 'fails only on method calls', 'candidates': [
            {'id': 'M' + str(i), 'file': 'pkg/File' + str(i) + '.java',
             'prior': 1 - i / 24, 'l1': 'java test', 'full': 'public int go() { return 1; }'}
            for i in range(24)]}
        obs = {}
        for c in bug['candidates']:
            for kind in ('l1', 'card'):
                obs[agent.observation_key(bug['id'], c['id'], kind)] = {
                    'status': 'ok', 'score': 0.3 + (0.4 if c['id'] == 'M1' else 0),
                    'uncertainty': 0.5, 'input_tokens': 100 if kind == 'l1' else 40,
                    'output_tokens': 10, 'missing_dependency': False,
                }
        return bug, obs

    def test_policy_prior_top8_exact(self):
        self.assertEqual([mixed.kind_for_rank(i) for i in range(24)],
                         ['l1'] * 8 + ['card'] * 16)

    def test_policy_evaluates_real_observation_and_provider_tokens(self):
        bug, obs = self.fixture()
        r = mixed.mixed_rank(bug, obs)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['tokens'], 8 * 110 + 16 * 50)
        self.assertEqual(r['llm_calls'], 24)
        self.assertEqual(r['full_calls'], 0)
        self.assertEqual([a['kind'] for a in r['actions']], ['l1'] * 8 + ['card'] * 16)
        self.assertEqual(r['ranked'][0], 'M1')

    def test_missing_llm_result_never_synthesized(self):
        bug, obs = self.fixture()
        del obs[agent.observation_key(bug['id'], 'M8', 'card')]
        r = mixed.mixed_rank(bug, obs)
        self.assertEqual(r['status'], 'missing_real_observation')

    def test_budget_overrun_is_not_complete(self):
        bug, obs = self.fixture()
        r = mixed.mixed_rank(bug, obs, token_budget=100)
        self.assertEqual(r['status'], 'measured_budget_overrun')

    def test_extra_bugs_are_new_and_configs_unchanged(self):
        seen = set(mixed.OLD_BUGS)
        for batch in mixed.BATCHES:
            config_path = ROOT / 'data/graft_100_cohort_20261115/batches' / (
                'batch_' + batch) / 'experiment/config.json'
            opts = json.loads(config_path.read_text())['benchmark']
            self.assertEqual(opts['candidate_count'], 24)
            self.assertEqual(opts['candidate_provider'], 'bm25')
            bugs = set(opts['bugs'])
            self.assertEqual(len(bugs), 5)
            self.assertFalse(bugs & seen)
            seen |= bugs
        self.assertEqual(len(seen), 30)

    def test_collector_refuses_paid_call_without_consent(self):
        with self.assertRaisesRegex(RuntimeError, '--allow-inference'):
            mixed.collect('03', SimpleNamespace(allow_inference=False))


class TestRetrievalFilter(unittest.TestCase):
    @staticmethod
    def fixtures():
        # The target fault is 32nd in raw BM25, and filename appears in the
        # failed test log. No gold label is passed into the selection policy.
        candidates = [SimpleNamespace(
            path='src/c' + str(i) + '/File' + str(i) + '.java',
            entity_id='pkg/File' + str(i) + '.java::Class.method' + str(i) + '()')
            for i in range(50)]
        return candidates, 'NullPointerException at File31.java:15'

    def test_baseline_and_anchor_response(self):
        candidates, evidence = self.fixtures()
        base = filt.select_indices(candidates, evidence, 'bm25_top24')
        anchored = filt.select_indices(candidates, evidence, 'failure_anchor_first')
        self.assertEqual(base, list(range(24)))
        self.assertEqual(anchored[0], 31)
        self.assertEqual(len(anchored), 24)
        self.assertEqual(len(set(anchored)), 24)

    def test_file_group_deduplication(self):
        candidates, evidence = self.fixtures()
        candidates[1].path = candidates[0].path
        candidates[3].path = candidates[0].path
        selection = filt.select_indices(candidates, evidence, 'unique_file_first', 24)
        self.assertEqual(len(selection), 24)
        self.assertEqual(len(set(candidates[i].path for i in selection)), 24)
        self.assertNotIn(1, selection)
        self.assertNotIn(3, selection)

    def test_known_policies_deterministic_and_gold_blind(self):
        candidates, evidence = self.fixtures()
        for policy in filt.STRATEGIES:
            one = filt.select_indices(candidates, evidence, policy)
            two = filt.select_indices(candidates, evidence, policy)
            self.assertEqual(one, two)
            self.assertEqual(len(one), 24)
            self.assertEqual(len(set(one)), 24)
            self.assertTrue(all(0 <= i < len(candidates) for i in one))


if __name__ == '__main__':
    unittest.main()
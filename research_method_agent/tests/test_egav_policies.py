"""Synthetic software tests only, NOT real Defects4J benchmark improvements."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egav_policies import (observed_failure_anchor, ranked_full_targets,
                           evidence_sufficient, file_rank_by_candidate, boundary_weight)


def c(cid, f, prior=0.5):
    return {'id': cid, 'file': f, 'prior': prior}


def risk(candidate, seen, graph=False):
    return seen[candidate['id']]['uncertainty']


class TestEGAVPolicy(unittest.TestCase):
    def test_file_boundary_is_distinct_file_not_method_count(self):
        cs = [c('a', 'A.java'), c('b', 'A.java'), c('c', 'B.java')]
        rs = file_rank_by_candidate(cs, {'a': .7, 'b': .8, 'c': .75})
        self.assertEqual(rs, {'a': 1, 'b': 1, 'c': 2})
        self.assertEqual(boundary_weight(5), 1.)
        self.assertLess(boundary_weight(1), boundary_weight(5))

    def test_failure_anchor_matches_exact_filename_or_method_call_only(self):
        candidate = c('src/Foo.java::Foo.writeValue()', 'src/Foo.java')
        self.assertEqual(observed_failure_anchor(candidate, 'Exception at Foo.java:12'), 1.)
        self.assertEqual(observed_failure_anchor(candidate, 'java.lang.Exception at writeValue(x)'), .75)
        self.assertEqual(observed_failure_anchor(candidate, 'something about another class'), 0.)
        self.assertEqual(observed_failure_anchor(candidate, ''), 0.)

    def test_failure_signal_changes_full_target_without_future_full_scores(self):
        cs = [c('a', 'A.java'), c('b', 'B.java')]
        seen = {'a': {'uncertainty': .65}, 'b': {'uncertainty': .5}}
        state = {'a': .4, 'b': .4}
        self.assertEqual(ranked_full_targets(cs, state, seen, '', 'card_failure_full', risk)[0]['id'], 'a')
        self.assertEqual(ranked_full_targets(cs, state, seen, 'B.java:100', 'card_failure_full', risk)[0]['id'], 'b')

    def test_top_five_boundary_changes_target(self):
        cs = [c(f'c{i}', f'F{i}.java') for i in range(7)]
        state = {f'c{i}': 1 - i * .1 for i in range(7)}
        seen = {f'c{i}': {'uncertainty': .55 if i == 0 else .50} for i in range(7)}
        self.assertEqual(ranked_full_targets(cs, state, seen, '', 'card_rank_full', risk)[0]['id'], 'c4')

    def test_stop_requires_strong_margin_low_uncertainty_and_no_missing_dependency(self):
        cs = [c('a', 'A.java'), c('b', 'B.java')]
        state = {'a': .90, 'b': .3}
        seen = {'a': {'uncertainty': .10, 'missing_dependency': False},
                'b': {'uncertainty': .30}}
        self.assertTrue(evidence_sufficient(cs, state, seen))
        seen['a']['missing_dependency'] = True
        self.assertFalse(evidence_sufficient(cs, state, seen))
        seen['a']['missing_dependency'] = False
        seen['a']['uncertainty'] = .9
        self.assertFalse(evidence_sufficient(cs, state, seen))
        seen['a']['uncertainty'] = .1
        state['b'] = .75
        self.assertFalse(evidence_sufficient(cs, state, seen))

    def test_no_gold_or_future_full_observation_required(self):
        cs = [c('a', 'A.java'), c('b', 'B.java')]
        seen = {'a': {'uncertainty': .8, 'score': .5}, 'b': {'uncertainty': .3, 'score': .6}}
        state = {'a': .5, 'b': .6}
        choices = ranked_full_targets(cs, state, seen, 'A.java', 'card_egav', risk)
        self.assertEqual(choices[0]['id'], 'a')
        self.assertFalse(evidence_sufficient(cs, state, seen))


if __name__ == '__main__':
    unittest.main()
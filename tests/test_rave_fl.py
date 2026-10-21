"""Run with: python -m unittest discover -s tests -p 'test_rave_fl.py'"""
import unittest

from rave_fl.core import Action, Candidate, Evidence, RaveFL, features, sketch


class RaveFLTests(unittest.TestCase):
    def test_sketch_levels(self):
        c = Candidate("f", "def f():", "return g()", calls=("g",))
        self.assertNotIn("Calls:", sketch(c, 0).text)
        self.assertIn("Calls: g", sketch(c, 1).text)
        self.assertIn("return g()", sketch(c, 2).text)

    def test_feature_extraction(self):
        c = Candidate("f", "def f():", semantic_score=.8, coverage_score=.5)
        self.assertAlmostEqual(features(c, 1, .6)["ranking_gap"], .2)

    def test_budget_never_exceeded(self):
        candidates = [Candidate(str(i), "def fn():", "x" * 100, .9, .9) for i in range(5)]
        result = RaveFL(candidates, token_budget=7).execute()
        self.assertLessEqual(result["tokens_used"], 7)
        self.assertGreaterEqual(result["tokens_remaining"], 0)

    def test_verifier_and_ranking(self):
        def verify(sketches, action):
            return [
                Evidence(s.function_id, 1.0 if s.function_id == "b" else 0.0,
                         "test", 1, .1, action)
                for s in sketches
            ]

        candidates = [
            Candidate("a", "def a():", semantic_score=.1),
            Candidate("b", "def b():", semantic_score=.2),
        ]
        result = RaveFL(candidates, token_budget=100, verifier=verify, risk_threshold=1).execute()
        self.assertEqual(result["ranked_functions"][0], "b")
        self.assertIn("BATCH_SCREEN", result["actions"])


if __name__ == "__main__":
    unittest.main()

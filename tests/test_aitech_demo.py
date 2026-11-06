"""AITECH smoke tests for demo and evaluation utilities."""
import unittest

from aitech.demo import run_demo
from aitech.evidence import Evidence, select_evidence
from aitech.metrics import evaluate


class TestAITECHDemo(unittest.TestCase):
    def test_demo_compares_same_budget(self):
        cases = run_demo()["cases"]
        self.assertEqual(cases["relevance_only"]["selected_tokens"], 8)
        self.assertEqual(cases["dependency_aware"]["selected_tokens"], 8)
        self.assertFalse(cases["relevance_only"]["complete"])
        self.assertTrue(cases["dependency_aware"]["complete"])

    def test_evaluator_gold_is_optional(self):
        result = select_evidence([Evidence("x", "x", 2, 3)], {}, 2)
        self.assertIsNone(evaluate(result, raw_tokens=4, dependencies={}).gold_evidence_recall)
        metrics = evaluate(result, raw_tokens=4, dependencies={}, gold_evidence=["x", "y"])
        self.assertEqual(metrics.gold_evidence_recall, 0.5)
        self.assertEqual(metrics.compression_ratio, 0.5)
        self.assertEqual(metrics.dependency_violation_count, 0)

    def test_invalid_raw_count_rejected(self):
        result = select_evidence([], {}, 0)
        with self.assertRaises(ValueError):
            evaluate(result, raw_tokens=0, dependencies={})


if __name__ == "__main__":
    unittest.main()

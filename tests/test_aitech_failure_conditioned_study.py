"""End-to-end study invariants, source provenance and held-out split checks."""
import unittest

from aitech.failure_conditioned_study import BUG_INPUTS, experiment


class TestFailureConditionedStudy(unittest.TestCase):
    def test_all_inputs_pinned_and_available(self):
        for name, path in BUG_INPUTS.items():
            with self.subTest(bug=name):
                self.assertTrue(path.is_file())

    def test_study_budget_and_splits(self):
        result = experiment()
        self.assertEqual(result["development_bugs"], ["Lang-1", "Time-1"])
        self.assertEqual(result["evaluation_bugs"], ["Time-25"])
        self.assertFalse(result["gold_used"])
        self.assertFalse(result["llm_inference_performed"])
        self.assertEqual(len(result["cases"]), 3)
        for case in result["cases"]:
            self.assertEqual(len(case["rows"]), 16)
            self.assertEqual(len(case["raw_source_sha256"]), 64)
            for row in case["rows"]:
                self.assertGreaterEqual(row["used_proxy_tokens"], 0)
                self.assertLessEqual(row["used_proxy_tokens"],
                                     row["budget_proxy"])
                self.assertLessEqual(row["failure_anchors_kept"],
                                     row["failure_anchors_total"])

    def test_study_is_deterministic(self):
        self.assertEqual(experiment(), experiment())


if __name__ == "__main__":
    unittest.main()

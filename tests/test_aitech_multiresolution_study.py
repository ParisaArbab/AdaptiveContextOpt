"""Real-archive multi-resolution study acceptance tests."""
import unittest

from aitech.multiresolution_study import experiment


class TestMultiResolutionStudy(unittest.TestCase):
    def test_three_bug_experiment_is_budget_safe(self):
        result = experiment()
        self.assertEqual(len(result["cases"]), 3)
        self.assertTrue(result["evaluation_not_clean_unseen"])
        self.assertFalse(result["gold_used"])
        self.assertFalse(result["llm_ran"])
        for case in result["cases"]:
            self.assertEqual(len(case["rows"]), 4)
            for row in case["rows"]:
                self.assertLessEqual(row["single_l1_tokens"], row["budget_proxy"])
                self.assertLessEqual(row["tiered_tokens"], row["budget_proxy"])
                self.assertLessEqual(row["tiered_anchor_l1"],
                                     row["tiered_anchor_identities"])
                self.assertLessEqual(row["tiered_anchor_identities"],
                                     row["tiered_anchor_total"])

    def test_repeated_experiment_is_identical(self):
        self.assertEqual(experiment(), experiment())


if __name__ == "__main__":
    unittest.main()

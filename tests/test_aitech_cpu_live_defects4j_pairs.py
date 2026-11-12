"""Real Defects4J paired counterfactual scoring safety checks."""
import unittest

from aitech.cpu_live_defects4j_pairs import (
    _predicted_class, score_file_outcomes
)


class TestCpuLiveDefects4jPairs(unittest.TestCase):
    def test_file_outcome_is_scored_only_posthoc(self):
        rows = [{
            "bug": "Math-1", "skip_reason": None,
            "predictions": [
                {"condition": "STOP", "predicted_method": "org.math.B.foo()",
                 "actual_total_input_tokens": 400,
                 "generation_seconds": 1.0},
                {"condition": "UPGRADE_L1",
                 "predicted_method": "org.math.A.bar()",
                 "actual_total_input_tokens": 425,
                 "generation_seconds": 1.2},
            ],
        }]
        outcomes = score_file_outcomes(rows, {"Math-1": {"org.math.A"}})
        self.assertEqual(outcomes[0]["outcome"], 1)
        self.assertFalse(outcomes[0]["stop_file_hit"])
        self.assertTrue(outcomes[0]["upgrade_file_hit"])

    def test_invalid_predictions_do_not_match_gold(self):
        self.assertIsNone(_predicted_class(None))
        self.assertEqual(_predicted_class("org.math.A.bar(int)"), "org.math.A")
        data = [{
            "bug": "Chart-1", "skip_reason": None,
            "predictions": [
                {"condition": "STOP", "predicted_method": None,
                 "actual_total_input_tokens": 90,
                 "generation_seconds": 1.0},
                {"condition": "UPGRADE_L1",
                 "predicted_method": "org.chart.A.foo()",
                 "actual_total_input_tokens": 100,
                 "generation_seconds": 2.0},
            ],
        }]
        self.assertEqual(score_file_outcomes(
            data, {"Chart-1": {"org.chart.A"}})[0]["outcome"], 1)

    def test_skipped_pair_remains_explicit(self):
        data = [{
            "bug": "Math-1", "skip_reason": "no_upgrade", "predictions": []
        }]
        outcome = score_file_outcomes(data, {"Math-1": {"org.math.A"}})
        self.assertTrue(outcome[0]["skipped"])

    def test_pair_order_enforced(self):
        bad = [{
            "bug": "Math-1", "skip_reason": None,
            "predictions": [
                {"condition": "UPGRADE_L1"},
                {"condition": "STOP"},
            ],
        }]
        with self.assertRaisesRegex(ValueError, "mismatched"):
            score_file_outcomes(bad, {"Math-1": set()})


if __name__ == "__main__":
    unittest.main()

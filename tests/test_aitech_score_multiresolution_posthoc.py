"""Posthoc evidence scoring must never conflate L0 identity with L1 code."""
import unittest

from aitech.score_multiresolution_posthoc import score


class TestScoreMultiresolutionPosthoc(unittest.TestCase):
    def test_identity_gain_can_coexist_with_detail_loss(self):
        frozen = {
            "gold_used": False,
            "cases": [{
                "bug": "Fixture-1", "split": "development",
                "rows": [{
                    "budget_proxy": 100,
                    "single_selected_ids": ["pkg.A.bug()"],
                    "tiered_l0_ids": ["pkg.A.bug()"],
                    "tiered_l1_ids": ["pkg.A.other()"],
                    "single_l1_tokens": 75,
                    "tiered_tokens": 80,
                }],
            }],
        }
        labels = {"scoring_only": True, "bugs": {
            "Fixture-1": {"changed_methods": ["pkg.A.bug()"]}
        }}
        result = score(frozen, labels)
        row = result["rows"][0]
        self.assertEqual(row["tiered_identity_hits"], 1)
        self.assertEqual(row["tiered_l1_detail_hits"], 0)
        self.assertEqual(row["baseline_l1_hits"], 1)
        self.assertEqual(result["detail_loss_conditions"], 1)
        self.assertEqual(result["identity_gain_conditions"], 0)

    def test_gold_use_for_selection_rejected(self):
        with self.assertRaises(ValueError):
            score({"gold_used": True}, {"scoring_only": True})

    def test_overlapping_tiers_rejected(self):
        frozen = {"gold_used": False, "cases": [{
            "bug": "Fixture-1", "split": "development",
            "rows": [{
                "budget_proxy": 100,
                "single_selected_ids": [],
                "tiered_l0_ids": ["pkg.A.bug()"],
                "tiered_l1_ids": ["pkg.A.bug()"],
                "single_l1_tokens": 0, "tiered_tokens": 80,
            }],
        }]}
        labels = {"scoring_only": True, "bugs": {
            "Fixture-1": {"changed_methods": ["pkg.A.bug()"]}
        }}
        with self.assertRaises(ValueError):
            score(frozen, labels)


if __name__ == "__main__":
    unittest.main()

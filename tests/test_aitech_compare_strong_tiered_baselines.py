"""Strong-baseline comparison must not reward a weaker novel-looking policy."""
import unittest

from aitech.compare_strong_tiered_baselines import compare


class TestStrongTieredBaselines(unittest.TestCase):
    def test_rejects_frozen_selection_with_gold(self):
        with self.assertRaises(ValueError):
            compare({"gold_used": True}, {"cases": []}, {"scoring_only": True})

    def test_existing_stronger_detail_is_reported_as_loss(self):
        frozen = {"gold_used": False, "cases": [{
            "bug": "Demo-1",
            "rows": [{
                "budget_proxy": 100,
                "tiered_l0_ids": ["A.bug()"],
                "tiered_l1_ids": [],
                "tiered_tokens": 60,
            }]
        }]}
        existing = {"cases": [{
            "bug": "Demo-1",
            "rows": [
                {"budget_proxy": 100, "method": policy,
                 "candidate_id_tier": {"A.bug()": "L1"},
                 "used_proxy_tokens": 100}
                for policy in (
                    "l1_only", "tiered_upgrade",
                    "anchor_detail_first", "anchor_rank_detail"
                )
            ]
        }]}
        labels = {"scoring_only": True, "bugs": {
            "Demo-1": {"changed_methods": ["A.bug()"]}
        }}
        scored = compare(frozen, existing, labels)
        self.assertEqual(scored["new_detail_losses_to_best_existing"], 1)
        self.assertEqual(scored["new_detail_wins_over_best_existing"], 0)


if __name__ == "__main__":
    unittest.main()

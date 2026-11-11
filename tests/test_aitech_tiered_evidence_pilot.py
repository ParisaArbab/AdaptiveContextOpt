"""Tiered evidence: distinguish method ID coverage from detailed source."""
import unittest
from aitech.tiered_evidence_pilot import select_tiers, experiment


class TestTieredEvidence(unittest.TestCase):
    def snapshot(self):
        return {"context": "failure in big()", "candidates": [
            {"function_id":"pkg.X.big()", "features":{"l0_tokens":10,"l1_tokens":100}},
            {"function_id":"pkg.X.small()", "features":{"l0_tokens":10,"l1_tokens":20}},
            {"function_id":"pkg.X.other()", "features":{"l0_tokens":10,"l1_tokens":20}},
        ]}

    def test_l0_presence_is_not_l1_source(self):
        raw=select_tiers(self.snapshot(), 40, "l1_only")
        tier=select_tiers(self.snapshot(), 40, "tiered_l0")
        self.assertEqual(raw["candidate_id_tier"], {"pkg.X.small()":"L1","pkg.X.other()":"L1"})
        self.assertEqual(tier["candidate_id_tier"]["pkg.X.big()"],"L0")
        self.assertEqual(tier["anchors_at_any_level"],1)
        self.assertEqual(tier["anchors_with_l1_details"],0)

    def test_upgrade_only_if_budget_allows(self):
        case=select_tiers(self.snapshot(), 100, "tiered_upgrade")
        self.assertLessEqual(case["used_proxy_tokens"],100)
        self.assertEqual(case["anchors_at_any_level"],1)
        self.assertEqual(case["anchors_with_l1_details"],0)
        self.assertEqual(case["candidate_id_tier"]["pkg.X.small()"],"L1")

    def test_anchor_detail_before_optional_identifiers(self):
        result=select_tiers(self.snapshot(),120,"anchor_detail_first")
        self.assertEqual(result["candidate_id_tier"]["pkg.X.big()"],"L1")
        self.assertEqual(result["anchors_with_l1_details"],1)
        self.assertLessEqual(result["used_proxy_tokens"],120)

    def test_rejects_invalid_costs(self):
        s=self.snapshot()
        s["candidates"][0]["features"]["l1_tokens"]=3
        with self.assertRaises(ValueError):
            select_tiers(s,100,"tiered_upgrade")
        with self.assertRaises(ValueError):
            select_tiers(self.snapshot(),-1,"tiered_l0")

    def test_archived_bugs_respect_all_budgets(self):
        result=experiment()
        self.assertEqual(len(result["cases"]),3)
        for case in result["cases"]:
            self.assertEqual(len(case["rows"]),16)
            for row in case["rows"]:
                self.assertLessEqual(row["used_proxy_tokens"],row["budget_proxy"])
                self.assertLessEqual(row["anchors_with_l1_details"],
                                     row["anchors_at_any_level"])


if __name__=="__main__":
    unittest.main()

"""Regression tests for Pareto separation of L0 identifiers and L1 source."""
import unittest
from aitech.tiered_pareto_analysis import analyze, render


class TestTieredPareto(unittest.TestCase):
    def example(self):
        rows=[]
        for method,identity,detail in (
            ("l1_only",.5,.5),
            ("tiered_l0",1.,0.),
            ("tiered_upgrade",1.,0.),
            ("anchor_detail_first",1.,.5),
            ("anchor_rank_detail",1.,.5),
        ):
            rows.append({"bug":"Toy-1","budget_proxy":200,"method":method,
                         "identity_recall":identity,"detailed_source_recall":detail})
        return {"rows":rows}

    def test_does_not_reward_identity_gain_that_loses_details(self):
        summary=analyze(self.example())
        self.assertFalse(next(r for r in summary["rows"]
                              if r["method"]=="tiered_l0")["weakly_pareto_dominates_l1_only"])
        self.assertTrue(next(r for r in summary["rows"]
                             if r["method"]=="anchor_rank_detail")["strictly_pareto_improves_l1_only"])
        self.assertEqual(summary["method_counts"]["anchor_rank_detail"]["strict_improvement"],1)
        self.assertIn("NOT a measured debugging accuracy gain",render(summary))

    def test_missing_policy_is_rejected(self):
        example=self.example()
        example["rows"].pop()
        with self.assertRaisesRegex(ValueError,"missing policy"):
            analyze(example)

    def test_duplicate_policy_is_rejected(self):
        example=self.example()
        example["rows"].append(dict(example["rows"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            analyze(example)


if __name__=="__main__":
    unittest.main()

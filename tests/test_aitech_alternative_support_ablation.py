"""Guard against overstating the alternative support controlled experiment."""
import unittest

from aitech.alternative_support_ablation import run


class TestAlternativeSupportAblation(unittest.TestCase):
    def test_conditions_are_matched_and_provenance_is_synthetic(self):
        outcome = run()
        self.assertFalse(outcome["llm_executed"])
        self.assertFalse(outcome["fault_localization_accuracy_measured"])
        rows = outcome["results"]
        self.assertEqual(len(rows), 15)
        for budget in outcome["budgets"]:
            subset = [r for r in rows if r["budget"] == budget]
            self.assertEqual({r["method"] for r in subset}, set(outcome["methods"]))
            self.assertTrue(all(r["spent_proxy_tokens"] <= budget for r in subset))

    def test_or_support_restores_failure_at_small_budget(self):
        rows = [r for r in run()["results"] if r["budget"] == 8]
        methods = {r["method"]: r for r in rows}
        self.assertTrue(methods["one_sufficient_path_exact"]["failure_retained"])
        self.assertFalse(methods["one_sufficient_path_exact"]["sufficient_support_missing"])
        self.assertFalse(methods["all_edges_required"]["failure_retained"])
        self.assertEqual(
            methods["relevance_only"]["sufficient_support_missing"],
            ["failed_assertion"],
        )


if __name__ == "__main__":
    unittest.main()

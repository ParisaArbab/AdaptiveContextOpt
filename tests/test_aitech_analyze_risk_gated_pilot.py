"""Paired research report tests, independent of LLM or gold patch labels."""
import unittest

from aitech.analyze_risk_gated_pilot import analyze, report


class TestAnalyzeRiskGatedPilot(unittest.TestCase):
    def fixture(self):
        rows = []
        for method, ids, used, promotions in (
            ("rank_fill", ["a"], 20, 0),
            ("anchor_first", ["a"], 20, 0),
            ("always_expand", ["a", "b"], 40, 0),
            ("risk_gated", ["a", "b"], 40, 1),
        ):
            rows.append({
                "method": method, "budget_proxy": 50, "selected_ids": ids,
                "used_proxy_tokens": used, "graph_promoted_choices": promotions,
                "missing_arity_sketch_edges": 0, "failure_name_anchors_kept": 1,
            })
        return {"cases": [{"bug": "Test-1", "arity_sketch_edges": 1, "rows": rows}]}

    def test_reports_actual_difference_not_accuracy(self):
        result = analyze(self.fixture())
        self.assertEqual(result["context_changed_conditions"], 1)
        self.assertEqual(result["graph_promoted_choices"], 1)
        self.assertEqual(result["anchor_gains_vs_anchor_first"], 0)
        self.assertIn("No LLM localization", report(result))

    def test_missing_method_rejected(self):
        data = self.fixture()
        data["cases"][0]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing policy"):
            analyze(data)

    def test_over_budget_rejected(self):
        data = self.fixture()
        data["cases"][0]["rows"][-1]["used_proxy_tokens"] = 60
        with self.assertRaisesRegex(ValueError, "exceeded"):
            analyze(data)


if __name__ == "__main__":
    unittest.main()

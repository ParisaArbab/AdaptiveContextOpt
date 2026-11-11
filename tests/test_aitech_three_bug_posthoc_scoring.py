"""Test posthoc changed-method scoring without using labels in selection."""
import unittest

from aitech.three_bug_posthoc_scoring import score


class TestThreeBugPosthoc(unittest.TestCase):
    def data(self):
        selection = {"cases": [{
            "bug": "Toy-1",
            "rows": [
                {"bug": "Toy-1", "budget_proxy": 100, "method": "anchor_first",
                 "selected_ids": ["A.good"], "used_proxy_tokens": 70},
                {"bug": "Toy-1", "budget_proxy": 100, "method": "risk_gated",
                 "selected_ids": ["A.bad"], "used_proxy_tokens": 80},
            ],
        }]}
        labels = {"scoring_only": True,
                  "selection_frozen_before_labels": True,
                  "bugs": {"Toy-1": {"patch": "official_patch",
                                     "changed_methods": ["A.good"]}}}
        return selection, labels

    def test_detects_real_regression_after_selection(self):
        selections, labels = self.data()
        result = score(selections, labels)
        self.assertEqual(result["risk_worse_than_anchor"], 1)
        self.assertEqual(result["paired"][0]["risk_minus_anchor_changed_method_recall"], -1)

    def test_rejects_nonfrozen_gold(self):
        selections, labels = self.data()
        labels["selection_frozen_before_labels"] = False
        with self.assertRaises(ValueError):
            score(selections, labels)

    def test_never_mutates_frozen_selected_ids(self):
        selections, labels = self.data()
        before = [list(r["selected_ids"]) for r in selections["cases"][0]["rows"]]
        score(selections, labels)
        self.assertEqual(before, [r["selected_ids"] for r in selections["cases"][0]["rows"]])


if __name__ == "__main__":
    unittest.main()

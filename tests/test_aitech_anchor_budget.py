"""Unit tests for the evidence anchor protection and explicit edge audit."""
import unittest

from aitech.anchor_budget import anchor_first_selection, disambiguate_lexical_edges


class TestAnchorBudget(unittest.TestCase):
    def test_anchor_kept_even_if_full_dependency_bundle_does_not_fit(self):
        result = anchor_first_selection(
            [75, 80, 45], anchors=[0], edges={(0, 1)}, budget=120)
        self.assertIn(0, result.selected)
        self.assertNotIn(1, result.selected)
        self.assertIn((0, 1), result.unsatisfied_edges)
        self.assertLessEqual(result.tokens, 120)

    def test_adds_support_when_affordable(self):
        result = anchor_first_selection(
            [20, 30, 60], anchors=[0], edges={(0, 1)}, budget=50)
        self.assertEqual(result.selected, (0, 1))
        self.assertFalse(result.unsatisfied_edges)
        self.assertEqual(result.tokens, 50)

    def test_priority_over_ranked_non_anchor(self):
        result = anchor_first_selection([90, 40, 25], [1], set(), 90)
        self.assertIn(1, result.selected)
        self.assertNotIn(0, result.selected)

    def test_overloaded_callee_names_are_rejected(self):
        ids = [
            "pkg.Num.createNumber(String)",
            "pkg.Num.createInteger(String)",
            "pkg.Num.createInteger(Object)",
        ]
        clean, ambiguous = disambiguate_lexical_edges(
            ids, {(0, 1), (0, 2), (1, 0)})
        self.assertEqual(clean, {(1, 0)})
        self.assertEqual(ambiguous, {(0, 1), (0, 2)})

    def test_invalid_anchor_and_cost(self):
        with self.assertRaises(ValueError):
            anchor_first_selection([1], anchors=[2], edges=[], budget=3)
        with self.assertRaises(ValueError):
            anchor_first_selection([0], anchors=[0], edges=[], budget=3)
        with self.assertRaises(ValueError):
            anchor_first_selection([1], anchors=[], edges=[], budget=-1)


if __name__ == "__main__":
    unittest.main()

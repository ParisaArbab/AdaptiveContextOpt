"""Tests for the AITECH evidence-selection baseline.

Run: python -m unittest discover -s tests -p 'test_aitech_*.py' -v
"""
import unittest

from aitech.evidence import (
    Evidence, dependency_closure, missing_dependencies, select_evidence,
)


class TestDependencySelection(unittest.TestCase):
    def setUp(self):
        # Failure looks relevant, but cannot be interpreted without its bridge.
        self.items = [
            Evidence("failure", "AssertionError in test_parse", 4, 10),
            Evidence("bridge", "test_parse calls parse_expression", 4, 1),
            Evidence("noise", "unrelated warning", 4, 8),
        ]
        self.links = {"failure": ("bridge",)}

    def test_relevance_only_may_dangle(self):
        result = select_evidence(self.items, self.links, budget=8, preserve_dependencies=False)
        self.assertEqual(result.ids, ("failure", "noise"))
        self.assertEqual(result.dangling, ("failure",))

    def test_dependency_constraint_prevents_dangling(self):
        result = select_evidence(self.items, self.links, budget=8)
        self.assertEqual(result.ids, ("failure", "bridge"))
        self.assertEqual(result.dangling, ())
        self.assertEqual(result.tokens, 8)

    def test_insufficient_budget_skips_incomplete_high_value_pair(self):
        result = select_evidence(self.items, self.links, budget=4)
        self.assertEqual(result.ids, ("noise",))
        self.assertEqual(result.dangling, ())

    def test_transitive_closure(self):
        chain = {"error": ("function",), "function": ("source",)}
        self.assertEqual(dependency_closure(("error",), chain), {"error", "function", "source"})

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):
            dependency_closure(("a",), {"a": ("b",), "b": ("a",)})

    def test_unknown_support_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            select_evidence([Evidence("x", "x", 1, 1)], {"x": ("not_found",)}, 2)

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_evidence([Evidence("x", "a", 1), Evidence("x", "b", 1)], {}, 2)

    def test_zero_and_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            select_evidence([Evidence("x", "a", 0)], {}, 1)
        with self.assertRaises(ValueError):
            select_evidence([], {}, -1)

    def test_missing_dependencies_reports_violations(self):
        self.assertEqual(missing_dependencies(["failure"], self.links), ("failure",))


if __name__ == "__main__":
    unittest.main()

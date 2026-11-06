"""Tests for the saved SymPy trace extraction and its diagnostic microcase."""
import unittest

from aitech.sympy_trace_pilot import ANCHORS, extract, run_pilot

# Verbatim source lines from previously recorded SymPy Agent4SR tool outputs.
# This excerpt does not reproduce the original agent run.
TRACE_EXCERPT = "\n".join(
    f"    {i}: {line}" for i, (_, line, _) in enumerate(ANCHORS, start=1)
)


class TestRealTracePilot(unittest.TestCase):
    def test_all_extracted_lines_are_real_anchors(self):
        extracted = extract(TRACE_EXCERPT)
        self.assertEqual(len(extracted), 7)
        self.assertEqual(extracted[0].id, "test_assertion")

    def test_missing_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing trace anchor"):
            extract("not an archived tool trace")

    def test_equal_budget_dependency_safety(self):
        result = run_pilot(TRACE_EXCERPT)
        a = result["conditions"]["relevance_only"]
        b = result["conditions"]["dependency_aware"]
        self.assertEqual(result["budget_tokens_proxy"], 36)
        self.assertLessEqual(a["tokens_proxy"], 36)
        self.assertLessEqual(b["tokens_proxy"], 36)
        self.assertEqual(a["known_dependency_violations"], 1)
        self.assertEqual(b["known_dependency_violations"], 0)
        self.assertEqual(b["tokens_proxy"], a["tokens_proxy"])
        self.assertIn("printable_import", b["selected"])
        self.assertNotIn("printable_import", a["selected"])

    def test_tight_budget_can_drop_important_evidence(self):
        result = run_pilot(TRACE_EXCERPT, budget=31)
        b = result["conditions"]["dependency_aware"]
        self.assertNotIn("test_assertion", b["selected"])
        self.assertEqual(b["known_dependency_violations"], 0)


if __name__ == "__main__":
    unittest.main()

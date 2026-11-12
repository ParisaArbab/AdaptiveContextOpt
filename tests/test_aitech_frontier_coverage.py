"""Tests for source-provenance-aware cross-class frontier coverage."""
import unittest

from aitech.frontier_coverage import inspect_frontier


def item(signature, calls):
    return {"function_id": signature, "levels": [
        "", "\n".join("CALL " + x for x in calls)]}


class TestFrontierCoverage(unittest.TestCase):
    def test_qualified_cross_class_and_missing_method(self):
        candidates = [
            item("pkg.NumberUtils.use(String)", [
                "StringUtils.isBlank(value)",
                "StringUtils.isEmpty(value)",
                "UnseenClass.build(value)",
                "value.trim()",
            ]),
            item("pkg.StringUtils.isBlank(String)", []),
        ]
        out = inspect_frontier(candidates)
        self.assertEqual(out.in_pool, ((0, 1),))
        self.assertIn("StringUtils.isEmpty/1", out.missing_ranked_method)
        self.assertIn("UnseenClass.build/1", out.outside_ranked_owner)
        self.assertEqual(out.qualified_calls, 3)

    def test_same_simple_name_in_two_packages_abstains(self):
        candidates = [
            item("root.Caller.run()", ["Date.parse(value)"]),
            item("a.Date.parse(String)", []),
            item("b.Date.parse(String)", []),
        ]
        out = inspect_frontier(candidates)
        self.assertEqual(out.in_pool, ())
        self.assertIn("Date.parse/1:owner_collision", out.ambiguous)

    def test_same_arity_overload_abstains(self):
        candidates = [
            item("pkg.Caller.run()", ["Utils.parse(value)"]),
            item("pkg.Utils.parse(String)", []),
            item("pkg.Utils.parse(Object)", []),
        ]
        out = inspect_frontier(candidates)
        self.assertEqual(out.in_pool, ())
        self.assertIn("Utils.parse/1:overloaded_target", out.ambiguous)

    def test_nested_call_expression_records_first_receiver(self):
        candidates = [
            item("pkg.Caller.run()", ["DateTimeUtils.getChronology(now).withUTC()"]),
            item("pkg.DateTimeUtils.other()", []),
        ]
        out = inspect_frontier(candidates)
        self.assertIn("DateTimeUtils.getChronology/1", out.missing_ranked_method)


if __name__ == "__main__":
    unittest.main()

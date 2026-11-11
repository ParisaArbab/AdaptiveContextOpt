"""Oracle ceiling is scoring-only, never an inference policy."""
import unittest

from aitech.changed_method_oracle_ceiling import oracle_ceiling


def candidate(name, cost):
    return {"function_id": name, "features": {"l1_tokens": cost}}


class TestChangedMethodOracleCeiling(unittest.TestCase):
    def test_cheapest_gold_first_is_maximum_cardinality(self):
        cases = [
            candidate("A.expensive()", 300),
            candidate("A.cheap()", 50),
            candidate("A.middle()", 80),
            candidate("A.distractor()", 1),
        ]
        gold = {"A.expensive()", "A.cheap()", "A.middle()"}
        result = oracle_ceiling(cases, gold, 140)
        self.assertEqual(result["oracle_gold_l1_hits"], 2)
        self.assertEqual(result["oracle_gold_ids"],
                         ["A.cheap()", "A.middle()"])

    def test_missing_gold_is_not_counted_as_retrievable(self):
        result = oracle_ceiling([candidate("A.other()", 10)],
                                {"A.missing()"}, 100)
        self.assertEqual(result["gold_present_in_pool"], 0)
        self.assertEqual(result["oracle_gold_l1_hits"], 0)
        self.assertEqual(result["missing_from_pool"], ["A.missing()"])

    def test_no_partial_method_allowed(self):
        result = oracle_ceiling([candidate("A.fail()", 101)],
                                {"A.fail()"}, 100)
        self.assertEqual(result["oracle_gold_l1_hits"], 0)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            oracle_ceiling([], set(), -1)


if __name__ == "__main__":
    unittest.main()

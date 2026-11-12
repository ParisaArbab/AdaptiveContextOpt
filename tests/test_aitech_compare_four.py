"""Tests for joining previously measured A/B/C with new provenance D."""
import unittest

from aitech.compare_four import compare, BASE, EXTRA


def fake_result(conditions):
    return {
        "instance": "sympy__sympy-20590",
        "model": "test-model",
        "seed": 42,
        "temperature": 0,
        "budget_proxy": 36,
        "raw_proxy_tokens": 49,
        "conditions": conditions,
    }


def arm(ids):
    return {
        "selected_ids": ids,
        "prediction": {"file": "file.py", "function": "Function"},
        "offline_score": {"file_hit_1": False, "entity_hit_1": False},
        "prompt_eval_count": 100,
        "eval_count": 20,
        "elapsed_seconds": 2.5,
        "tokens_proxy": 35,
        "dangling_ids": [],
        "prompt_chars": 200,
    }


class TestCompareFour(unittest.TestCase):
    def setUp(self):
        self.previous = fake_result({k: arm(["a", "b"]) for k in BASE})
        self.new = fake_result({EXTRA: arm(["a", "b"])})

    def test_comparison_has_four_rows(self):
        combined = compare(self.previous, self.new)
        self.assertEqual(len(combined["summary"]), 4)
        self.assertIn("NOT total-token matched", combined["warning"])
        self.assertEqual(list(combined["conditions"]), [*BASE, EXTRA])

    def test_rejects_mismatched_budget(self):
        self.new["budget_proxy"] = 35
        with self.assertRaisesRegex(ValueError, "budget_proxy"):
            compare(self.previous, self.new)

    def test_rejects_changed_evidence_selection(self):
        self.new["conditions"][EXTRA]["selected_ids"] = ["x"]
        with self.assertRaisesRegex(ValueError, "selected evidence differs"):
            compare(self.previous, self.new)


if __name__ == "__main__":
    unittest.main()

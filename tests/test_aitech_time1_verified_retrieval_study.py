"""Frozen real historical Time-1 source retrieval experiment tests."""
import json
import unittest

from aitech.time1_verified_retrieval_study import (
    SNAPSHOT, SOURCE_RECORD, run, token_proxy,
)


class TestVerifiedTime1Retrieval(unittest.TestCase):
    def fixture(self):
        return (json.loads(SNAPSHOT.read_text(encoding="utf-8")),
                json.loads(SOURCE_RECORD.read_text(encoding="utf-8")))

    def test_pinned_source_and_identical_token_proxy(self):
        data, verified = self.fixture()
        self.assertEqual(verified["buggy_revision"],
                         "8612f9e5b88c1bea933ef9ab1e431f5db3006b48")
        self.assertEqual(verified["start_line"], 283)
        self.assertEqual(token_proxy(verified["method_source"]), 30)
        result = run(data, verified)
        self.assertFalse(result["llm_executed"])
        self.assertFalse(result["gold_patch_used"])
        self.assertEqual(len(result["rows"]), 15)
        for row in result["rows"]:
            self.assertLessEqual(row["selected_regex_proxy"],
                                 row["budget_regex_proxy"])

    def test_helper_added_only_if_it_fits_after_anchors(self):
        data, verified = self.fixture()
        result = run(data, verified, (600,))
        rows = {r["condition"]: r for r in result["rows"]}
        self.assertFalse(rows["anchor_first"]["retrieved_helper_included"])
        self.assertTrue(rows["force_verified_source"]["retrieved_helper_included"])
        self.assertFalse(rows["gate_unverified_binding"]["retrieved_helper_included"])
        self.assertEqual(rows["force_verified_source"]["failure_name_anchors_kept"], 2)

    def test_never_accepts_unverified_source(self):
        data, verified = self.fixture()
        verified["source_declaration_verified"] = False
        with self.assertRaises(ValueError):
            run(data, verified)

    def test_never_silently_claims_binding(self):
        data, verified = self.fixture()
        verified["compiler_binding_verified"] = True
        with self.assertRaises(ValueError):
            run(data, verified)


if __name__ == "__main__":
    unittest.main()

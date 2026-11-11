"""No-network acceptance checks for real bug research output quality."""
import unittest

from aitech.validate_real_bug_graph import validate


def valid_record():
    return {
        "status": "compiler_attributed_static_targets",
        "gold_patch_used": False,
        "source_hashes": {"Source.java": "a" * 64},
        "resolved_static_calls": 3,
        "unresolved_calls": 0,
        "candidate_mapping": {
            "candidate_source_sha256": "b" * 64,
            "mapped_calls": 2,
            "ambiguous_short_type_identity_count": 0,
            "selection_comparison": [
                {"method": m, "budget_proxy": 100,
                 "selected_proxy_tokens": 90,
                 "failure_name_anchors_kept": 1,
                 "failure_name_anchors_total": 2}
                for m in ("rank_fill", "anchor_first", "compiler_graph_first")
            ],
        },
    }


class TestRealBugGraphValidation(unittest.TestCase):
    def test_valid_record_is_eligible_for_evidence_analysis(self):
        summary = validate(valid_record())
        self.assertTrue(summary["valid_for_downstream_evidence_analysis"])
        self.assertEqual(summary["comparison_rows"], 3)

    def test_missing_provenance_and_baselines_rejected(self):
        record = valid_record()
        record.pop("source_hashes")
        record["candidate_mapping"]["selection_comparison"].pop()
        summary = validate(record)
        self.assertFalse(summary["valid_for_downstream_evidence_analysis"])
        self.assertTrue(any("baseline" in reason for reason in summary["reasons"]))

    def test_overbudget_and_gold_misuse_rejected(self):
        record = valid_record()
        record["gold_patch_used"] = True
        record["candidate_mapping"]["selection_comparison"][0]["selected_proxy_tokens"] = 101
        summary = validate(record)
        self.assertFalse(summary["valid_for_downstream_evidence_analysis"])
        self.assertTrue(any("budget" in reason for reason in summary["reasons"]))
        self.assertTrue(any("gold" in reason for reason in summary["reasons"]))


if __name__ == "__main__":
    unittest.main()

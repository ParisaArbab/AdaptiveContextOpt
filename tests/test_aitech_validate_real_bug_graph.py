"""CPU-only scientific acceptance tests for compiler graph experiments."""
import unittest

from aitech.validate_real_bug_graph import validate


def valid_record():
    ids = ["pkg.A.f()", "pkg.A.help()"]
    return {
        "status": "compiler_attributed_static_targets",
        "gold_patch_used": False,
        "source_hashes": {"Source.java": "a" * 64},
        "resolved_static_calls": 3,
        "unresolved_calls": 0,
        "candidate_mapping": {
            "candidate_source_sha256": "b" * 64,
            "candidate_count": len(ids),
            "edges": [[0, 1]],
            "mapped_calls": 1,
            "ambiguous_short_type_identity_count": 0,
            "selection_comparison": [
                {"method": method, "budget_proxy": 100,
                 "selected_proxy_tokens": 90,
                 "selected_ids": ids,
                 "failure_name_anchors_kept": 1,
                 "failure_name_anchors_total": 2}
                for method in ("rank_fill", "anchor_first", "compiler_graph_first")
            ],
        },
    }


class TestRealBugGraphValidation(unittest.TestCase):
    def test_structural_pass_does_not_imply_graph_changed_context(self):
        result = validate(valid_record())
        self.assertTrue(result["valid_for_downstream_evidence_analysis"])
        self.assertTrue(result["graph_available_for_comparison"])
        self.assertFalse(result["graph_changed_selected_evidence"])
        self.assertIn("did not change", result["graph_effect_interpretation"])

    def test_graph_changes_context_only_for_distinct_selected_ids(self):
        record = valid_record()
        record["candidate_mapping"]["selection_comparison"][-1]["selected_ids"] = ["pkg.A.f()"]
        result = validate(record)
        self.assertTrue(result["graph_changed_selected_evidence"])

    def test_zero_mapped_edges_is_not_graph_success(self):
        record = valid_record()
        record["candidate_mapping"]["edges"] = []
        record["candidate_mapping"]["mapped_calls"] = 0
        result = validate(record)
        self.assertTrue(result["valid_for_downstream_evidence_analysis"])
        self.assertFalse(result["graph_available_for_comparison"])
        self.assertFalse(result["graph_changed_selected_evidence"])

    def test_missing_provenance_or_one_policy_rejected(self):
        record = valid_record()
        record.pop("source_hashes")
        record["candidate_mapping"]["selection_comparison"].pop()
        result = validate(record)
        self.assertFalse(result["valid_for_downstream_evidence_analysis"])
        self.assertTrue(any("missing method" in s for s in result["reasons"]))

    def test_gold_use_and_budget_violation_rejected(self):
        record = valid_record()
        record["gold_patch_used"] = True
        record["candidate_mapping"]["selection_comparison"][0]["selected_proxy_tokens"] = 101
        result = validate(record)
        self.assertFalse(result["valid_for_downstream_evidence_analysis"])
        self.assertTrue(any("budget" in s for s in result["reasons"]))

    def test_duplicate_rows_or_outside_graph_indices_rejected(self):
        record = valid_record()
        record["candidate_mapping"]["edges"] = [[0, 3]]
        record["candidate_mapping"]["selection_comparison"].append(
            dict(record["candidate_mapping"]["selection_comparison"][0])
        )
        result = validate(record)
        self.assertFalse(result["valid_for_downstream_evidence_analysis"])
        self.assertTrue(any("edge 0" in s for s in result["reasons"]))
        self.assertTrue(any("duplicate method" in s for s in result["reasons"]))


if __name__ == "__main__":
    unittest.main()

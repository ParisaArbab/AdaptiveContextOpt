"""Acceptance report has to distinguish graph effect from localization."""
import unittest

from aitech.report_compiler_graph import render


class TestCompilerGraphReport(unittest.TestCase):
    def test_no_mapped_graph_is_not_improvement(self):
        record = {
            "status": "compiler_attributed_static_targets",
            "gold_patch_used": False,
            "source_hashes": {"Demo.java": "a" * 64},
            "resolved_static_calls": 15, "unresolved_calls": 0,
            "candidate_mapping": {
                "candidate_source_sha256": "b" * 64,
                "candidate_count": 2, "edges": [], "mapped_calls": 0,
                "ambiguous_short_type_identity_count": 0,
                "selection_comparison": [
                    {"method": method, "budget_proxy": 40,
                     "selected_proxy_tokens": 20,
                     "failure_name_anchors_kept": 1,
                     "failure_name_anchors_total": 1,
                     "unresolved_mapped_support": 0,
                     "selected_ids": ["pkg.Demo.run()"]}
                    for method in ("rank_fill", "anchor_first", "compiler_graph_first")
                ],
            },
        }
        result = render(record)
        self.assertIn("Mapped graph available:** False", result)
        self.assertIn("no compiler-resolved edges mapped", result)
        self.assertIn("not a fault-localization result", result)
        self.assertIn("| 40 | compiler_graph_first | 20 | 1/1 | 0 |", result)


if __name__ == "__main__":
    unittest.main()

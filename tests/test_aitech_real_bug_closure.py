"""Regression checks for the two real archived Defects4J inputs (no network)."""
import unittest

from aitech.real_bug_closure_stress import ROOT, BUGS, experiment


class TestRealBugClosureStress(unittest.TestCase):
    def test_pinned_inputs_exist(self):
        for bug in BUGS:
            self.assertTrue((ROOT / bug / "input_snapshot.json").is_file())

    def test_budget_and_anchor_accounting(self):
        output = experiment()
        self.assertEqual(len(output["results"]), 2)
        for case in output["results"]:
            self.assertEqual(case["source_candidate_count"], 25)
            self.assertEqual(len(case["rows"]), 16)
            for row in case["rows"]:
                self.assertLessEqual(row["used_proxy_tokens"], row["budget"])
                self.assertLessEqual(row["failure_name_anchors_kept"],
                                     row["failure_name_anchors_total"])

    def test_anchor_first_records_unresolved_links(self):
        cases = {case["bug"]: case for case in experiment()["results"]}
        time_rows = {(r["budget"], r["method"]): r
                     for r in cases["Time-1"]["rows"]}
        for budget in (200, 450, 600):
            baseline = time_rows[(budget, "closure")]
            guarded = time_rows[(budget, "anchor_first_audited")]
            self.assertGreaterEqual(guarded["failure_name_anchors_kept"],
                                    baseline["failure_name_anchors_kept"])
            self.assertLessEqual(guarded["used_proxy_tokens"], budget)
            self.assertIn("unambiguous_missing_support", guarded)

    def test_closure_can_drop_anchor(self):
        cases = {case["bug"]: case for case in experiment()["results"]}
        time_rows = {(r["budget"], r["method"]): r
                     for r in cases["Time-1"]["rows"]}
        self.assertEqual(time_rows[(200, "rank_fill")]["failure_name_anchors_kept"], 1)
        self.assertEqual(time_rows[(200, "closure")]["failure_name_anchors_kept"], 0)
        self.assertEqual(time_rows[(200, "closure")]["known_lexical_dangling"], 0)


if __name__ == "__main__":
    unittest.main()

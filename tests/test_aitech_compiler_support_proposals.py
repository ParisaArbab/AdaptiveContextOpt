"""Compiler-to-support bridge never mistakes calls for sufficient explanations."""
import unittest

from aitech.compiler_support_proposals import candidates_from_compiler
from aitech.java_compiler_graph import ResolvedCall


class TestCompilerSupportProposals(unittest.TestCase):
    def test_mapped_call_is_candidate_not_sufficient_evidence(self):
        ids = ["pkg.A.fail(String)", "pkg.A.helper(String)"]
        calls = [
            ResolvedCall(
                "A.java", "pkg.A", "fail", "METHOD",
                "pkg.A", "helper", "java.lang.String", "java.lang.String"
            )
        ]
        report = candidates_from_compiler(ids, calls)
        self.assertEqual(report["compiler_edge_count"], 1)
        self.assertEqual(len(report["candidate_proposals"]), 1)
        self.assertFalse(report["candidate_proposals"][0]["sufficiency_audited"])
        self.assertEqual(report["sufficiency_gate"]["audited_sufficient_bundles"], 0)
        self.assertFalse(report["automatic_sufficiency_promotion"])

    def test_missing_candidate_target_cannot_be_promoted(self):
        ids = ["pkg.A.fail(String)"]
        calls = [
            ResolvedCall(
                "A.java", "pkg.A", "fail", "METHOD",
                "pkg.B", "missing", "", "java.lang.String"
            )
        ]
        report = candidates_from_compiler(ids, calls)
        self.assertEqual(report["compiler_edge_count"], 0)
        self.assertEqual(report["candidate_proposals"], [])
        self.assertGreater(report["unmatched_target_count"], 0)


if __name__ == "__main__":
    unittest.main()

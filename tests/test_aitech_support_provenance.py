"""Provenance gates prevent false causal-sufficiency claims."""
import unittest

from aitech.support_provenance import (
    SupportProposal, audited_support_sets, proposal_summary
)


class TestSupportProvenance(unittest.TestCase):
    def test_compiler_binding_alone_never_certifies_sufficiency(self):
        p = SupportProposal("failure", ("callee",),
                            "javac:resolved:pkg.Foo.call", True, False)
        result = proposal_summary([p], {"failure", "callee"})
        self.assertEqual(result["candidate_proposals"], 1)
        self.assertEqual(result["audited_sufficient_bundles"], 0)
        self.assertEqual(result["audited_support_sets"], {})

    def test_manually_audited_path_is_accepted(self):
        p = SupportProposal("failure", ("callee",),
                            "javac:resolved:pkg.Foo.call", True, True,
                            "audit-review-case-001")
        result = audited_support_sets([p], {"failure", "callee"})
        self.assertEqual(result, {"failure": (("callee",),)})

    def test_unverified_binding_cannot_be_promoted(self):
        p = SupportProposal("failure", ("callee",), "lexical sketch", False, True,
                            "audit-review-case-001")
        with self.assertRaisesRegex(ValueError, "verified call binding"):
            audited_support_sets([p], {"failure", "callee"})

    def test_unaudited_support_does_not_require_false_audit_id(self):
        p = SupportProposal("failure", ("callee",), "compiler candidate", True, False)
        self.assertEqual(audited_support_sets([p], {"failure", "callee"}), {})

    def test_audited_support_needs_reference(self):
        p = SupportProposal("failure", ("callee",), "javac", True, True)
        with self.assertRaisesRegex(ValueError, "audit reference"):
            audited_support_sets([p], {"failure", "callee"})

    def test_unknown_support_not_accepted(self):
        p = SupportProposal("failure", ("missing",), "javac", True, True, "audit1")
        with self.assertRaises(ValueError):
            audited_support_sets([p], {"failure"})

    def test_two_sufficient_paths_remain_alternatives(self):
        candidates = [
            SupportProposal("failure", ("trace",), "javac:1", True, True, "audit1"),
            SupportProposal("failure", ("definition", "assertion"), "javac:2",
                            True, True, "audit2"),
        ]
        result = audited_support_sets(
            candidates, {"failure", "trace", "definition", "assertion"})
        self.assertEqual(len(result["failure"]), 2)


if __name__ == "__main__":
    unittest.main()

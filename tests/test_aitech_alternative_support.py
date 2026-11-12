"""Small exhaustive tests of OR-of-AND debugging evidence support."""
import unittest

from aitech.alternative_support import dangling_evidence, exact_select
from aitech.evidence import Evidence


class TestAlternativeSupport(unittest.TestCase):
    def test_one_of_two_paths_suffices(self):
        items = [
            Evidence("failed_test", "assert failure", 4, 10),
            Evidence("call_bridge", "call trace", 4, 1),
            Evidence("definition", "method definition", 3, 0.6),
            Evidence("assertion", "test assertion", 3, 0.6),
            Evidence("noise", "irrelevant log", 4, 5),
        ]
        support = {"failed_test": (("call_bridge",), ("definition", "assertion"))}
        result = exact_select(items, support, 8)
        self.assertEqual(result.ids, ("failed_test", "call_bridge"))
        self.assertEqual(result.tokens, 8)
        self.assertFalse(result.dangling_ids)
        self.assertEqual(result.supported_ids, ("failed_test",))

    def test_two_supports_in_same_option_both_required(self):
        sets = {"failure": (("trace", "definition"),)}
        self.assertEqual(dangling_evidence(["failure", "trace"], sets), ("failure",))
        self.assertFalse(dangling_evidence(["failure", "trace", "definition"], sets))

    def test_nested_alternative_support_is_checked(self):
        items = [
            Evidence("failure", "", 2, 10),
            Evidence("call", "", 2, 1),
            Evidence("callee_def", "", 2, 1),
            Evidence("unrelated", "", 2, 4),
        ]
        sets = {"failure": (("call",),), "call": (("callee_def",),)}
        best = exact_select(items, sets, 6)
        self.assertEqual(best.ids, ("failure", "call", "callee_def"))
        self.assertFalse(best.dangling_ids)

    def test_incomplete_baseline_can_outscore_complete_selection(self):
        items = [
            Evidence("failure", "", 4, 10),
            Evidence("required", "", 4, 1),
            Evidence("noise", "", 4, 5),
        ]
        sets = {"failure": (("required",),)}
        loose = exact_select(items, sets, 8, require_complete=False)
        complete = exact_select(items, sets, 8, require_complete=True)
        self.assertEqual(loose.ids, ("failure", "noise"))
        self.assertEqual(loose.dangling_ids, ("failure",))
        self.assertEqual(complete.ids, ("failure", "required"))

    def test_circular_only_support_is_not_sufficient(self):
        items = [Evidence("a", "", 2, 2), Evidence("b", "", 2, 2)]
        support = {"a": (("b",),), "b": (("a",),)}
        result = exact_select(items, support, 4)
        self.assertEqual(result.ids, ())
        self.assertEqual(dangling_evidence(["a", "b"], support), ("a", "b"))

    def test_grounded_support_chain_is_accepted(self):
        items = [
            Evidence("anchor", "", 2, 6),
            Evidence("bridge", "", 2, 1),
            Evidence("root", "", 2, 1),
        ]
        supports = {"anchor": (("bridge",),), "bridge": (("root",),)}
        self.assertEqual(exact_select(items, supports, 6).ids,
                         ("anchor", "bridge", "root"))

    def test_unknown_support_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown support"):
            exact_select([Evidence("a", "", 1)], {"a": (("missing",),)}, 1)

    def test_bad_or_empty_alternatives_rejected(self):
        with self.assertRaisesRegex(ValueError, "no support"):
            exact_select([Evidence("a", "", 1)], {"a": ()}, 1)
        with self.assertRaisesRegex(ValueError, "empty support"):
            exact_select([Evidence("a", "", 1)], {"a": ((),)}, 1)

    def test_no_evidence_and_tiny_budget(self):
        self.assertEqual(exact_select([], {}, 0).ids, ())
        self.assertEqual(exact_select([Evidence("x", "", 2, 4)], {}, 1).ids, ())


if __name__ == "__main__":
    unittest.main()

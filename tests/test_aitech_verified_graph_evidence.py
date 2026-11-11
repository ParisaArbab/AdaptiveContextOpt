"""Conservative evidence mapping tests for compiler-attributed Java calls."""
import unittest

from aitech.java_compiler_graph import ResolvedCall
from aitech.verified_graph_evidence import map_calls


def call(owner, method, target_owner, target, types="", kind="METHOD"):
    return ResolvedCall("src/X.java", owner, method, kind,
                        target_owner, target, types)


class TestVerifiedGraphEvidence(unittest.TestCase):
    def test_maps_distinct_compiler_resolved_same_arity_overloads(self):
        ids = [
            "pkg.Num.use(String)",
            "pkg.Num.convert(String)",
            "pkg.Num.convert(Object)",
        ]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert", "java.lang.String"),
            call("pkg.Num", "use", "pkg.Num", "convert", "java.lang.Object"),
        ])
        self.assertEqual(result.edges, ((0, 1), (0, 2)))
        self.assertEqual(result.mapped_calls, 2)
        self.assertFalse(result.unmatched_targets)
        self.assertFalse(result.ambiguous_targets)

    def test_never_maps_overloaded_caller_without_signature(self):
        ids = [
            "pkg.Num.use(String)",
            "pkg.Num.use(Object)",
            "pkg.Num.convert(String)",
        ]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert", "java.lang.String")
        ])
        self.assertFalse(result.edges)
        self.assertEqual(result.ambiguous_callers, ("pkg.Num.use",))

    def test_excludes_target_missing_from_evidence_pool(self):
        ids = ["pkg.Num.use(String)"]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Out.convert", "foo", "int")
        ])
        self.assertFalse(result.edges)
        self.assertTrue(result.unmatched_targets)

    def test_matches_array_and_generic_names_conservatively(self):
        ids = [
            "pkg.Num.use(String)",
            "pkg.Num.convert(java.util.List<String>,int[])",
        ]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert",
                 "java.util.List<java.lang.String>,int[]")
        ])
        self.assertEqual(result.edges, ((0, 1),))

    def test_rejects_duplicate_candidate_signature(self):
        ids = ["pkg.Num.use(String)", "pkg.Num.convert(String)",
               "pkg.Num.convert(String)"]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert", "java.lang.String")
        ])
        self.assertFalse(result.edges)
        self.assertTrue(result.ambiguous_targets)


if __name__ == "__main__":
    unittest.main()

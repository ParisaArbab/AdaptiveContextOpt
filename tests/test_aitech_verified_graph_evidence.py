"""Conservative evidence mapping tests for compiler-attributed Java calls."""
import unittest

from aitech.java_compiler_graph import ResolvedCall
from aitech.verified_graph_evidence import map_calls


def call(owner, method, target_owner, target, types="", kind="METHOD", caller_types=None):
    return ResolvedCall("src/X.java", owner, method, kind,
                        target_owner, target, types, caller_types)


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

    def test_resolves_overloaded_callers_using_compiler_signature(self):
        ids = [
            "pkg.Num.use(String)",
            "pkg.Num.use(Object)",
            "pkg.Num.convert(String)",
            "pkg.Num.convert(Object)",
        ]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert",
                 "java.lang.String", caller_types="java.lang.String"),
            call("pkg.Num", "use", "pkg.Num", "convert",
                 "java.lang.Object", caller_types="java.lang.Object"),
        ])
        self.assertEqual(result.edges, ((0, 2), (1, 3)))
        self.assertFalse(result.ambiguous_callers)
        self.assertEqual(result.mapped_calls, 2)

    def test_unknown_caller_signature_still_abstains(self):
        ids = ["pkg.Num.use(String)", "pkg.Num.use(Object)", "pkg.Num.convert(String)"]
        result = map_calls(ids, [
            call("pkg.Num", "use", "pkg.Num", "convert", "java.lang.String")
        ])
        self.assertEqual(result.edges, ())
        self.assertIn("pkg.Num.use", result.ambiguous_callers)

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

    def test_rejects_two_different_java_date_types_with_same_short_name(self):
        ids = ["pkg.Num.use(String)", "pkg.Num.parse(Date)"]
        records = [
            call("pkg.Num", "use", "pkg.Num", "parse",
                 "java.util.Date", caller_types="java.lang.String"),
            call("pkg.Num", "use", "pkg.Num", "parse",
                 "java.sql.Date", caller_types="java.lang.String"),
        ]
        mapped = map_calls(ids, records)
        self.assertEqual(mapped.edges, ())
        self.assertEqual(mapped.ambiguous_type_identities, ("pkg.Num.parse(Date)",))

    def test_fully_qualified_date_type_matches_only_its_package(self):
        ids = ["pkg.Num.use(String)", "pkg.Num.parse(java.util.Date)"]
        records = [
            call("pkg.Num", "use", "pkg.Num", "parse",
                 "java.util.Date", caller_types="java.lang.String"),
            call("pkg.Num", "use", "pkg.Num", "parse",
                 "java.sql.Date", caller_types="java.lang.String"),
        ]
        mapped = map_calls(ids, records)
        self.assertEqual(mapped.edges, ((0, 1),))
        self.assertFalse(mapped.ambiguous_type_identities)
        self.assertIn("pkg.Num.parse(java.sql.Date)", mapped.unmatched_targets)

    def test_rejects_short_name_collision_in_caller_identity(self):
        ids = ["pkg.Num.run(Date)", "pkg.Num.target()"]
        records = [
            call("pkg.Num", "run", "pkg.Num", "target",
                 caller_types="java.sql.Date"),
            call("pkg.Num", "run", "pkg.Num", "target",
                 caller_types="java.util.Date"),
        ]
        mapped = map_calls(ids, records)
        self.assertFalse(mapped.edges)
        self.assertIn("pkg.Num.run(Date)", mapped.ambiguous_type_identities)

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

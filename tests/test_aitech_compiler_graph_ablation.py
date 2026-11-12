"""Compiler graph selector ablation tests (no network, no JVM)."""
import unittest

from aitech.compiler_graph_ablation import compare_policies


class TestCompilerGraphAblation(unittest.TestCase):
    def test_graph_includes_support_without_exceeding_budget(self):
        ids = ["pkg.A.fail()", "pkg.A.helper()", "pkg.A.noise()"]
        rows = compare_policies(ids, [20, 20, 20], "fail() raised", [(0, 1)], (40,))
        by_name = {r["method"]: r for r in rows}
        self.assertEqual(by_name["compiler_graph_first"]["selected_ids"][:2], ids[:2])
        self.assertEqual(by_name["compiler_graph_first"]["unresolved_mapped_support"], 0)
        self.assertEqual(by_name["compiler_graph_first"]["selected_proxy_tokens"], 40)

    def test_missing_support_is_recorded_not_hidden(self):
        ids = ["pkg.A.fail()", "pkg.A.helper()"]
        rows = compare_policies(ids, [20, 30], "fail()", [(0, 1)], (20,))
        graph = next(r for r in rows if r["method"] == "compiler_graph_first")
        self.assertEqual(graph["failure_name_anchors_kept"], 1)
        self.assertEqual(graph["unresolved_mapped_support"], 1)
        self.assertEqual(graph["selected_proxy_tokens"], 20)

    def test_rejects_invalid_costs(self):
        with self.assertRaises(ValueError):
            compare_policies(["pkg.A.f()"], [], "f()", [], (10,))
        with self.assertRaises(ValueError):
            compare_policies(["pkg.A.f()"], [0], "f()", [], (10,))


if __name__ == "__main__":
    unittest.main()

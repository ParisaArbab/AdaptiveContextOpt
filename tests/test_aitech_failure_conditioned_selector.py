"""Research selector invariants and negative-case regression tests."""
import unittest

from aitech.failure_conditioned_selector import select


def candidate(name, tokens, prior, failure, trace=0):
    return {
        "function_id": name,
        "prior": prior,
        "features": {
            "l1_tokens": tokens,
            "failure_relevance": failure,
            "trace_match": trace,
            "ranker_agreement": 0.5,
        },
    }


class TestFailureConditionedSelector(unittest.TestCase):
    def test_failure_anchor_survives_expensive_dependency(self):
        items = [
            candidate("pkg.Fail.problem(String)", 25, 0.9, 0.8),
            candidate("pkg.Fail.expensiveSupport()", 150, 0.8, 0.3),
            candidate("pkg.Fail.tiny()", 15, 0.6, 0.3),
        ]
        result = select(items, "problem(x) fails", 45,
                        method="failure_graph", edges={(0, 1)})
        self.assertIn(0, result.selected)
        self.assertNotIn(1, result.selected)
        self.assertEqual(result.unresolved_edges, 1)
        self.assertLessEqual(result.proxy_tokens, 45)

    def test_method_controls_use_same_budget(self):
        items = [
            candidate("pkg.Test.alpha()", 35, 0.9, 0.8),
            candidate("pkg.Test.beta()", 45, 0.8, 0.2),
            candidate("pkg.Test.gamma()", 15, 0.6, 0.6),
        ]
        for method in ("rank_fill", "anchor_first", "relevance_density",
                       "failure_graph"):
            result = select(items, "alpha()", 50, method=method)
            self.assertLessEqual(result.proxy_tokens, 50)
            self.assertEqual(result.anchors_total, 1)

    def test_lexical_graph_never_claims_compiler_confidence(self):
        items = [candidate("A.a()", 10, 1, 1)]
        with self.assertRaises(ValueError):
            select(items, "a()", 20, method="failure_graph",
                   graph_confidence="unverified")

    def test_deterministic_repetition(self):
        items = [
            candidate("A.a()", 25, 0.8, 0.5),
            candidate("A.b()", 15, 0.8, 0.5),
        ]
        a = select(items, "a()", 35, method="failure_graph", edges={(0, 1)})
        b = select(items, "a()", 35, method="failure_graph", edges={(0, 1)})
        self.assertEqual(a, b)

    def test_rejects_bad_graph_indices(self):
        items = [candidate("A.a()", 10, 1, 0)]
        with self.assertRaises(ValueError):
            select(items, "", 20, method="failure_graph", edges={(0, 3)})


if __name__ == "__main__":
    unittest.main()

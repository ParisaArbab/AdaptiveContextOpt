"""Regression checks for graph opportunity cost and protected failure evidence."""
import unittest

from aitech.risk_gated_context import Edge, select_context


class TestRiskGatedContext(unittest.TestCase):
    def test_failure_anchor_survives_expensive_dependency(self):
        result = select_context([25, 90, 25], [0],
                                [Edge(0, 1, 1., "compiler")], 50)
        self.assertEqual(result.selected, (0, 2))
        self.assertEqual(result.anchors_retained, (0,))
        self.assertEqual(result.unsatisfied, ((0, 1),))
        self.assertLessEqual(result.tokens, 50)

    def test_confident_affordable_graph_support(self):
        result = select_context([20, 25, 30], [0],
                                [Edge(0, 2, 1., "compiler")], 50)
        self.assertEqual(result.selected, (0, 2))
        self.assertFalse(result.unsatisfied)
        self.assertTrue(any(d["reason"] == "graph_support"
                            for d in result.decisions))

    def test_weak_graph_edge_abstains(self):
        result = select_context([20, 25, 30], [0],
                                [Edge(0, 2, .5, "lexical")], 50)
        self.assertEqual(result.selected, (0, 1))
        self.assertEqual(result.graph_rejected, ((0, 2),))

    def test_zero_weight_matches_rank_counterfactual(self):
        graph = [Edge(0, 2, 1., "compiler")]
        a = select_context([20, 25, 30], [0], graph, 50, graph_weight=0)
        b = select_context([20, 25, 30], [0], [], 50)
        self.assertEqual(a.selected, b.selected)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            select_context([2, 0], [], [], 5)
        with self.assertRaises(ValueError):
            select_context([2], [2], [], 5)
        with self.assertRaises(ValueError):
            select_context([2], [], [Edge(0, 3, 1, "compiler")], 5)
        with self.assertRaises(ValueError):
            select_context([2], [], [], -1)


if __name__ == "__main__":
    unittest.main()

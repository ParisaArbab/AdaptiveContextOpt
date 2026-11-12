"""Tests on two real archived Defects4J source candidate snapshots."""
import unittest

from aitech.multibug_call_ablation import (
    SOURCES, candidate_graph, run_snapshots, select,
)


class TestMultibugCallAblation(unittest.TestCase):
    def test_real_snapshots_budget_and_edge_accounting(self):
        report = run_snapshots()
        self.assertEqual(set(report["provenance"]), {"Lang-1", "Time-25"})
        self.assertEqual(len(report["rows"]), 18)
        self.assertFalse(report["gold_labels_used"])
        for row in report["rows"]:
            self.assertLessEqual(row["words"], row["budget"])
            self.assertLessEqual(row["closed_edges"], row["retained_edges"])
        lookup = {(r["bug"], r["budget_pct"], r["method"]): r
                  for r in report["rows"]}
        self.assertEqual(lookup[("Lang-1", 35, "rank_fill")]["kept"], 7)
        self.assertEqual(lookup[("Lang-1", 35, "closure")]["kept"], 12)
        self.assertEqual(lookup[("Time-25", 35, "rank_fill")]["kept"], 13)
        self.assertEqual(lookup[("Time-25", 35, "closure")]["kept"], 10)
        self.assertEqual(lookup[("Time-25", 35, "closure")]["edge_completion"], 1.0)

    def test_closure_detects_call_and_can_refuse_group(self):
        items = [
            {"id": "A", "name": "alpha", "source": "beta();", "words": 2},
            {"id": "B", "name": "beta", "source": "return 1;", "words": 4},
        ]
        graph = candidate_graph(items)
        self.assertEqual(graph[0], [1])
        relevance = select(items, graph, budget=2, mode="rank_fill")
        closure = select(items, graph, budget=2, mode="closure")
        self.assertEqual(relevance["kept"], 1)
        self.assertEqual(closure["kept"], 0)


if __name__ == "__main__":
    unittest.main()

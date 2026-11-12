"""Tests for pre-gold, failure-aligned reranking and order controls."""
import unittest

from aitech.failure_aligned_reranker import failure_rerank, shortlist


def snap():
    return {
        "context": "Failing tests: 1\njava.lang.AssertionError\n"
                   "at org.example.Parser.parse(Parser.java:21)\n",
        "candidates": [
            {"function_id": "org.other.Helper.util()",
             "source_path": "Helper.java",
             "levels": ["util", "does a thing"]},
            {"function_id": "org.example.Parser.parse(String)",
             "source_path": "Parser.java",
             "levels": ["parse(String)", "parse the failing input string"]},
            {"function_id": "org.example.Parser.format()",
             "source_path": "Parser.java",
             "levels": ["format()", "format a value"]},
            {"function_id": "org.other.Helper.string()",
             "source_path": "Helper.java",
             "levels": ["string()", "string value"]},
        ],
    }


class TestFailureAlignedReranker(unittest.TestCase):
    def test_exact_trace_method_ranks_first(self):
        ranked = failure_rerank(snap())
        self.assertEqual(ranked[0]["index"], 1)
        self.assertEqual(ranked[0]["trace_boost"], 16.0)

    def test_no_gold_classes_or_patches_required(self):
        data = snap()
        data["provenance"] = {"gold_patches_loaded": False}
        self.assertEqual(shortlist(data, 3)[0], 1)

    def test_deterministic_and_no_candidate_mutation(self):
        data = snap()
        initial = [c["function_id"] for c in data["candidates"]]
        self.assertEqual(shortlist(data, 4), shortlist(data, 4))
        self.assertEqual(initial, [c["function_id"] for c in data["candidates"]])

    def test_diversity_restricts_one_file(self):
        data = snap()
        data["candidates"].extend(
            {"function_id": f"org.example.Parser.other{i}()",
             "source_path": "Parser.java", "levels": ["id", "parse input"]}
            for i in range(5)
        )
        result = shortlist(data, 5)
        self.assertLessEqual(
            sum(data["candidates"][i]["source_path"] == "Parser.java"
                for i in result), 2)

    def test_rejects_tiny_limit(self):
        with self.assertRaises(ValueError):
            shortlist(snap(), 1)


if __name__ == "__main__":
    unittest.main()

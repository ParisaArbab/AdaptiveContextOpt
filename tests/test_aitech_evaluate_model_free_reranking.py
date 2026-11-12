"""Gold-frozen model-free ranking baseline tests."""
import unittest

from aitech.evaluate_model_free_reranking import pre_gold_predictions, score_frozen


def snapshot():
    return {
        "context": ("--- org.example.ParserTests::testParse\n"
                    "java.lang.AssertionError"),
        "provenance": {"gold_patches_loaded": False, "failure_sha256": "testsha"},
        "candidates": [
            {"function_id": "org.other.Other.unrelated()",
             "source_path": "Other.java", "levels": ["other", "unrelated"]},
            {"function_id": "org.example.Parser.parse()",
             "source_path": "Parser.java", "levels": ["parse", "parses a value"]},
        ],
    }


class TestModelFreeRanking(unittest.TestCase):
    def test_freeze_without_llm_or_gold(self):
        frozen = pre_gold_predictions({"Fixture-1": snapshot()})
        self.assertEqual(len(frozen), 3)
        self.assertEqual(
            [x["condition"] for x in frozen],
            ["original_top8", "trace_idf", "test_name_anchor"])
        self.assertTrue(all(r["model_input_tokens"] == 0 for r in frozen))
        self.assertTrue(all(not r["model_inference_executed"] for r in frozen))

    def test_scoring_happens_after_freeze(self):
        frozen = pre_gold_predictions({"Fixture-1": snapshot()})
        scored = score_frozen(frozen, {"results": [{
            "bug": "Fixture-1", "gold_modified_classes": ["org.example.Parser"]}]})
        self.assertEqual(
            scored["model_free_baselines"]["test_name_anchor"]
            ["changed_class_hits_at_1"], 1)
        self.assertFalse(scored["model_inference_executed"])

    def test_tainted_snapshot_rejected(self):
        data = snapshot()
        data["provenance"]["gold_patches_loaded"] = True
        with self.assertRaisesRegex(ValueError, "gold-tainted"):
            pre_gold_predictions({"Fixture-1": data})


if __name__ == "__main__":
    unittest.main()

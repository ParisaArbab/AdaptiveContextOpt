"""No-model tests for candidate-order controls and gold isolation."""
import unittest

from aitech.cpu_failure_rerank_ablation import (
    build_frozen_prompts, score_predictions
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " assistant:"


def snapshot():
    return {
        "context": "java.lang.AssertionError\n"
                   "at org.example.Parser.parse(Parser.java:20)",
        "provenance": {"gold_patches_loaded": False, "failure_sha256": "abc"},
        "candidates": [
            {"function_id": "org.other.Helper.thing()",
             "source_path": "Helper.java",
             "levels": ["Helper thing", "thing java"],
             "features": {"l0_tokens": 3, "l1_tokens": 3}},
            {"function_id": "org.example.Parser.parse(int)",
             "source_path": "Parser.java",
             "levels": ["Parser parse", "parse int value"],
             "features": {"l0_tokens": 3, "l1_tokens": 3}},
            {"function_id": "org.other.Second.other()",
             "source_path": "Second.java",
             "levels": ["Second other", "other java"],
             "features": {"l0_tokens": 3, "l1_tokens": 3}},
        ]
    }


class TestCpuFailureRerankAblation(unittest.TestCase):
    def test_original_and_reranked_have_distinct_first_method(self):
        frozen = build_frozen_prompts(snapshot(), Tokenizer(),
                                      max_input_tokens=250, limit=3)
        arms = frozen["conditions"]
        self.assertEqual(arms["original_top8"]["candidate_indices"][0], 0)
        self.assertEqual(arms["failure_rerank"]["candidate_indices"][0], 1)
        self.assertEqual(
            arms["failure_rerank_reversed"]["candidate_indices"],
            list(reversed(arms["failure_rerank"]["candidate_indices"])))
        self.assertEqual(
            set(arms["failure_rerank"]["candidate_ids"]),
            set(arms["failure_rerank_reversed"]["candidate_ids"]))
        self.assertEqual(len(arms["original_top8"]["candidate_ids"]), 3)

    def test_candidate_order_changes_prompt_hash(self):
        frozen = build_frozen_prompts(snapshot(), Tokenizer(),
                                      max_input_tokens=250, limit=3)
        arms = frozen["conditions"]
        self.assertNotEqual(
            arms["failure_rerank"]["prompt_sha256"],
            arms["failure_rerank_reversed"]["prompt_sha256"])
        self.assertTrue(frozen["no_gold_used"])

    def test_tight_common_ceiling_is_respected(self):
        frozen = build_frozen_prompts(snapshot(), Tokenizer(),
                                      max_input_tokens=33, limit=3)
        for row in frozen["conditions"].values():
            if not row["skip_reason"]:
                self.assertLessEqual(row["actual_input_tokens"], 33)

    def test_golden_patch_data_never_allowed_in_snapshot(self):
        data = snapshot()
        data["provenance"]["gold_patches_loaded"] = True
        with self.assertRaisesRegex(ValueError, "pre-gold"):
            build_frozen_prompts(data, Tokenizer())

    def test_score_is_late_and_does_not_change_predictions(self):
        raw = [{"bug": "Fixture-1", "predictions": [{
            "condition": "failure_rerank",
            "predicted_method": "org.example.Parser.parse(int)",
            "actual_input_tokens": 90, "output_tokens": 1,
            "generation_seconds": 0.1,
            "candidate_ids": ["org.example.Parser.parse(int)"],
        }]}]
        posthoc = {"results": [{"bug": "Fixture-1",
                               "gold_modified_classes": ["org.example.Parser"]}]}
        scored = score_predictions(raw, posthoc)
        self.assertTrue(scored["results"][0]["changed_class_hit_at_1"])
        self.assertEqual(scored["results"][0]["predicted_method"],
                         raw[0]["predictions"][0]["predicted_method"])


if __name__ == "__main__":
    unittest.main()

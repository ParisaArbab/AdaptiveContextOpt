"""No-network unit tests for Ollama localization adapter."""
import unittest

from aitech.ollama_localization import (
    GOLD_FILE, GOLD_ENTITY, experiment, parse_prediction, score_offline,
)
from aitech.sympy_trace_pilot import ANCHORS

TRACE = "\n".join(f"  {i}: {text}" for i, (_, text, _) in enumerate(ANCHORS, 1))


class TestLocalizationAdapter(unittest.TestCase):
    def test_json_parsing_and_gold_free_prompt(self):
        prompts = []
        def fake_call(endpoint, model, prompt, timeout, seed):
            prompts.append(prompt)
            return {
                "message": {"content": '{"file": "sympy/core/basic.py", "function": "Basic", "reason": "hypothesis"}'},
                "prompt_eval_count": 50,
                "eval_count": 20,
                "measured_elapsed_seconds": 0.05,
            }
        result = experiment(TRACE, model="fake-model", infer=fake_call)
        self.assertEqual(set(result["conditions"]), {"raw_excerpt", "relevance_only", "dependency_aware"})
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all(GOLD_FILE not in p for p in prompts))
        self.assertTrue(all("Identify a likely source" in p for p in prompts))
        self.assertEqual(result["conditions"]["relevance_only"]["tokens_proxy"],
                         result["conditions"]["dependency_aware"]["tokens_proxy"])
        self.assertFalse(result["conditions"]["raw_excerpt"]["offline_score"]["file_hit_1"])
        self.assertEqual(result["conditions"]["raw_excerpt"]["prompt_eval_count"], 50)

    def test_gold_scoring_is_only_in_evaluator(self):
        result = score_offline({"file": GOLD_FILE, "function": GOLD_ENTITY})
        self.assertTrue(result["file_hit_1"])
        self.assertTrue(result["entity_hit_1"])

    def test_invalid_model_output_remains_a_failure(self):
        self.assertFalse(parse_prediction({"message": {"content": "not json"}})["parse_ok"])
        self.assertFalse(parse_prediction({"message": {"content": "[]"}})["parse_ok"])


if __name__ == "__main__":
    unittest.main()

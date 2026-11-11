"""Fast, offline invariants for the CPU LLM localization harness."""
import unittest

from aitech.cpu_llm_localization import (
    build_messages, choose_evidence, freeze_selection, parse_prediction,
    score_frozen_predictions, with_real_costs,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(x["role"] + ": " + x["content"] + "\n" for x in messages)


def fake_source():
    return {
        "context": "bad() raised an exception",
        "candidates": [{
            "function_id": "pkg.Test.bad()",
            "levels": ["METHOD bad()\nSIGNATURE bad()", "METHOD bad()\nreturn broken();"],
            "prior": 0.9,
            "features": {
                "l0_tokens": 10, "l1_tokens": 20,
                "failure_relevance": 0.9, "trace_match": 1.0,
            },
        }, {
            "function_id": "pkg.Test.other()",
            "levels": ["METHOD other()", "METHOD other()\nreturn 42;"],
            "prior": 0.5,
            "features": {
                "l0_tokens": 10, "l1_tokens": 20,
                "failure_relevance": 0.1, "trace_match": 0,
            },
        }],
    }


class TestCpuLLMLocalization(unittest.TestCase):
    def test_real_cost_replacement_and_freezing(self):
        tokenizer = FakeTokenizer()
        snapshot = fake_source()
        normalized = with_real_costs(snapshot, tokenizer)
        self.assertGreater(normalized["candidates"][0]["features"]["l1_tokens"], 10)
        data = {"Fixture-1": {"source": snapshot, "sha256": "a" * 64}}
        frozen = freeze_selection(data, tokenizer, 300)
        self.assertEqual(len(frozen), 3)
        self.assertEqual({r["policy"] for r in frozen},
                         {"rank_fill", "anchor_rank_detail", "tiered_marginal"})
        for row in frozen:
            self.assertLessEqual(row["actual_evidence_tokens"], 300)
            self.assertEqual(len(row["prompt_sha256"]), 64)

    def test_predictions_cannot_use_missing_candidate(self):
        picked = {0: 2, 3: 1}
        self.assertEqual(parse_prediction("INDEX: 4", picked), 3)
        self.assertEqual(parse_prediction("INDEX: 2", picked), None)
        self.assertEqual(parse_prediction("I don't know", picked), None)

    def test_scoring_reports_l0_as_identity_not_source(self):
        prediction = {
            "bug": "Fixture-1", "policy": "tiered_marginal",
            "predicted_method": "pkg.Test.bad()",
            "selected_ids": ["pkg.Test.bad()"],
            "selected_levels": {"0": 1},
            "candidate_id_by_index": {"0": "pkg.Test.bad()"},
            "actual_total_input_tokens": 100,
            "actual_evidence_tokens": 30,
            "generation_seconds": 0.1,
        }
        labels = {"scoring_only": True,
                  "bugs": {"Fixture-1": {"changed_methods": ["pkg.Test.bad()"]}}}
        scored = score_frozen_predictions([prediction], labels)["results"][0]
        self.assertTrue(scored["predicted_gold_changed_method"])
        self.assertEqual(scored["changed_methods_in_context"], 1)
        self.assertEqual(scored["changed_methods_with_L1"], 0)


if __name__ == "__main__":
    unittest.main()

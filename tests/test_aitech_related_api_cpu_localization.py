"""Tests for matched related-API Qwen protocol without real model dependency."""
import unittest

from aitech.related_api_cpu_localization import ARMS, freeze, score


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return " ".join(m["role"] + ":" + m["content"]
                        for m in messages)


def snapshot():
    items = []
    for i in range(9):
        owner = ("pkg.Reader" if i < 2 else
                 "pkg.Writer" if i == 8 else f"pkg.Helper{i}")
        name = f"m{i}"
        function = f"{owner}.{name}()"
        items.append({
            "function_id": function,
            "levels": [function, function + "\nmethod body uses reader"],
        })
    return {
        "context": "at pkg.Reader.m0(Reader.java:24)",
        "candidates": items,
        "provenance": {"gold_patches_loaded": False},
    }


class TestRelatedApiCpuLocalization(unittest.TestCase):
    def test_all_arms_fixed_count_and_budget(self):
        chosen = freeze(snapshot(), Tokenizer(), max_input_tokens=1200)
        self.assertEqual(tuple(item["arm"] for item in chosen), ARMS)
        self.assertTrue(all(len(item["selected_method_ids"]) == 8
                            for item in chosen))
        self.assertEqual(
            chosen[1]["selected_method_ids"],
            list(reversed(chosen[3]["selected_method_ids"])))
        self.assertEqual(
            chosen[1]["selected_method_ids"], chosen[2]["selected_method_ids"])
        self.assertTrue(all(item["actual_input_tokens"] <= 1200
                            for item in chosen))

    def test_related_peer_l1_exposes_implementation_when_present(self):
        chosen = {arm["arm"]: arm for arm in freeze(
            snapshot(), Tokenizer(), max_input_tokens=1200)}
        peer = chosen["related_peer_L1"]
        self.assertFalse(peer.get("skip_reason"))
        self.assertIn(2, peer["selected_levels"].values())
        self.assertEqual(peer["selected_method_ids"],
                         chosen["related_L0"]["selected_method_ids"])
        self.assertGreater(peer["actual_input_tokens"],
                           chosen["related_L0"]["actual_input_tokens"])

    def test_changed_class_scoring_is_distinct_from_retrieval_recall(self):
        arms = freeze(snapshot(), Tokenizer(), max_input_tokens=1200)
        raw = [{"bug": "Fixture-1", "arms": [
            {**{k: v for k, v in arm.items() if k != "prompt"},
             "predicted_method": arm["selected_method_ids"][0],
             "generation_seconds": 1.0}
            for arm in arms]}]
        result = score(raw, {"Fixture-1": {"pkg.Writer"}})
        metrics = result["results"][0]
        self.assertTrue(metrics["changed_class_covered_by_related_8"])
        self.assertFalse(metrics["arms"]["related_L0"]["class_hit_at_1"])
        self.assertTrue(result["raw_responses_saved_before_labels"])

    def test_pre_gold_source_required(self):
        source = snapshot()
        source["provenance"]["gold_patches_loaded"] = True
        with self.assertRaisesRegex(ValueError, "pre-gold"):
            freeze(source, Tokenizer())


if __name__ == "__main__":
    unittest.main()

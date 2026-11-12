"""Tests for gold-free stack reranking, matched arms and order control."""
import unittest

from aitech.failure_grounded_rerank import rerank, shortlist, terms, failing_test_class_hints
from aitech.grounded_rerank_cpu_ablation import freeze_arms, score_arms


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return " ".join(x["role"] + ": " + x["content"]
                        for x in messages) + " assistant:"


def snapshot():
    candidates = []
    for i in range(10):
        name = "check" if i == 8 else f"unused{i}"
        owner = "pkg.Core" if i == 8 else f"pkg.Helper{i}"
        fn = f"{owner}.{name}()"
        candidates.append({
            "function_id": fn,
            "levels": [
                f"FILE src/{owner}.java: METHOD {fn}",
                f"FILE src/{owner}.java: METHOD {fn}\n"
                "public void method() { int x = 0; "
                "long detail = 7; verifyResult(); }"
            ],
            "features": {"l0_tokens": 10, "l1_tokens": 20,
                         "failure_relevance": 0.5}
        })
    return {
        "context": ("Failing tests: 1\n"
                    "java.lang.IllegalArgumentException: wrong value\n"
                    "at pkg.Core.check(Core.java:31)\n"),
        "candidates": candidates,
        "provenance": {"gold_patches_loaded": False,
                       "failure_sha256": "failure-hash"}
    }


class TestGroundedRerank(unittest.TestCase):
    def test_trace_binding_lifts_important_candidate(self):
        snap = snapshot()
        ranked = rerank(snap)
        self.assertEqual(ranked[0], 8)
        self.assertEqual(set(ranked), set(range(10)))
        self.assertEqual(len(set(terms("badCode valueHandler"))), 4)

    def test_failing_test_name_generates_weak_class_hint(self):
        failure = ("--- pkg.junit.WidgetTests::testBroken\\n"
                   "--- pkg.TestFraction::failure\\n"
                   "java.lang.AssertionError")
        self.assertEqual(failing_test_class_hints(failure),
                         {"Widget", "Fraction"})

    def test_no_bug_oracle_from_unrelated_assertion(self):
        self.assertEqual(
            failing_test_class_hints(
                "java.lang.AssertionError at junit.framework.Assert.fail"),
            set())

    def test_shortlist_rejects_duplicated_indices(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            shortlist(snapshot(), [0, 0], 2)

    def test_all_arms_share_candidate_count_and_reversed_identity(self):
        frozen = {x["arm"]: x for x in freeze_arms(
            snapshot(), Tokenizer(), count=8, max_input_tokens=700)}
        self.assertEqual(len(frozen), 4)
        self.assertEqual(frozen["grounded_first8"]["selected_method_ids"],
                         list(reversed(frozen["grounded_reversed"]
                                       ["selected_method_ids"])))
        self.assertEqual(
            set(frozen["grounded_first8"]["selected_method_ids"]),
            set(frozen["grounded_one_L1"]["selected_method_ids"]))
        self.assertEqual(frozen["grounded_first8"]["selected_method_ids"][0],
                         "pkg.Core.check()")
        for arm in frozen.values():
            if arm["skip_reason"] is None:
                self.assertLessEqual(arm["actual_input_tokens"], 700)

    def test_scoring_rejects_order_bias_as_agreement(self):
        snap = snapshot()
        arms = freeze_arms(snap, Tokenizer(), count=8,
                           max_input_tokens=700)
        raw = [{"bug": "Fixture-1", "arms": [
            {**x, "predicted_method": x["selected_method_ids"][0]}
            for x in arms
        ]}]
        scored = score_arms(raw, {"Fixture-1": {"pkg.Core"}})
        result = scored["results"][0]["arms"]
        self.assertFalse(result["position_agreement"])
        self.assertTrue(result["grounded_has_gold_in_pool"])
        self.assertFalse(result["original_has_gold_in_pool"])
        self.assertTrue(result["grounded_first8"]["hit_at_1_modified_class"])


if __name__ == "__main__":
    unittest.main()

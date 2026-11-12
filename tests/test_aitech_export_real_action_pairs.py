"""Verify that real-model artifacts become correctly labeled training rows."""
import unittest

from aitech.export_real_action_pairs import export_rows


def fixture():
    candidate = {
        "function_id": "org.test.A.fail()",
        "features": {"failure_relevance": 0.8},
    }
    snap = {
        "context": "fail() raised error",
        "candidates": [candidate],
        "provenance": {
            "gold_patches_loaded": False,
            "failure_sha256": "failure-hash",
        },
    }
    spec = {
        "skip_reason": None,
        "target_method": "org.test.A.fail()",
        "upgrade_index": 0,
        "snapshot_sha256": "buggy-source-hash",
        "common_total_input_ceiling": 500,
        "prompts": [
            {"condition": "STOP", "prompt_sha256": "before"},
            {"condition": "UPGRADE_L1", "prompt_sha256": "after"},
        ],
    }
    raw = [{
        "bug": "Math-1",
        "skip_reason": None,
        "model_id": "Qwen-test", "model_revision": "revision",
        "predictions": [
            {"condition": "STOP", "prompt_sha256": "before",
             "actual_total_input_tokens": 200,
             "raw_response": "2", "predicted_method": "other()"},
            {"condition": "UPGRADE_L1", "prompt_sha256": "after",
             "actual_total_input_tokens": 220,
             "raw_response": "1",
             "predicted_method": "org.test.A.fail()"},
        ],
    }]
    posthoc = {
        "frozen_before_gold": True,
        "results": [{
            "bug": "Math-1", "skipped": False,
            "outcome": 1, "stop_file_hit": False,
            "upgrade_file_hit": True,
            "stop_actual_input_tokens": 200,
            "upgrade_actual_input_tokens": 220,
        }],
    }
    return raw, {"Math-1": spec}, posthoc, {"Math-1": snap}, {"Math": "train"}


class TestExportRealActionPairs(unittest.TestCase):
    def test_exported_row_validated_with_realistic_labels(self):
        rows, summary = export_rows(*fixture())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], 1)
        self.assertEqual(rows[0]["outcome_definition"],
                         "defects4j_changed_class_hit_at_1")
        self.assertEqual(rows[0]["action"]["incremental_input_tokens"], 20)
        self.assertEqual(summary["validation"]["count"], 1)
        self.assertEqual(rows[0]["action"]["source_revision"],
                         "buggy-source-hash")

    def test_no_training_rows_from_skipped_bug(self):
        raw, frozen, scored, snapshots, split = fixture()
        raw[0]["skip_reason"] = "no_upgrade"
        scored["results"][0] = {
            "bug": "Math-1", "skipped": True, "reason": "no_upgrade"}
        rows, summary = export_rows(raw, frozen, scored, snapshots, split)
        self.assertEqual(rows, [])
        self.assertEqual(summary["exported_pairs"], 0)

    def test_prompt_tamper_rejected(self):
        raw, frozen, scored, snapshots, split = fixture()
        raw[0]["predictions"][0]["prompt_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "frozen prompts"):
            export_rows(raw, frozen, scored, snapshots, split)

    def test_posthoc_mismatch_rejected(self):
        raw, frozen, scored, snapshots, split = fixture()
        scored["results"][0]["outcome"] = -1
        with self.assertRaisesRegex(ValueError, "disagree"):
            export_rows(raw, frozen, scored, snapshots, split)

    def test_requires_explicit_project_split(self):
        raw, frozen, scored, snapshots, split = fixture()
        with self.assertRaisesRegex(ValueError, "declared split"):
            export_rows(raw, frozen, scored, snapshots, {})


if __name__ == "__main__":
    unittest.main()

"""Post-selection scoring checks, separate from retrieval and evidence ranking."""
import json
import unittest

from aitech.time1_posthoc_scoring import PATCH_LABELS, score
from aitech.time1_verified_retrieval_study import SNAPSHOT, SOURCE_RECORD, run


class TestTime1PosthocScoring(unittest.TestCase):
    def setup_data(self):
        selection = run(
            json.loads(SNAPSHOT.read_text(encoding="utf-8")),
            json.loads(SOURCE_RECORD.read_text(encoding="utf-8")))
        patch = json.loads(PATCH_LABELS.read_text(encoding="utf-8"))
        return selection, patch

    def test_labels_source_is_official_patch_and_two_real_methods(self):
        selection, patch = self.setup_data()
        self.assertEqual(len(patch["changed_method_candidate_ids"]), 2)
        self.assertIn("Time/patches/1.src.patch", patch["label_source"])
        result = score(selection, patch)
        self.assertTrue(result["selection_frozen_before_scoring"])
        self.assertFalse(result["causal_fault_localization_measured"])
        self.assertEqual(len(result["rows"]), 15)
        for row in result["rows"]:
            self.assertLessEqual(row["changed_method_hits"],
                                 row["changed_method_total"])
            self.assertIn(row["changed_method_recall"],
                          (0, .5, 1.))

    def test_score_does_not_change_evidence_selections(self):
        selection, patch = self.setup_data()
        before = json.dumps(selection, sort_keys=True)
        score(selection, patch)
        self.assertEqual(json.dumps(selection, sort_keys=True), before)

    def test_unfrozen_patch_labels_rejected(self):
        selection, patch = self.setup_data()
        patch["scoring_only"] = False
        with self.assertRaises(ValueError):
            score(selection, patch)


if __name__ == "__main__":
    unittest.main()

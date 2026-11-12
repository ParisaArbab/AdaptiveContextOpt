"""Explicitly separate candidate recall from downstream localization accuracy."""
import unittest

from aitech.related_api_retrieval_pilot import freeze, score


def candidate(id):
    return {"function_id": id, "levels": [id, id + " sample source"]}


class TestRelatedApiRetrievalPilot(unittest.TestCase):
    def snapshot(self):
        methods = [
            "pkg.ArchiveInputStream.read()",
            "pkg.ArchiveInputStream.next()",
            "pkg.Other.alpha()",
            "pkg.Another.beta()",
            "pkg.Foo.gamma()",
            "pkg.Bar.delta()",
            "pkg.ArchiveOutputStream.write()",
            "pkg.Zoo.epsilon()",
            "pkg.Zoo.zeta()",
        ]
        return {
            "context": "at pkg.ArchiveInputStream.read(File.java:10)",
            "candidates": [candidate(id) for id in methods],
            "provenance": {
                "gold_patches_loaded": False,
                "failure_sha256": "verified-raw",
            },
        }

    def test_related_api_recovers_present_gold_class_at_same_count(self):
        frozen = freeze(self.snapshot(), count=4)
        original = score(frozen, {"pkg.ArchiveOutputStream"})
        self.assertFalse(original["original_top8"]["modified_class_present"])
        self.assertTrue(original["related_api_top8"]["modified_class_present"])
        self.assertTrue(original["related_api_top8"]["not_a_model_hit_at_1"])
        for selected in frozen["selections"].values():
            self.assertEqual(len(selected["indices"]), 4)

    def test_gold_not_in_snapshot_before_freeze(self):
        source = self.snapshot()
        source["provenance"]["gold_patches_loaded"] = True
        with self.assertRaisesRegex(ValueError, "pre-gold"):
            freeze(source, count=4)

    def test_scoring_rejects_missing_modified_classes(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            score(freeze(self.snapshot(), count=4), set())


if __name__ == "__main__":
    unittest.main()

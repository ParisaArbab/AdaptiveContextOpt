"""Support-annotation collection schema tests, with no fabricated training."""
import unittest
from aitech.support_annotation_schema import validate_annotations


def row(bug, project, split, sufficient=True):
    return {
        "bug": bug, "project": project, "buggy_revision": "a" * 40,
        "anchor_id": "pkg.A.failure()", "support_ids": ["pkg.A.helper()"],
        "provenance": "javac:source-commit:audit", "compiler_binding_verified": True,
        "sufficient": sufficient, "audit_reference": f"manual-review:{bug}",
        "split": split,
    }


class TestSupportAnnotationSchema(unittest.TestCase):
    def test_three_independent_projects_can_be_split(self):
        result = validate_annotations([
            row("Lang-1", "Lang", "train"),
            row("Time-1", "Time", "calibration", False),
            row("Math-1", "Math", "test"),
        ])
        self.assertEqual(result["project_count"], 3)
        self.assertEqual(result["splits"], {"calibration": 1, "test": 1, "train": 1})
        self.assertFalse(result["trained_model"])

    def test_same_project_in_different_splits_is_leakage(self):
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_annotations([
                row("Time-1", "Time", "train"),
                row("Time-25", "Time", "test"),
            ])

    def test_claim_of_sufficiency_without_verified_binding_rejected(self):
        record = row("Lang-1", "Lang", "train")
        record["compiler_binding_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified binding"):
            validate_annotations([record])

    def test_duplicate_label_and_missing_audit_rejected(self):
        record = row("Lang-1", "Lang", "train")
        with self.assertRaisesRegex(ValueError, "repeated annotation"):
            validate_annotations([record, dict(record)])
        record["audit_reference"] = ""
        with self.assertRaisesRegex(ValueError, "audit reference"):
            validate_annotations([record])

    def test_no_fake_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, "no annotations"):
            validate_annotations([])


if __name__ == "__main__":
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rave_fl.flexfl_adapter import load_candidates
from rave_fl.experiment import profile


class FlexFLAdapterTests(unittest.TestCase):
    def test_read_real_format_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "data" / "FL_results" / "Ochiai" / "Defects4J"
            results.mkdir(parents=True)
            with (results / "Time-25_method-susps.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["File", "Signature"])
                writer.writeheader()
                writer.writerow({"File": "org.demo.Parser", "Signature": "parse()"})
                writer.writerow({"File": "org.demo.Renderer", "Signature": "render()"})
            gt = root / "data" / "input" / "ground_truth" / "Defects4J"
            gt.mkdir(parents=True)
            (gt / "gt.json").write_text(json.dumps({"Time-25": ["org.demo.Parser.parse()"]}))
            candidates = load_candidates(root, "Time-25")
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].function_id, "org.demo.Parser.parse()")
            self.assertEqual(profile(root, "Time-25", "Ochiai", 100)["baseline_accuracy"]["top1"], True)
            self.assertFalse(profile(root, "Time-25", "Ochiai", 100)["real_verifier"])

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_candidates(Path(tmp), "Time-25")


if __name__ == "__main__":
    unittest.main()

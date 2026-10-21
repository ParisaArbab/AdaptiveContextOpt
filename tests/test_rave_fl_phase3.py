import unittest
from unittest.mock import patch
from rave_fl.core import Candidate
from rave_fl.compare_llm import run_arm
from rave_fl.ollama_verify import Result, admission_estimate


class Phase3Tests(unittest.TestCase):
    def test_admission_positive(self):
        self.assertGreater(admission_estimate("error", "a()", "return null;", 96), 96)

    @patch("rave_fl.compare_llm.verify")
    def test_selective_calls_fewer_methods(self, mocked):
        mocked.side_effect = lambda url, model, ctx, name, source, limit: Result(
            name, 0.8, "test", 15, 7, 0.1)
        candidates = [Candidate("a", "a()", semantic_score=1.0),
                      Candidate("b", "b()", semantic_score=0.5)]
        sources = {"a": "return 1;", "b": "return 2;"}
        full = run_arm(candidates, sources, "failure", "model", "url", 3000, False, 96)
        partial = run_arm(candidates, sources, "failure", "model", "url", 3000, True, 96)
        self.assertEqual(full["verification_count"], 2)
        self.assertEqual(partial["verification_count"], 1)
        self.assertEqual(full["actual_tokens"], 44)
        self.assertEqual(partial["actual_tokens"], 22)
        self.assertFalse(partial["budget_exceeded"])

    @patch("rave_fl.compare_llm.verify")
    def test_no_call_when_budget_small(self, mocked):
        c = [Candidate("a", "a()", semantic_score=1)]
        output = run_arm(c, {"a": "return 1;"}, "failure", "m", "url", 1, False, 96)
        self.assertEqual(output["verification_count"], 0)
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()

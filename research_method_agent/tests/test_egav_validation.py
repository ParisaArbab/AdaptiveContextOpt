"""Offline tests for frozen EGAV development validation. No Ollama calls."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_method_agent"))
import egav_validation as v


class TestFrozenEGAVValidation(unittest.TestCase):
    def test_source_policy_is_frozen(self):
        v.frozen_policy_check()

    def test_two_batches_are_distinct_and_exclude_pilot(self):
        groups = []
        for batch in ("01", "02"):
            path = ROOT / ("data/graft_100_cohort_20261115/batches/batch_" +
                           batch + "/experiment/config.json")
            bugs = json.loads(path.read_text())["benchmark"]["bugs"]
            self.assertEqual(len(bugs), 5)
            self.assertEqual(len(set(bugs)), 5)
            self.assertFalse(set(bugs) & v.PILOT)
            groups.append(set(bugs))
        self.assertFalse(groups[0] & groups[1])

    def test_prepare_has_no_llm_permission_or_gold(self):
        args = types.SimpleNamespace(allow_inference=False, ollama_url="unused",
                                     max_calls=450, physical_token_cap=800000)
        with patch.object(v, "config", return_value=(ROOT / "test-config.json",
                                                       ["Chart-1", "Closure-128",
                                                        "Lang-35", "Math-57", "Time-14"])):
            with patch.object(v.subprocess, "run") as fake:
                v.invoke("01", False, args)
        command = fake.call_args[0][0]
        self.assertEqual(command[2], "study")
        self.assertNotIn("--allow-inference", command)
        self.assertNotIn("--gold", command)
        self.assertEqual(command[command.index("--bugs") + 1],
                         "Chart-1,Closure-128,Lang-35,Math-57,Time-14")

    def test_live_run_requires_explicit_permission(self):
        args = types.SimpleNamespace(allow_inference=False, ollama_url="unused",
                                     max_calls=450, physical_token_cap=800000)
        with patch.object(v, "config", return_value=(ROOT / "test-config.json",
                                                       ["Chart-1", "Closure-128",
                                                        "Lang-35", "Math-57", "Time-14"])):
            with patch.object(v.subprocess, "run") as fake:
                with self.assertRaisesRegex(ValueError, "require --allow-inference"):
                    v.invoke("01", True, args)
                fake.assert_not_called()


if __name__ == "__main__":
    unittest.main()

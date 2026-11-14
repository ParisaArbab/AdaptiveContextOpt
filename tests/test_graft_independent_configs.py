"""Keep Qwen independent-bug batch contract at one entity per request."""
import json
from pathlib import Path


def test_qwen_independent_configs_use_single_candidate_screening():
    root = Path(__file__).resolve().parents[1]
    for arm in ("control", "anchored"):
        config = json.loads(
            (root / f"configs/graft_independent_qwen7b_{arm}.json").read_text()
        )
        bench = config["benchmark"]
        assert bench["model"] == "qwen2.5:7b"
        assert bench["batch_size"] == 1
        assert bench["candidate_provider"] == "flexfl"
        assert bench["top_per_method"] == 10
        assert bench["candidate_count"] == 24
        assert bench["bugs"] == ["Time-1", "Lang-1", "Math-1"]
        assert bench["admission"] == "strict"
        assert config["graft"]["failure_name_anchor"] == (arm == "anchored")

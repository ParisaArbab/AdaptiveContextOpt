from dataclasses import asdict
import json
from pathlib import Path
import pytest
from test_graft_graph import fixture_graph
from rave_fl.eval.representations import prepare_entities
from rave_fl.eval.schema import file_hash
from rave_fl.graft.diagnostics import generate


def test_report_regeneration_from_saved_artifacts_without_a_provider(tmp_path, monkeypatch):
    from rave_fl.eval.backend import Provider

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline diagnostics must not invoke a provider")

    monkeypatch.setattr(Provider, "execute", forbidden)
    instance, entities, _ = fixture_graph()
    candidates, _, _ = prepare_entities(instance, entities, "l1", 2200)
    source = tmp_path / "archive"
    case = source / "Fixture-1"
    folder = case / "instances/fixture"
    folder.mkdir(parents=True)
    (folder / "input.json").write_text(
        json.dumps(
            {
                "online": instance.to_dict(),
                "entities": [asdict(e) for e in entities],
                "candidates": [asdict(c) for c in candidates],
            }
        )
    )
    (folder / "evaluation_only.json").write_text(
        json.dumps({"functions": ["x.A.a()"], "files": ["A.java"]})
    )
    (case / "results.json").write_text(
        json.dumps(
            [
                {
                    "policy": p,
                    "status": "completed",
                    "accounted_tokens": t,
                    "arm": {"ranked_functions": [c.function_id for c in candidates]},
                }
                for p, t in [("full", 1000), ("fixed_l1", 700), ("v2_utility", 800)]
            ]
        )
    )
    (case / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": "synthetic-fixture",
                    "context_window": 4096,
                    "output_tokens": 160,
                    "budget": 16000,
                }
            }
        )
    )
    before = {str(p): file_hash(p) for p in source.rglob("*.json")}
    output = tmp_path / "report"
    generated = generate(source, output)
    assert generated["new_model_calls"] == 0 and generated["archive_files_unchanged"]
    assert before == {str(p): file_hash(p) for p in source.rglob("*.json")}
    assert len(list((output / "figures").glob("*.svg"))) == 5
    assert len(list((output / "figures").glob("*.png"))) == 5
    assert (
        "Implementation Change Log and Experimental Impact"
        in (output / "RESEARCH_PROGRESS.md").read_text()
    )
    with pytest.raises(ValueError, match="preserve"):
        generate(source, output)

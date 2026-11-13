"""Portable launcher acceptance checks. All execution uses synthetic replies."""
import json
from pathlib import Path
import platform

import pytest

from rave_fl.eval.schema import CompatibilityError, safe_name
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft import portable_experiment as portable
from rave_fl.pilot import write_json
from test_graft_fixed_evidence_schedule import fixed_fixture


HOST = "local_mac" if platform.system() == "Darwin" else "remote_linux_a100"


@pytest.fixture
def portable_config(fixed_fixture, tmp_path):
    value, _ = fixed_fixture
    path = tmp_path / "config.json"
    write_json(path, value)
    return path


def freeze(path, output, *extra, profile=HOST):
    return portable.main(["preflight", "--profile", profile, "--simulate-host", "--config", str(path),
                          "--output", str(output), *extra])


def execute(command, path, cohort, output, index=0, profile=HOST, extra=()):
    return portable.main([command, "--profile", profile, "--simulate-host", "--config", str(path),
                          "--cohort", str(cohort), "--shard-index", str(index), "--output", str(output), *extra])


def test_two_profiles_shards_and_aggregate_preserve_frozen_experiment(portable_config, tmp_path):
    caps = ("--shard-count", "2", "--cohort-physical-budget", "24000", "--cohort-max-new-calls", "8")
    for host in portable.HOSTS:
        assert freeze(portable_config, tmp_path / host, *caps, profile=host) == 0
    paths = [tmp_path / host / "cohort.json" for host in portable.HOSTS]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    lock = portable.read_lock(paths[0])
    assert [s["ids"] for s in lock["shards"]] == [["synthetic-java"], ["synthetic-python"]]
    runs = [tmp_path / f"run-{i}" for i in range(2)]
    for index, host in enumerate(portable.HOSTS):
        assert execute("run", portable_config, paths[0], runs[index], index, profile=host) == 0
        assert execute("resume", portable_config, paths[0], runs[index], index, profile=host) == 0
    assert execute("run", portable_config, paths[0], tmp_path / "alternate", 0, profile="remote_linux_a100") == 0
    first = json.loads((runs[0] / "results.json").read_text())
    alternate = json.loads((tmp_path / "alternate/results.json").read_text())
    assert [a["ranked_functions"] for a in first["pairs"][0]["arms"].values()] == [
        a["ranked_functions"] for a in alternate["pairs"][0]["arms"].values()
    ]
    aggregate = tmp_path / "merged"
    summary = portable.merge(lock, list(reversed(runs)), aggregate)
    assert summary["selected_pairs"] == 2 and summary["spending"]["new_calls"] == 8
    assert summary["spending"]["physical_inference_tokens"] == 0
    assert len(list((aggregate / "figures").glob("*"))) == 10
    repeated = tmp_path / "repeated"
    portable.merge(lock, runs, repeated)
    assert (aggregate / "summary.json").read_bytes() == (repeated / "summary.json").read_bytes()
    assert (aggregate / "figures/pareto.svg").read_bytes() == (repeated / "figures/pareto.svg").read_bytes()
    with pytest.raises(CompatibilityError, match="Incomplete shard coverage"):
        portable.merge(lock, runs[:1], tmp_path / "incomplete")
    with pytest.raises(CompatibilityError, match="duplicate shard index"):
        portable.merge(lock, [runs[0], runs[0]], tmp_path / "duplicate")


def test_selection_cap_and_partition_failures(portable_config, tmp_path):
    with pytest.raises(SystemExit) as err:
        freeze(portable_config, tmp_path / "duplicate", "--bugs", "synthetic-java,synthetic-java")
    assert err.value.code == 2
    assert freeze(portable_config, tmp_path / "too-many", "--limit", "1", "--shard-count", "2") == 3
    assert freeze(portable_config, tmp_path / "cap", "--shard-count", "2",
                  "--cohort-physical-budget", "100", "--cohort-max-new-calls", "8") == 3
    assert not (tmp_path / "cap/cohort.json").exists()
    assert "Combined shard reservations" in (tmp_path / "cap/preflight.json").read_text()
    assert freeze(portable_config, tmp_path / "one", "--all", "--limit", "1", "--sample", "random", "--seed", "17") == 0
    assert freeze(portable_config, tmp_path / "one-again", "--all", "--limit", "1", "--sample", "random", "--seed", "17") == 0
    assert (tmp_path / "one/cohort.json").read_bytes() == (tmp_path / "one-again/cohort.json").read_bytes()


def test_model_labels_and_candidate_fingerprints_are_immutable(portable_config, tmp_path, monkeypatch):
    pre = tmp_path / "pre"
    assert freeze(portable_config, pre) == 0
    lock = portable.read_lock(pre / "cohort.json")
    base = json.loads(portable_config.read_text())
    config = fixed.configuration(base)
    plan, _ = fixed.prepare(portable.shard_config(config, lock["shards"][0]["ids"]))
    labels = portable.label_identity(config)
    with pytest.raises(CompatibilityError, match="condition"):
        portable.validate_shard(lock, 0, plan, "other-labels", None)
    plan["pairs"][0]["pool_sha256"] = "foreign-pool"
    with pytest.raises(CompatibilityError, match="candidate pools"):
        portable.validate_shard(lock, 0, plan, labels, None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Incompatible configuration reached execution")

    monkeypatch.setattr(fixed.GraftProvider, "execute", forbidden)
    with pytest.raises(SystemExit) as error:
        execute("run", portable_config, pre / "cohort.json", tmp_path / "foreign", extra=("--model", "other"))
    assert error.value.code == 2 and not (tmp_path / "foreign").exists()


def test_offline_inventory_keeps_missing_embedding_and_dirty_source(portable_config, tmp_path):
    config = json.loads(portable_config.read_text())
    (Path(config["embeddings_dir"]) / (safe_name("synthetic-java") + ".json")).unlink()
    checkout = Path(config["benchmark"]["dataset_file"]).parent / "checkout/example.py"
    checkout.write_text(checkout.read_text() + "# dirty fixture\n")
    target = tmp_path / "bad-input"
    assert freeze(portable_config, target) == 3
    diagnostics = json.loads((target / "preflight.json").read_text())
    assert diagnostics["new_model_calls"] == 0
    assert len(diagnostics["inventory"]) == 2
    assert all(row["status"] == "infeasible" for row in diagnostics["inventory"])
    assert any("Tracked checkout files modified" in reason for row in diagnostics["inventory"] for reason in row["reasons"])


def test_unknown_cost_is_not_retried_through_launcher(portable_config, tmp_path, monkeypatch):
    pre = tmp_path / "pre"
    assert freeze(portable_config, pre) == 0
    run = tmp_path / "unknown"
    attempts = 0

    def interrupted(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("fixture transport timed out")

    monkeypatch.setattr(fixed.GraftProvider, "execute", interrupted)
    for verb in ("run", "resume"):
        with pytest.raises(SystemExit) as error:
            execute(verb, portable_config, pre / "cohort.json", run)
        assert error.value.code == 2
    snapshot = json.loads((run / "accounting.json").read_text())
    assert attempts == 1 and snapshot["reserved_or_unknown_tokens"] == 4256


def test_restart_requires_fresh_directory_and_paid_resume_only_runs_remainder(portable_config, tmp_path, monkeypatch):
    pre = tmp_path / "pre"
    assert freeze(portable_config, pre) == 0
    run = tmp_path / "interrupted"
    execute_call = fixed.FixedGraphBackend.execute

    def interrupt(backend, candidates, action, level):
        execute_call(backend, candidates, action, level)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(fixed.FixedGraphBackend, "execute", interrupt)
        with pytest.raises(KeyboardInterrupt):
            execute("run", portable_config, pre / "cohort.json", run)
    assert json.loads((run / "accounting.json").read_text())["new_calls"] == 1
    with pytest.raises(SystemExit):
        execute("run", portable_config, pre / "cohort.json", run)
    assert execute("resume", portable_config, pre / "cohort.json", run) == 0
    saved = json.loads((run / "results.json").read_text())
    assert saved["pairs"][0]["arms"][fixed.T0]["accounting"][0]["reuse"] == "journal"
    assert saved["spending"]["new_calls"] == 8


def test_offline_model_and_host_fail_closed(portable_config, tmp_path, monkeypatch):
    other = "remote_linux_a100" if HOST == "local_mac" else "local_mac"
    with pytest.raises(CompatibilityError, match="does not match"):
        portable.read_host(other, False, "fake")
    with pytest.raises(CompatibilityError, match="only with the fake"):
        portable.read_host(other, True, "ollama")
    pre = tmp_path / "model-missing"
    assert freeze(portable_config, pre, "--encoder-model-dir", str(tmp_path / "missing")) == 3
    assert "checkpoint directory is missing" in (pre / "preflight.json").read_text()

    def forbidden(*args, **kwargs):
        raise AssertionError("Real provider instantiated")

    monkeypatch.setattr(fixed, "GraftProvider", forbidden)
    with pytest.raises(SystemExit) as error:
        portable.main(["run", "--profile", HOST, "--backend", "ollama", "--allow-inference",
                       "--config", str(portable_config), "--cohort", str(pre / "cohort.json"),
                       "--output", str(tmp_path / "live")])
    assert error.value.code == 2 and not (tmp_path / "live").exists()


def test_bug_file_and_sharded_encoder_dispatch(portable_config, tmp_path, monkeypatch):
    from rave_fl.graft import encode

    bug_file = tmp_path / "bugs.txt"
    bug_file.write_text("synthetic-python\nsynthetic-java\n")
    output = tmp_path / "vectors"
    seen = []
    monkeypatch.setattr(portable, "inspect_encoder", lambda directory: None)

    def exporter(graph, model_dir, revision, target, *, device):
        seen.append((graph.language, target.name, device, revision))

    monkeypatch.setattr(encode, "export", exporter)
    assert portable.main([
        "prepare-embeddings", "--config", str(portable_config), "--profile", HOST,
        "--bug-file", str(bug_file), "--shard-count", "2", "--shard-index", "1",
        "--embeddings-dir", str(output), "--model-dir", str(tmp_path), "--revision", "fixture-pin",
    ]) == 0
    assert seen == [("python", safe_name("synthetic-python") + ".json", "cpu", "fixture-pin")]


def test_resigned_lock_still_rejects_overlapping_shards(portable_config, tmp_path):
    from rave_fl.eval.schema import fingerprint

    pre = tmp_path / "pre"
    assert freeze(portable_config, pre, "--shard-count", "2", "--cohort-physical-budget", "24000",
                  "--cohort-max-new-calls", "8") == 0
    lock = portable.read_lock(pre / "cohort.json")
    lock.pop("sha256")
    lock["shards"][1]["ids"] = lock["shards"][0]["ids"]
    invalid = tmp_path / "invalid-lock.json"
    write_json(invalid, {**lock, "sha256": fingerprint(lock)})
    with pytest.raises(CompatibilityError, match="coverage or order"):
        portable.read_lock(invalid)

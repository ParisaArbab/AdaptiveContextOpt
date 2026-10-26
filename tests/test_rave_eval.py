"""Synthetic-only universal benchmark validation. No real LLMs or benchmark tests."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rave_fl.eval.adapters import load_instances, normalize, capabilities
from rave_fl.eval.backend import Backend, Provider, Store, ExecutionLimit, UnknownUsage
from rave_fl.eval.cli import DEFAULTS, main
from rave_fl.eval.fusion import Fusion
from rave_fl.eval.metrics import ranking_metrics, reference_patch_files, load_gold
from rave_fl.eval.policies import engine, POLICIES
from rave_fl.eval.representations import index_source, python_sketch
from rave_fl.eval.runner import run, prepare_one, run_lock
from rave_fl.eval.schema import (
    CompatibilityError,
    OnlineInstance,
    Evidence,
    CodeEntity,
    fingerprint,
    safe_name,
)
from rave_fl.pilot_types import CandidateState, Observation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy(tmp_path):
    repo = tmp_path / "buggy"
    repo.mkdir()
    (repo / "module.py").write_text(
        "class Example:\n    flag = 1\n    def calculate(self, value):\n        if value < 0:\n            raise ValueError(value)\n        return value + self.flag\n\ndef helper(value):\n    return value * 2\n"
    )
    (repo / "Other.java").write_text(
        "class Other { int calc(int value) { if (value < 0) throw new RuntimeException(); return value + 1; } }\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Immutable synthetic buggy fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    row = {
        "instance_id": "toy__repo-1",
        "repository": "toy/repo",
        "language": "python",
        "revision": sha,
        "checkout": str(repo),
        "issue": "calculate should handle a negative value",
        "gold": {"files": ["module.py"], "functions": ["module.py::Example.calculate"]},
        "patch": "GOLD_PATCH_SENTINEL",
        "hints_text": "FUTURE_HINT_SENTINEL",
    }
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"version": "fixture-v1", "instances": [row]}))
    opts = {
        **deepcopy(DEFAULTS),
        "dataset_file": str(manifest),
        "backend": "fake",
        "model": "fake:fixture",
        "run_id": "fixture",
        "output_dir": str(tmp_path / "run"),
        "candidate_count": 3,
        "context_window": 4096,
        "budget": 16000,
        "physical_budget": 80000,
        "per_bug_physical_budget": 60000,
    }
    return opts, row, repo


def test_typed_schema_and_no_gold_in_online_object(toy):
    opts, row, _ = toy
    instances, source = load_instances(opts)
    payload = json.dumps(instances[0].to_dict())
    assert (
        "GOLD_PATCH_SENTINEL" not in payload
        and "FUTURE_HINT_SENTINEL" not in payload
        and "gold" not in instances[0].to_dict()
    )
    assert load_gold(instances[0], source).functions == ("module.py::Example.calculate",)
    with pytest.raises(CompatibilityError):
        CodeEntity("x", "../outside", 1, 1, "source", 0.5)
    with pytest.raises(CompatibilityError):
        CodeEntity("x", "x.py", 1, 2, "source", float("nan"))


@pytest.mark.parametrize(
    "dataset", ["swebench-lite", "swebench-verified", "swebench-full", "swebench-multilingual"]
)
def test_standard_swe_schema(dataset, toy, tmp_path):
    opts, row, _ = toy
    r = {
        "instance_id": "org__repo-2",
        "repo": "org/repo",
        "base_commit": row["revision"],
        "problem_statement": "Observed wrong result",
        "checkout": row["checkout"],
        "language": "python",
        "patch": "GOLD_SENTINEL",
        "test_patch": "TEST_PATCH_SENTINEL",
    }
    i = normalize(r, dataset, {}, tmp_path, opts)
    assert i.issue == "Observed wrong result" and "SENTINEL" not in json.dumps(i.to_dict())


def test_authentic_multi_schema_not_generic_swe_java(toy, tmp_path):
    opts, row, _ = toy
    r = {
        "org": "org",
        "repo": "repo",
        "number": 2,
        "base": {"sha": row["revision"]},
        "language": "java",
        "resolved_issues": [{"title": "Wrong result", "body": "calc fails"}],
        "body": "FUTURE_PR_BODY",
        "fix_patch": "GOLD_SENTINEL",
        "run_result": {"bad": "POSTFIX_LOG"},
    }
    i = normalize(r, "multi-swe-bench", {}, tmp_path, opts)
    assert (
        i.repository == "org/repo"
        and i.revision == row["revision"]
        and i.instance_id == "org__repo-2"
    )
    assert all(
        s not in json.dumps(i.to_dict()) for s in ["FUTURE_PR_BODY", "GOLD_SENTINEL", "POSTFIX_LOG"]
    )
    with pytest.raises(CompatibilityError):
        normalize(
            {"instance_id": "x", "base_commit": row["revision"]},
            "multi-swe-bench",
            {},
            tmp_path,
            opts,
        )


def test_python_class_function_and_java_ast_indexing(toy):
    opts, _, _ = toy
    instance = load_instances(opts)[0][0]
    entities = index_source(instance, "method")
    assert {e.entity_id for e in entities} == {
        "module.py::Example",
        "module.py::Example.calculate",
        "module.py::helper",
    }
    levels, meta = python_sketch("x", entities[1].source)
    assert not meta["parse_error"] and "CALL" in levels[1] and "return" in levels[1]
    java = index_source(replace(instance, language="java"), "method")
    assert len(java) == 1 and "calc" in java[0].entity_id
    broken, meta = python_sketch("x", "def invalid(")
    assert meta["parse_error"]


def test_candidate_provider_is_not_repaired_from_gold(toy):
    opts, row, _ = toy
    opts["candidate_count"] = 1
    instance = load_instances(opts)[0][0]
    first = prepare_one(instance, opts)
    row["gold"] = {"files": ["not_in_repository.py"], "functions": ["secret::gold"]}
    Path(opts["dataset_file"]).write_text(json.dumps([row]))
    second = prepare_one(load_instances(opts)[0][0], opts)
    assert first[-1]["pool_hash"] == second[-1]["pool_hash"]


def test_multi_location_metrics_and_patch_filtering():
    m = ranking_metrics(["a", "a", "b", "c"], ["a", "c", "missing"])
    assert (
        m["first_relevant_rank"] == 1
        and m["Recall@3"] == pytest.approx(2 / 3)
        and m["candidate_pool_coverage"] == pytest.approx(2 / 3)
    )
    assert not m["all_gold_recovered"] and m["all_gold_rank"] is None
    patch = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-return 1\n+return 2\n"
        + "diff --git a/tests/test_a.py b/tests/test_a.py\n-x\n+y\n"
    )
    files, excluded = reference_patch_files(patch)
    assert files == ("src/a.py",) and excluded
    assert ranking_metrics([], []) is None


def test_complete_fake_cli_pipeline_all_policies_fusions_and_gold_isolation(toy):
    opts, _, _ = toy
    opts.update(
        policies=list(POLICIES),
        fusions=["shared_prior_evidence_fusion_v1", "consistent_score", "rrf", "prior_aware"],
    )
    result = run(opts)
    directory = Path(result["run_dir"])
    rows = json.loads((directory / "results.json").read_text())
    assert len(rows) == len(POLICIES) * 4
    assert all(
        r["status"] == ("failed" if r["policy"] == "ranker_disagreement" else "completed")
        for r in rows
    )
    assert (
        result["spending"]["physical_inference_tokens"] == 0
        and result["spending"]["simulated_provider_tokens"] > 0
    )
    assert (directory / "reports/evaluation.md").exists() and len(
        list((directory / "figures").glob("*.svg"))
    ) >= 6
    ledger = (directory / "raw_ledgers/requests.jsonl").read_text()
    assert "GOLD_PATCH_SENTINEL" not in ledger and "FUTURE_HINT_SENTINEL" not in ledger
    assert all(
        r["accounted_tokens"]
        == sum(c["input_tokens"] + c["output_tokens"] for c in r["arm"]["calls"])
        for r in rows
        if "arm" in r
    )
    assert result["spending"]["new_calls"] < sum(r["calls"] for r in rows)
    resumed = run({"allow_inference": False}, directory)
    assert resumed["spending"]["new_calls"] == result["spending"]["new_calls"]


def test_shell_entrypoint_synthetic_run_and_report(toy):
    opts, _, _ = toy
    path = Path(opts["output_dir"])
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/rave_fl_run.sh"),
            "run",
            "--dataset",
            "custom",
            "--dataset-file",
            opts["dataset_file"],
            "--backend",
            "fake",
            "--model",
            "fake:fixture",
            "--context-window",
            "4096",
            "--policies",
            "fixed_l1,highest_suspicion",
            "--output-dir",
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        subprocess.run(
            ["bash", str(ROOT / "scripts/rave_fl_run.sh"), "report", "--run-dir", str(path)],
            capture_output=True,
        ).returncode
        == 0
    )


def test_dry_run_never_calls_provider(monkeypatch, toy):
    opts, _, _ = toy
    opts.update(backend="ollama", dry_run=True)
    monkeypatch.setattr(Provider, "execute", lambda *a: pytest.fail("Inference in dry-run"))
    result = run(opts)
    assert result["spending"]["new_calls"] == 0
    assert set(result["statuses"].values()) == {"prepared"}


def test_missing_model_no_implicit_download(monkeypatch, toy):
    opts, _, _ = toy
    opts["backend"] = "ollama"
    p = Provider(opts)
    monkeypatch.setattr(p, "get", lambda path: {"models": []})
    with pytest.raises(CompatibilityError, match="not installed"):
        p.preflight()


def test_real_run_needs_explicit_authorization(toy):
    opts, _, _ = toy
    opts["backend"] = "ollama"
    with pytest.raises(CompatibilityError, match="inference disabled"):
        run(opts)


def backend(toy, **changes):
    opts, _, _ = toy
    opts = {**opts, **changes}
    i = load_instances(opts)[0][0]
    i, es, cs, *_ = prepare_one(i, opts)
    root = Path(opts["output_dir"])
    root.mkdir(exist_ok=True)
    p = Provider(opts)
    p.preflight()
    store = Store(root, opts)
    return Backend(i, es, p, store, opts, opts["budget"]), cs, store, p


def test_cache_changed_source_evidence_model_settings_invalidates(toy):
    b, cs, store, p = backend(toy)
    b.execute(cs[:1], __import__("rave_fl.core", fromlist=["Action"]).Action.FULL_VERIFY, 2)
    first = store.snapshot()["new_calls"]
    from rave_fl.core import Action

    b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == first
    b.execute([replace(cs[0], source=cs[0].source + "\n# new source")], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == first + 1
    b.opts = {**b.opts, "seed": 10}
    b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == first + 2


def test_strict_budget_refuses_before_revealing_cache(toy):
    b, cs, store, p = backend(toy, admission="strict", budget=1)
    from rave_fl.core import Action

    with pytest.raises(ExecutionLimit):
        b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == 0


def test_physical_limit_unknown_usage_and_resume_never_retries(toy):
    from rave_fl.core import Action

    b, cs, store, p = backend(toy, physical_budget=1)
    with pytest.raises(ExecutionLimit):
        b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == 0
    store.opts["physical_budget"] = 80000
    p.execute = lambda *a: (_ for _ in ()).throw(TimeoutError("Unknown paid request"))
    with pytest.raises(TimeoutError):
        b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["usage_unknown"] and store.snapshot()["reserved_or_unknown_tokens"] > 0
    with pytest.raises(UnknownUsage):
        b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert store.snapshot()["new_calls"] == 1


def test_malformed_reply_is_paid_and_cached(toy, monkeypatch):
    b, cs, store, p = backend(toy)
    from rave_fl.core import Action

    original = p.execute

    def bad(*args):
        r = original(*args)
        r["message"]["content"] = "{malformed"
        return r

    monkeypatch.setattr(p, "execute", bad)
    result = b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert result.error and not result.observations[0].valid and result.total_tokens > 0
    assert store.snapshot()["new_calls"] == 1


def test_fusion_no_full_scale_bonus_and_deterministic_ties(toy):
    opts, _, _ = toy
    i = load_instances(opts)[0][0]
    _, _, cs, *_ = prepare_one(i, opts)
    states = {
        c.function_id: CandidateState(1, Observation(c.function_id, 0.8, 0.2, "cheap")) for c in cs
    }
    full = deepcopy(states)
    full[cs[0].function_id].level = 2
    for name in ["consistent_score", "prior_aware", "rrf"]:
        f = Fusion(name, tie="hash")
        assert f.scores(cs, states) == f.scores(cs, full)
        assert f.order(cs, f.scores(cs, states)) == f.order(cs, f.scores(cs, states))
    assert Fusion().scores(cs, states) != Fusion().scores(cs, full)


def test_selection_filters_before_missing_unselected_evidence(toy):
    opts, row, _ = toy
    bad = {**row, "instance_id": "unused", "evidence_file": "missing.txt"}
    Path(opts["dataset_file"]).write_text(json.dumps([row, bad]))
    opts["bugs"] = [row["instance_id"]]
    assert len(load_instances(opts)[0]) == 1


def test_split_overlap_rejected(toy, tmp_path):
    opts, row, _ = toy
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"train": [row["instance_id"]], "test": [row["instance_id"]]}))
    with pytest.raises(CompatibilityError, match="overlap"):
        load_instances({**opts, "split_manifest": str(p), "split": "train"})


def test_existing_time25_exact_policy_regression_no_network(monkeypatch):
    path = ROOT / "results/rave_fl_pilot/time25_llama31_8b/run.json"
    if not path.exists():
        pytest.skip("Optional historical artifacts unavailable")
    monkeypatch.setattr(
        Provider, "execute", lambda *a: pytest.fail("Historical replay called provider")
    )
    from rave_fl.replay import ReplayCorpus
    from rave_fl.pilot_evaluation import evaluate
    from wp1.flexfl_data import read_ground_truth

    corpus = ReplayCorpus(
        path, json.loads((ROOT / "configs/rave_pilot_time25.json").read_text())["cases"][0]
    )
    metadata = {c.function_id: {"ranks": {}} for c in corpus.candidates}
    depths = {m: 10 for m in ["SBIR", "Ochiai", "BoostN"]}
    opts = {**deepcopy(DEFAULTS), "max_full": 1}
    truth = read_ground_truth(Path("references/FlexFL_Time25"), "Time-25")
    for policy, tokens, rank in [
        ("fixed_l1", 11298, 4),
        ("highest_suspicion", 12516, 1),
        ("v2_utility", 12516, 1),
    ]:
        a = engine(
            corpus.candidates, corpus.backend(), policy, opts, metadata, depths, Fusion()
        ).execute()
        assert (
            a["actual_tokens"] == tokens
            and evaluate(a["ranked_functions"], truth, a["states"])["first_relevant_rank"] == rank
        )


def test_empty_split_never_broadens_selection(toy, tmp_path):
    opts, row, _ = toy
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"development": [row["instance_id"]], "test": []}))
    with pytest.raises(CompatibilityError, match="empty"):
        load_instances({**opts, "split_manifest": str(p), "split": "test"})


def test_default_swe_language_filter(toy):
    opts, row, _ = toy
    r = {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "base_commit": row["revision"],
        "problem_statement": "Failure",
    }
    Path(opts["dataset_file"]).write_text(json.dumps([r]))
    with pytest.raises(CompatibilityError, match="empty"):
        load_instances({**opts, "dataset": "swebench-lite", "language": "java"})


def test_observed_failure_required_not_fabricated(toy):
    opts, _, _ = toy
    i = load_instances(opts)[0][0]
    with pytest.raises(CompatibilityError, match="Observed"):
        prepare_one(i, {**opts, "require_observed_failure": True})


def test_pending_request_from_crash_blocks_retry(toy):
    b, cs, store, p = backend(toy)
    from rave_fl.core import Action

    payload = b.payload(cs[:1], Action.FULL_VERIFY, 2)
    store.db.execute(
        "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
        ("pending", "id", "pending", json.dumps(payload), None, 100, None, None, None),
    )
    store.db.commit()
    with pytest.raises(UnknownUsage):
        store.request("pending", "id", p, payload, [cs[0].function_id], Action.FULL_VERIFY)
    assert store.snapshot()["new_calls"] == 1


def test_concurrent_identical_action_is_paid_once(toy):
    b, cs, store, p = backend(toy)
    from rave_fl.core import Action

    payload = b.payload(cs[:1], Action.FULL_VERIFY, 2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: store.request(
                    "same", "toy", p, payload, [cs[0].function_id], Action.FULL_VERIFY
                ),
                range(4),
            )
        )
    assert store.snapshot()["new_calls"] == 1 and sum(reuse == "fresh" for _, reuse in results) == 1


def test_provider_bound_violation_retains_accounted_paid_tokens(toy):
    b, cs, store, p = backend(toy)
    from rave_fl.core import Action

    original = p.execute

    def bad(*args):
        value = original(*args)
        value["eval_count"] = p.opts["output_tokens"] + 1
        return value

    p.execute = bad
    with pytest.raises(ExecutionLimit):
        b.execute(cs[:1], Action.FULL_VERIFY, 2)
    assert b.spent == store.snapshot()["simulated_provider_tokens"] and b.spent > 0
    assert store.snapshot()["usage_unknown"]
    with pytest.raises(UnknownUsage):
        b.execute(cs[1:2], Action.FULL_VERIFY, 2)


def test_shared_cache_and_digest_invalidation(toy, tmp_path):
    cache = str(tmp_path / "shared")
    b, cs, s, p = backend(toy, cache_dir=cache)
    from rave_fl.core import Action

    b.execute(cs[:1], Action.FULL_VERIFY, 2)
    b2, cs2, s2, p2 = backend(toy, cache_dir=cache, output_dir=str(tmp_path / "second"))
    b2.execute(cs2[:1], Action.FULL_VERIFY, 2)
    assert s2.snapshot()["new_calls"] == 0 and b2.spent == b.spent
    p2.metadata["digest"] = "changed-model"
    b2.execute(cs2[:1], Action.FULL_VERIFY, 2)
    assert s2.snapshot()["new_calls"] == 1


def test_resume_checks_completed_artifact_and_dataset_before_mutation(toy):
    opts, _, _ = toy
    opts["policies"] = ["fixed_l1"]
    directory = Path(run(opts)["run_dir"])
    target = next((directory / "instances").glob("*/*__*.json"))
    target.write_text("{}")
    before = (directory / "execution_status.json").read_bytes()
    with pytest.raises(CompatibilityError, match="checksum"):
        run({}, directory)
    assert before == (directory / "execution_status.json").read_bytes()


def test_frozen_reranking_never_calls_provider_and_compare_all_runs(toy, tmp_path, monkeypatch):
    opts, _, _ = toy
    opts["policies"] = ["fixed_l1"]
    source = Path(run(opts)["run_dir"])
    monkeypatch.setattr(
        Provider, "execute", lambda *a: pytest.fail("Frozen reranking invoked provider")
    )
    monkeypatch.setattr(
        Provider, "preflight", lambda *a: pytest.fail("Frozen reranking contacted provider")
    )
    frozen = {
        **opts,
        "output_dir": str(tmp_path / "frozen"),
        "backend": "ollama",
        "evaluation_mode": "frozen",
        "observations_dir": str(source),
        "fusions": ["consistent_score", "prior_aware"],
    }
    target = Path(run(frozen)["run_dir"])
    assert json.loads((target / "execution_status.json").read_text())["spending"]["new_calls"] == 0
    from rave_fl.eval.reporting import compare

    compared = compare([source, source, source], tmp_path / "comparison")
    assert compared["paired_rows"] == 4
    with pytest.raises(CompatibilityError, match="Incompatible"):
        compare([source, target], tmp_path / "bad_compare")
    assert not (tmp_path / "bad_compare").exists()


def test_project_split_groups_never_overlap_and_time25_development_only(tmp_path):
    from rave_fl.eval.splits import make_splits

    p = tmp_path / "rows.json"
    p.write_text(
        json.dumps(
            [{"bug": i} for i in ["Time-25", "Time-1", "Lang-1", "Math-2", "Chart-1", "Closure-1"]]
        )
    )
    splits, _ = make_splits(p, "defects4j")
    assert {"Time-25", "Time-1"} <= set(splits["development"])
    seen = {}
    for name, ids in splits.items():
        for ident in ids:
            project = ident.split("-")[0]
            assert project not in seen or seen[project] == name
            seen[project] = name


def test_mock_a100_readonly_preflight(toy, monkeypatch):
    from rave_fl.eval.preflight import inspect
    import rave_fl.eval.preflight as module

    opts, _, _ = toy
    opts["device"] = "a100"
    which = module.shutil.which
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/mock/nvidia-smi"
        if name == "nvidia-smi"
        else None
        if name == "java"
        else which(name),
    )
    original = module.subprocess.run

    def invoke(args, **kwargs):
        if args[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                args, 0, "NVIDIA A100, MOCK_DRIVER, 40960, 40000", ""
            )
        return original(args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", invoke)
    result = inspect(opts)
    assert "MOCK_DRIVER" in result["nvidia_gpus"] and result["inference_performed"] is False
    assert result["resource_estimate"]["configured_physical_token_cap"] == opts["physical_budget"]


def test_missing_checkout_retained_as_failed_attempt(toy):
    opts, row, repo = toy
    row["checkout"] = str(repo / "absent")
    Path(opts["dataset_file"]).write_text(json.dumps([row]))
    opts["policies"] = ["fixed_l1"]
    directory = Path(run(opts)["run_dir"])
    rows = json.loads((directory / "results.json").read_text())
    assert (
        len(rows) == 1
        and rows[0]["status"] == "failed"
        and rows[0]["metrics"]["file"]["MRR"] is None
    )
    summary = json.loads((directory / "summary.json").read_text())
    assert summary[0]["n"] == 1 and summary[0]["MRR"] == 0


def test_paired_uncertainty_requires_independent_defects_and_skips_synthetic():
    from rave_fl.eval.reporting import paired_uncertainty

    rows = []
    for n in range(12):
        for policy, value in [("fixed_l1", 0.5), ("highest_suspicion", 0.6)]:
            rows.append(
                {
                    "instance_id": f"bug-{n}",
                    "independent_bug_id": f"bug-{n}",
                    "repository": f"project-{n%3}",
                    "language": "python",
                    "policy": policy,
                    "fusion": "rrf",
                    "status": "completed",
                    "metrics": {"file": {"MRR": value}},
                }
            )
    result = paired_uncertainty(rows, 42)
    assert (
        len(result) == 1
        and result[0]["independent_bugs"] == 12
        and result[0]["delta_MRR"] == pytest.approx(0.1)
    )
    assert result[0]["method"] == "paired repository-cluster bootstrap"
    correlated = [{**r, "independent_bug_id": "one-bug"} for r in rows]
    assert paired_uncertainty(correlated) == []
    assert paired_uncertainty([{**r, "synthetic": True} for r in rows]) == []


def test_config_rejects_quoted_authorization_and_invalid_enums(toy, tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"allow_inference": "false"}))
    assert main(["preflight", "--config", str(p)]) == 2
    p.write_text(json.dumps({"evaluation_mode": "unimplemented"}))
    assert main(["preflight", "--config", str(p)]) == 2


def test_imported_bounds_file_pool_and_file_only_metrics(toy, tmp_path):
    from rave_fl.eval.candidates import imported
    from rave_fl.eval.metrics import evaluate
    from rave_fl.eval.schema import GoldLabels

    opts, _, repo = toy
    i = load_instances(opts)[0][0]
    p = tmp_path / "candidates.json"
    p.write_text(
        json.dumps([{"entity_id": "module.py", "path": "module.py", "kind": "file", "prior": 0.5}])
    )
    entities = imported(i, p)
    metrics = evaluate(
        {"ranked_functions": ["module.py"]},
        entities,
        GoldLabels(("module.py",), ("secret-function",)),
        "custom",
    )
    assert metrics["file"]["Top-1"] == 1 and metrics["method"] is None
    p.write_text(json.dumps([{"path": "module.py", "end_line": 999}]))
    with pytest.raises(CompatibilityError, match="outside"):
        imported(i, p)


def test_unregistered_language_and_incompatible_representation_fail(toy):
    opts, _, _ = toy
    i = load_instances(opts)[0][0]
    with pytest.raises(CompatibilityError, match="extension"):
        index_source(replace(i, language="unknown"), "file")
    with pytest.raises(CompatibilityError, match="file_compact"):
        prepare_one(i, {**opts, "representation": "file_compact"})


def test_model_specific_reservation_rejects_wrong_model_digest(toy):
    with pytest.raises(CompatibilityError, match="model/digest"):
        backend(
            toy,
            admission="strict",
            reservation_spec=str(ROOT / "configs/rave_v3_token_reservation.json"),
        )


def test_explicit_manifest_skip_is_preserved(toy):
    opts, row, _ = toy
    row["skip_reason"] = "Fixture intentionally unavailable; never replace with another bug"
    Path(opts["dataset_file"]).write_text(json.dumps([row]))
    opts["policies"] = ["fixed_l1"]
    result = run(opts)
    assert set(result["statuses"].values()) == {"skipped"} and result["spending"]["new_calls"] == 0
    rows = json.loads((Path(result["run_dir"]) / "results.json").read_text())
    assert rows[0]["status"] == "skipped" and "intentionally unavailable" in rows[0]["error"]


def test_disagreement_runs_only_with_actual_imported_rank_positions(toy, tmp_path):
    opts, row, _ = toy
    i = load_instances(opts)[0][0]
    entities = index_source(i, "method")
    records = []
    for n, e in enumerate(entities, 1):
        value = asdict(e)
        value.pop("source")
        value["ranks"] = {"SBIR": n, "Ochiai": len(entities) + 1 - n, "BoostN": n}
        records.append(value)
    p = tmp_path / "ranked.json"
    p.write_text(json.dumps(records))
    row["candidate_file"] = str(p)
    Path(opts["dataset_file"]).write_text(json.dumps([row]))
    result = run({**opts, "candidate_provider": "imported", "policies": ["ranker_disagreement"]})
    assert set(result["statuses"].values()) == {"completed"}

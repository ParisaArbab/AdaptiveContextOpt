"""Offline checks for pinned Java sources, ranker spans and independent method gold."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from rave_fl.eval.schema import CodeEntity, CompatibilityError
from rave_fl.pilot import write_json
from scripts.verify_graft_real_sources import candidates_verified, changed_methods, method_spans, verify


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", *args],
        text=True,
    ).strip()


def commit(root):
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def offline_bundle(tmp_path):
    checkout = tmp_path / "checkout"
    source = checkout / "src/x/A.java"
    source.parent.mkdir(parents=True)
    git(checkout, "init", "-q")
    buggy_source = """package x;
class A {
  int f(int n) {
    return n - 1;
  }
  int f(String s) {
    return 2;
  }
}
"""
    source.write_text(buggy_source)
    buggy = commit(checkout)
    source.write_text(buggy_source.replace("n - 1", "n + 1"))
    fixed = commit(checkout)
    git(checkout, "checkout", "-q", "--detach", buggy)

    metadata = tmp_path / "metadata"
    table = metadata / "framework/projects/Lang/active-bugs.csv"
    table.parent.mkdir(parents=True)
    git(metadata, "init", "-q")
    table.write_text("bug.id,revision.id.buggy,revision.id.fixed\n" + f"54,{buggy},{fixed}\n")
    metadata_sha = commit(metadata)
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    git(rankings, "init", "-q")
    for name in ("SBIR", "Ochiai", "BoostN"):
        ranking = rankings / f"data/FL_results/{name}/Defects4J/Lang-54_method-susps.csv"
        ranking.parent.mkdir(parents=True)
        ranking.write_text("File,Signature,StartLine,EndLine\nx.A,f(int),3,5\nx.A,f(String),6,8\n")
    labels = rankings / "data/input/ground_truth/Defects4J/gt.json"
    labels.parent.mkdir(parents=True)
    write_json(labels, {"Lang-54": ["x.A.f(int)"]})
    flexfl_sha = commit(rankings)
    online_rankings = tmp_path / "online_rankings"
    shutil.copytree(rankings / "data/FL_results", online_rankings / "data/FL_results")

    dataset = tmp_path / "instances.json"
    write_json(dataset, {"instances": [{
        "instance_id": "Lang-54", "repository": "synthetic-fixture", "language": "java",
        "revision": buggy, "checkout": str(checkout), "source_root": "src",
        "flexfl_repo": str(online_rankings),
        "evidence": {"kind": "observed_failure", "text": "Synthetic test failed"},
    }]})
    config = tmp_path / "config.json"
    write_json(config, {"benchmark": {"dataset": "defects4j", "dataset_file": str(dataset),
                                     "candidate_provider": "flexfl", "backend": "fake"},
                        "embeddings_dir": str(tmp_path / "unused-vectors")})
    return SimpleNamespace(config=config, metadata=metadata, metadata_revision=metadata_sha,
                           flexfl_reference=rankings, flexfl_revision=flexfl_sha,
                           historical_labels=rankings, bug="Lang-54", output=tmp_path / "audit"), checkout, buggy, fixed


def test_pinned_candidate_and_gold_audits_are_separate(offline_bundle):
    args, _, buggy, _ = offline_bundle
    assert verify(args) == 0
    candidates = json.loads((args.output / "candidate-audit.json").read_text())
    assert candidates["buggy_revision"] == buggy
    assert len(candidates["verified_spans"]) == 2
    assert candidates["trigger_replay_status"] == "requires_separate_verified_run"
    assert "gold" not in candidates and "functions" not in candidates
    labels = json.loads((args.output / "offline-gold.json").read_text())["instances"][0]["gold"]
    assert labels["functions"] == ["x.A.f(int)"]


def test_archival_source_audit_does_not_mark_archived_evidence_as_reproduced(offline_bundle):
    args, _, _, _ = offline_bundle
    path = args.config.parent / "instances.json"
    dataset = json.loads(path.read_text())
    dataset["instances"][0]["evidence"]["kind"] = "trigger_description"
    write_json(path, dataset)
    with pytest.raises(CompatibilityError, match="Observed buggy failure evidence required"):
        verify(args)
    args.archival_only = True
    assert verify(args) == 0
    audit = json.loads((args.output / "candidate-audit.json").read_text())
    assert audit["evidence_kind"] == "trigger_description"
    assert audit["trigger_replay_status"] == "requires_separate_verified_run"


def test_changed_pinned_ranking_fails_before_output(offline_bundle):
    args, _, _, _ = offline_bundle
    path = args.config.parent / "online_rankings/data/FL_results/Ochiai/Defects4J/Lang-54_method-susps.csv"
    path.write_text(path.read_text().replace("f(int)", "f(long)"))
    with pytest.raises(CompatibilityError, match="pinned Ochiai"):
        verify(args)
    assert not args.output.exists()


def test_online_flexfl_tree_cannot_carry_gold(offline_bundle):
    args, _, _, _ = offline_bundle
    (args.config.parent / "online_rankings/data/input/ground_truth").mkdir(parents=True)
    with pytest.raises(CompatibilityError, match="without the FlexFL ground-truth"):
        verify(args)


def test_same_name_wrong_overload_span_is_rejected(offline_bundle):
    _, checkout, _, _ = offline_bundle
    source = (checkout / "src/x/A.java").read_text()
    candidate = CodeEntity("x.A.f(int)", "src/x/A.java", 6, 8,
                           "\n".join(source.splitlines()[5:8]), 1.0, kind="method")
    assert len(method_spans(source)) == 2
    with pytest.raises(CompatibilityError, match="signature mismatch"):
        candidates_verified(SimpleNamespace(checkout=checkout), [candidate])


def test_patch_outside_method_requires_review(offline_bundle):
    _, checkout, buggy, _ = offline_bundle
    target = checkout / "src/x/A.java"
    target.write_text(target.read_text().replace("class A {", "class A {\n  int state = 0;"))
    fixed = commit(checkout)
    with pytest.raises(CompatibilityError, match="outside a unique buggy method"):
        changed_methods(checkout, buggy, fixed, "src")

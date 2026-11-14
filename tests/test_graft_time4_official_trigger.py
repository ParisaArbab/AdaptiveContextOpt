"""Regression: official Time-4 benchmark trigger exists only in fixed TEST revision.

These tests create an isolated toy Git repository. They do not invoke Qwen,
Defects4J, any network or the user's immutable Time-4 checkout.
"""
from pathlib import Path
import subprocess

import pytest

import scripts.prepare_graft_time4_authentic_test as prep


def git(root, *args):
    r = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30,
    )
    return r.stdout.strip()


def fixture_official_trigger(repo: Path):
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test Runner")
    git(repo, "config", "user.email", "test@example.invalid")
    test = repo / prep.OFFICIAL_TEST_PATH
    test.parent.mkdir(parents=True)
    test.write_text(
        "package org.joda.time;\n"
        "class TestPartial_Basics {\n"
        "  public void testWith2() { }\n"
        "}\n"
    )
    git(repo, "add", prep.OFFICIAL_TEST_PATH)
    git(repo, "commit", "-qm", "buggy with no new trigger")
    buggy = git(repo, "rev-parse", "HEAD")

    test.write_text(
        "package org.joda.time;\n"
        "class TestPartial_Basics {\n"
        "  public void testWith2() { }\n"
        "  public void testWith3() {\n"
        "    try {\n"
        "      test.with(DateTimeFieldType.clockhourOfDay(), 6);\n"
        "      fail();\n"
        "    } catch (IllegalArgumentException ex) {}\n"
        "  }\n"
        "}\n"
    )
    git(repo, "add", prep.OFFICIAL_TEST_PATH)
    git(repo, "commit", "-qm", "official trigger added in fixed test source")
    fixed = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--quiet", buggy)
    return buggy, fixed, test


def test_official_time4_trigger_comes_only_from_pinned_test_revision(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "joda"
    buggy, fixed, path = fixture_official_trigger(repo)
    monkeypatch.setattr(prep, "TIME4_BUGGY_REVISION", buggy)
    monkeypatch.setattr(prep, "TIME4_FIXED_TEST_REVISION", fixed)
    original = path.read_bytes()
    method, provenance = prep._verified_official_time4_test(
        repo, buggy, prep.OFFICIAL_TEST_ID,
    )
    assert b"public void testWith3()" in method
    assert b"catch (IllegalArgumentException ex)" in method
    assert b"testWith2" not in method
    assert path.read_bytes() == original
    assert not provenance["raw_buggy_revision_contains_test_method"]
    assert provenance["test_source_revision"] == fixed
    assert provenance["fixed_production_source_read"] is False
    assert provenance["buggy_checkout_modified"] is False
    assert provenance["new_test_execution_performed"] is False


def test_no_fallback_to_other_method_or_unpinned_revision(tmp_path, monkeypatch):
    repo = tmp_path / "joda"
    buggy, fixed, _ = fixture_official_trigger(repo)
    monkeypatch.setattr(prep, "TIME4_BUGGY_REVISION", buggy)
    monkeypatch.setattr(prep, "TIME4_FIXED_TEST_REVISION", fixed)
    with pytest.raises(ValueError, match="Only the pinned"):
        prep._verified_official_time4_test(
            repo, buggy, "org.joda.time.TestPartial_Basics::testWith3a"
        )
    with pytest.raises(ValueError, match="Only the pinned"):
        prep._verified_official_time4_test(
            repo, fixed, prep.OFFICIAL_TEST_ID
        )
    monkeypatch.setattr(prep, "TIME4_FIXED_TEST_REVISION", "f" * 40)
    with pytest.raises(FileNotFoundError, match="not in the local Git"):
        prep._verified_official_time4_test(
            repo, buggy, prep.OFFICIAL_TEST_ID
        )

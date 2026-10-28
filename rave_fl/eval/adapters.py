"""Local exports of distinct benchmark schemas; never automatically download."""
from dataclasses import asdict
import json
from pathlib import Path
import random

from rave_fl.eval.schema import (
    Capabilities,
    CompatibilityError,
    Evidence,
    OnlineInstance,
    file_hash,
    safe_name,
)

DATASETS = {
    "defects4j": ("official local Defects4J metadata", Capabilities(("java",))),
    "swebench-lite": ("SWE-bench/SWE-bench_Lite", Capabilities(("python",))),
    "swebench-verified": ("SWE-bench/SWE-bench_Verified", Capabilities(("python",))),
    "swebench-full": ("SWE-bench/SWE-bench", Capabilities(("python",))),
    "swebench-multilingual": (
        "SWE-bench/SWE-bench_Multilingual",
        Capabilities(("java", "python", "other: file/full only")),
    ),
    "multi-swe-bench": (
        "ByteDance-Seed/Multi-SWE-bench (native org/repo/base schema)",
        Capabilities(("java", "other: file/full only")),
    ),
    "custom": (
        "Local normalized JSON/JSONL manifest",
        Capabilities(("java", "python", "other: file/full only")),
    ),
}


def capabilities():
    from rave_fl.eval.policies import POLICIES

    return {
        k: {
            "source": source,
            "adapter_status": "Implemented and fixture-tested; external data required",
            **asdict(cap),
            "span_validation": "Implemented: checkout line bounds; Java/Python native AST indexing; imported spans are user supplied",
            "sketch_support": {
                "java": "Implemented and fixture-tested",
                "python": "Implemented and fixture-tested",
                "other": "Unsupported AST; file/full only for registered extensions",
            },
            "policy_support": {
                p: (
                    "Requires Java/Python and original SBIR/Ochiai/BoostN ranks"
                    if p == "ranker_disagreement"
                    else "Java/Python; Full also supports registered file/full languages"
                )
                for p in POLICIES
            },
            "full_inference": "Implemented; external provider/model and explicit authorization required",
            "replay": "Exact request cache; legacy conditional replay for matching Time-25 inputs",
            "macOS": "Fixture-tested; live model experiments not run for this implementation",
            "a100_linux": "Portable code and mock preflight tested; no A100 hardware measurement",
            "benchmark_harness": "External dependency; no builds, test suites, patch generation or official issue-resolution evaluator invoked",
        }
        for k, (source, cap) in DATASETS.items()
    }


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise CompatibilityError(
            f"Dataset file unavailable: {path}. Supply --dataset-file; no automatic download."
        )
    if path.suffix == ".jsonl":
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        meta = {}
    else:
        data = json.loads(path.read_text())
        meta = data if isinstance(data, dict) else {}
        rows = data.get("instances", data.get("cases", [])) if isinstance(data, dict) else data
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise CompatibilityError("Dataset must contain JSON object rows")
    return rows, meta


def resolved(value, base):
    return str((base / str(value)).resolve()) if value else ""


def normalize(row, dataset, meta, base, opts):
    if dataset not in DATASETS:
        raise CompatibilityError("Unknown dataset: " + dataset)
    ident = row.get("instance_id", row.get("bug", ""))
    if dataset == "multi-swe-bench":
        required = {"org", "repo", "number", "base"}
        if not required <= row.keys() or not isinstance(row["base"], dict):
            raise CompatibilityError(
                "Multi-SWE-bench requires native org, repo, number and base.sha; it is not the SWE-bench schema"
            )
        ident = ident or f"{row['org']}__{row['repo']}-{row['number']}"
        repository = row["org"] + "/" + row["repo"]
        revision = row["base"].get("sha", "")
        issues = row.get("resolved_issues", [])
        issue = "\n\n".join(
            str(i.get("title", "")) + "\n" + str(i.get("body", ""))
            for i in issues
            if isinstance(i, dict)
        )
        issue = row.get("problem_statement", issue)
        # PR body, fix_patch, hints, run_result and fix-run logs are not online evidence.
    elif dataset.startswith("swebench-"):
        if not {"instance_id", "repo", "base_commit", "problem_statement"} <= row.keys():
            raise CompatibilityError(
                "SWE-bench export needs instance_id, repo, base_commit, problem_statement"
            )
        repository = row["repo"]
        revision = row["base_commit"]
        issue = row["problem_statement"]
    else:
        repository = row.get("repository", row.get("repo", row.get("bug", "custom")))
        revision = row.get("revision", row.get("buggy_revision", ""))
        issue = row.get("issue", row.get("bug_report", ""))
    language = row.get("language", meta.get("language", opts.get("source_language")))
    if not language and dataset == "defects4j":
        language = "java"
    if not language and dataset in {"swebench-lite", "swebench-verified", "swebench-full"}:
        language = "python"
    if not language:
        raise CompatibilityError(
            "Language not recorded: supply --source-language for a language-specific native export"
        )
    language = str(language).lower()
    evidence_file = row.get("evidence_file", "")
    evidence = row.get("evidence")
    if evidence_file:
        path = Path(resolved(evidence_file, base))
        text = path.read_text()
        kind = row.get("evidence_kind", "observed_failure")
        source = str(path)
        if row.get("evidence_format") == "defects4j-compressed":
            from rave_fl.failure_evidence import compress_failures

            if not row.get("project_prefix"):
                raise CompatibilityError(
                    "Defects4J compression requires project_prefix; use raw text for unsupported failure types"
                )
            text = compress_failures(text, row["project_prefix"], row.get("max_frames", 3))
        elif row.get("evidence_format", "text") != "text":
            raise CompatibilityError(
                "Unsupported evidence_format; use text or defects4j-compressed for observed local logs"
            )
    elif isinstance(evidence, dict):
        text = evidence.get("text", "")
        kind = evidence.get("kind", "")
        source = evidence.get("source", "inline")
    else:
        text = issue
        kind = "issue_only"
        source = "dataset issue description"
    if not text and dataset == "defects4j" and row.get("flexfl_repo"):
        from rave_fl.compare_expanded import load_context

        text, provenance = load_context(Path(resolved(row["flexfl_repo"], base)), ident, None)
        kind = "trigger_description"
        source = str(provenance.get("path", "FlexFL trigger text"))
    checkout = row.get("checkout", row.get("repository_checkout", ""))
    if not checkout and opts.get("checkouts_dir"):
        checkout = str(Path(opts["checkouts_dir"]).resolve() / safe_name(ident))
    else:
        checkout = resolved(checkout, base)
    if dataset.startswith("swebench") or dataset == "multi-swe-bench":
        if not revision:
            raise CompatibilityError("Official SWE-family instance has no base commit SHA")
    return OnlineInstance(
        dataset,
        str(
            opts.get("dataset_version")
            or meta.get("version")
            or row.get("dataset_version")
            or "local-export"
        ),
        str(ident),
        str(repository),
        language,
        str(revision),
        checkout,
        str(issue),
        Evidence(kind, str(text), source),
        resolved(row.get("candidate_file", ""), base),
        resolved(row.get("flexfl_repo", ""), base),
        opts.get("source_root") or row.get("source_root", meta.get("source_root", ".")),
        {
            "schema_source": DATASETS[dataset][0],
            "evidence_is_observed": kind == "observed_failure",
            "independent_bug_id": str(row.get("defect_id", row.get("correlation_id", ident))),
            "skip_reason": str(row.get("skip_reason", "")),
        },
    )


def load_instances(opts):
    dataset = opts["dataset"]
    path = opts.get("dataset_file")
    if dataset == "defects4j" and not path:
        path = "configs/rave_pilot_time25.json"
    if not path:
        raise CompatibilityError(
            "Supply a local --dataset-file JSON/JSONL export. Downloads and inference are never implicit."
        )
    path = Path(path).resolve()
    rows, meta = read_rows(path)
    # Existing historical paths are relative to the repository cwd; normalized
    # manifests use their own directory. This distinction is explicitly recorded.
    base = (
        Path.cwd()
        if meta.get("path_base") == "cwd" or ("cases" in meta and "instances" not in meta)
        else path.parent
    )

    def row_id(row):
        return row.get("instance_id", row.get("bug", "")) or (
            f"{row.get('org')}__{row.get('repo')}-{row.get('number')}"
            if dataset == "multi-swe-bench"
            else ""
        )

    ids = [row_id(r) for r in rows]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise CompatibilityError("Missing or duplicate dataset instance IDs")
    requested = set(opts.get("bugs", []))
    explicit_selection = bool(requested)
    if requested - set(ids):
        raise CompatibilityError(
            "Requested IDs absent from export: " + ", ".join(sorted(requested - set(ids)))
        )
    if opts.get("split_manifest"):
        split = json.loads(Path(opts["split_manifest"]).read_text())
        if not opts.get("split") or opts["split"] not in split:
            raise CompatibilityError("Select a named --split from the split manifest")
        groups = [set(v) for v in split.values()]
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
            raise CompatibilityError("Train/development/calibration/test instance groups overlap")
        declared = set().union(*groups)
        if declared - set(ids):
            raise CompatibilityError("Split manifest contains unknown IDs")
        chosen = set(split[opts["split"]])
        requested = requested & chosen if explicit_selection else chosen
        explicit_selection = True
    eligible = []
    for row in rows:
        ident = row_id(row)
        if explicit_selection and ident not in requested:
            continue
        lang = row.get("language", meta.get("language", opts.get("source_language")))
        if not lang and dataset == "defects4j":
            lang = "java"
        if not lang and dataset in {"swebench-lite", "swebench-verified", "swebench-full"}:
            lang = "python"
        if lang and opts.get("language") and lang.lower() != opts["language"]:
            continue
        repo = (
            (row.get("org", "") + "/" + row.get("repo", ""))
            if dataset == "multi-swe-bench"
            else row.get("repository", row.get("repo", row.get("bug", "custom")))
        )
        if opts.get("repository") and repo != opts["repository"]:
            continue
        if opts.get("projects") and ident.split("-")[0] not in opts["projects"]:
            continue
        eligible.append(row)
    eligible.sort(key=row_id)
    if opts.get("sample") == "random":
        random.Random(opts["seed"]).shuffle(eligible)
    if opts.get("limit") is not None:
        eligible = eligible[: opts["limit"]]
    instances = [normalize(row, dataset, meta, base, opts) for row in eligible]
    # Keep the already-frozen first/random order.
    if not instances:
        raise CompatibilityError("Selection is empty; verify IDs/language/repository filters")
    return (
        instances,
        {
            "dataset_file": str(path),
            "dataset_sha256": file_hash(path),
            "path_base": str(base),
            "selection": {
                "ids": [i.instance_id for i in instances],
                "seed": opts["seed"],
                "sample": opts.get("sample", "first"),
            },
            "dataset_release": opts.get("dataset_version") or meta.get("version", "local-export"),
        },
    )

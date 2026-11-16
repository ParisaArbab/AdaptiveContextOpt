"""Seed an independent Phase-3 run with verified prior Qwen responses.

The old run is read-only. This script never calls the provider, never copies the
SQLite journal, and never claims copied responses are new physical Qwen calls.
Cache entries are still checked against their *exact* request identity by Store
at use time. Abort on incompatible cohort/configuration or corrupted responses.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from rave_fl.eval.schema import CompatibilityError, fingerprint


FROZEN_REQUEST_INPUTS = (
    "dataset", "dataset_file", "bugs", "sample", "seed", "candidate_provider",
    "candidate_count", "granularity", "representation", "sketch_chars",
    "max_source_files", "max_source_bytes", "model", "backend", "endpoint",
    "device", "context_window", "output_tokens", "temperature", "batch_size",
    "top_per_method", "rrf_k", "tie", "rounding", "require_observed_failure",
)


def seed(source_run: Path, destination: Path, *, dataset: Path, config: dict) -> dict:
    source_run = source_run.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite seeded cache: {destination}")
    if destination.is_relative_to(source_run):
        raise CompatibilityError("Seed must be a separate run, not inside original run")

    source_manifest = json.loads((source_run / "manifest.json").read_text())
    prior_opts = source_manifest["config"]["benchmark"]
    new_opts = config["benchmark"]
    for key in FROZEN_REQUEST_INPUTS:
        if prior_opts[key] != new_opts[key]:
            raise CompatibilityError(
                f"Cache seed changes request input {key!r}; do not reuse this run"
            )
    dataset_hash = sha256(dataset.read_bytes()).hexdigest()
    if source_manifest["dataset_source"]["dataset_sha256"] != dataset_hash:
        raise CompatibilityError("Seed source and new manifest SHA-256 mismatch")

    spending = json.loads((source_run / "accounting.json").read_text())
    if (
        spending.get("synthetic") is not False
        or spending.get("usage_unknown")
        or spending.get("pending")
        or not spending.get("new_calls")
    ):
        raise CompatibilityError("Seed source must be a completed-accounting real-model run")
    environment = json.loads((source_run / "environment.json").read_text())
    digest = environment["provider"]["digest"]
    declared_cache = prior_opts.get("cache_dir")
    old_cache = (
        Path(declared_cache).resolve()
        if declared_cache else source_run / "cache"
    )
    # Allow only the two immutable archive layouts used by this project,
    # never arbitrary config-controlled external directories.
    if old_cache not in {source_run / "cache", source_run.parent / "cache"}:
        raise CompatibilityError(
            "Source cache must be under its archived run or its experiment folder"
        )
    if not old_cache.is_dir():
        raise CompatibilityError(f"Archived provider cache absent: {old_cache}")
    files = sorted(old_cache.glob("*.json"))
    if not files:
        raise CompatibilityError("Source run contains no Qwen response cache")

    # Validate *all* files before copying any of them, fail closed.
    audited = []
    for source in files:
        if not re.fullmatch(r"[0-9a-f]{64}", source.stem):
            raise CompatibilityError("Non-fingerprint cache filename")
        cache = json.loads(source.read_text())
        if set(cache) != {"identity", "response", "response_sha256"}:
            raise CompatibilityError(f"Cache entry missing identity/response: {source}")
        identity = cache["identity"]
        response = cache["response"]
        payload = identity["request"]
        opts = payload["options"]
        if (
            identity["fingerprint"] != source.stem
            or identity["model_digest"] != digest
            or payload["model"] != new_opts["model"]
            or opts["num_ctx"] != new_opts["context_window"]
            or opts["seed"] != new_opts["seed"]
            or opts["temperature"] != new_opts["temperature"]
            or response.get("done") is not True
            or type(response.get("prompt_eval_count")) is not int
            or type(response.get("eval_count")) is not int
            or not 0 <= response["prompt_eval_count"] <= opts["num_ctx"]
            or not 0 <= response["eval_count"] <= opts["num_predict"]
            or fingerprint(response) != cache["response_sha256"]
        ):
            raise CompatibilityError(f"Unverified Qwen cache response: {source.name}")
        audited.append({
            "file": source.name,
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "provider_tokens_from_original_call": (
                response["prompt_eval_count"] + response["eval_count"]
            ),
        })

    destination.mkdir(parents=True)
    for item in audited:
        filename = item["file"]
        shutil.copyfile(old_cache / filename, destination / filename)
        if sha256((destination / filename).read_bytes()).hexdigest() != item["source_sha256"]:
            raise CompatibilityError("Cache copy hash mismatch: " + filename)

    provenance = {
        "schema": "graft-phase3-exact-cache-seed-v1",
        "source_run": str(source_run),
        "source_verified_cache_dir": str(old_cache),
        "source_run_manifest_sha256": sha256(
            (source_run / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_dataset_sha256": dataset_hash,
        "source_model_digest": digest,
        "source_physical_calls": spending["new_calls"],
        "source_physical_tokens": spending["physical_inference_tokens"],
        "seeded_response_files": len(audited),
        "seeded_provider_tokens_historical_not_new": sum(
            x["provider_tokens_from_original_call"] for x in audited
        ),
        "new_provider_calls_during_seeding": 0,
        "original_sqlite_journal_copied": False,
        "cache_entries_are_exact_request_checked_by_backend": True,
        "cached": audited,
    }
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    provenance = seed(
        args.source_run, args.destination, dataset=args.dataset, config=config
    )
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "Exact Qwen responses copied:",
        provenance["seeded_response_files"],
        "; new Qwen calls: 0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

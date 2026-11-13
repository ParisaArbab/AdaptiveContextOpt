"""Prepare, execute and audit independently sealed A100 groups."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import fcntl
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit

from rave_fl.eval.runner import implementation_identity
from rave_fl.eval.schema import file_hash, fingerprint
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.fixed_evidence_reporting import audit_saved_run, write_csv, write_report
from rave_fl.graft.linux_runtime import nvidia_inventory
from rave_fl.graft.ollama_live import AttestedOllama
from rave_fl.graft.portable_experiment import binding, read_lock, validate_shard
from rave_fl.graft.prompt_settlement import ARM_CAPS, BGE_FILE_MAP_SHA256, CAPS, MODEL, MODEL_DIGEST, SettlementCertificate, require
from rave_fl.pilot import write_json
from scripts.audit_graft_schedule_tokens import verify_local_inputs
from scripts.build_graft_settlement_certificate import build_schedule
from scripts.graft_server.data import ERRORS, prepare_one_defect, rank_blob
from scripts.graft_server.selection import D4J, ROOT, git, read_selection

ARGS = ("defects4j", "encoder", "tokenizer", "binary", "models", "java_home",
        "flexfl", "perl_lib", "build_cache")
HOST = "remote_linux_a100"
TOKENIZER_FILES = {
    "tokenizer_config.json": "db7976db33a60daf2589a824abddcbbd3569fab3c0f96bfe04b76ebdfdf5298a",
    "tokenizer.json": "af73226680ca33dfac341677815c9e1b2c56f3d812af2d0d3d4cdfa1a05df7c2",
}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0", "GIT_ALLOW_PROTOCOL": ""}


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path, value):
    write_json(path, {**value, "sha256": fingerprint(value)})


def unseal(path):
    obj = read(path)
    sha = obj.pop("sha256")
    require(fingerprint(obj) == sha, f"Corrupt sealed manifest: {path}")
    return {**obj, "sha256": sha}


@contextmanager
def lock(path, *, shared=False):
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield


def host_settings(args):
    raw = read(args.host_config)
    require(set(raw) <= set(ARGS) | {"endpoint"} and all(raw.get(k) for k in ARGS[:6]),
            "Provide explicit absolute host paths in the configuration")
    for key in ARGS:
        value = raw.get(key)
        if value:
            path = Path(value).expanduser()
            require(path.is_absolute(), f"Host path must be absolute: {key}")
            setattr(args, key, path.resolve())
        else:
            setattr(args, key, None)
    args.endpoint = raw.get("endpoint", "http://127.0.0.1:11434")
    url = urlsplit(args.endpoint)
    require(url.scheme == "http" and url.port and url.netloc == f"127.0.0.1:{url.port}"
            and not (url.path or url.query or url.fragment), "Dedicated same-user loopback listener required")
    args.manifest = args.models / "manifests/registry.ollama.ai/library/llama3.1/8b"
    return args


def host_inventory(args):
    checks = {}

    def check(key, fn):
        try:
            checks[key] = {"status": "PASS", "detail": fn()}
        except ERRORS as exc:
            checks[key] = {"status": "BLOCKED", "detail": str(exc)}

    def host():
        require(platform.system() == "Linux" and sys.version_info[:2] == (3, 11), "Linux and Python 3.11 required")
        return {"os": platform.platform(), "python": sys.version}

    def packages():
        names = ("FlagEmbedding", "transformers", "torch", "tree-sitter", "tree-sitter-java", "matplotlib")
        result = {}
        for name in names:
            try:
                result[name] = version(name)
            except PackageNotFoundError:
                result[name] = None
        require(all(result.values()) and result["FlagEmbedding"] == "1.4.2"
                and result["transformers"] == "4.56.1" and result["torch"] in {"2.8.0", "2.8.0+cpu"},
                f"Missing or incompatible Python packages: {result}")
        return result

    def tools():
        names = ("git", "curl", "nvidia-smi", "lsof", "ps", "unshare", "perl", "ldd")
        values = {name: shutil.which(name) for name in names}
        require(all(values.values()), f"Missing tools: {values}")
        interfaces = subprocess.check_output(["unshare", "--user", "--map-root-user", "--net", "--",
                                              "ls", "/sys/class/net"], text=True, timeout=10)
        require(interfaces.split() == ["lo"], "Cannot establish an isolated build network namespace")
        return values

    def java():
        raw = subprocess.check_output([args.java_home / "bin/java", "-version"], stderr=subprocess.STDOUT, timeout=10).decode()
        require('version "11.' in raw and (args.java_home / "bin/javac").is_file(), "Java 11 JDK required")
        return raw

    def d4j():
        require(safe_build_path(args.defects4j) and safe_build_path(args.java_home),
                "Defects4J and Java paths must be safe for the pinned shell wrappers")
        require(git(args.defects4j, "rev-parse", "HEAD").decode().strip() == D4J
                and not git(args.defects4j, "diff", "HEAD", "--"), "Tracked Defects4J resources drifted")
        missing = [p for p in ("major/bin/ant", "framework/lib/test_generation/generation",
                              "framework/lib/build_systems/gradle/dists", "framework/lib/build_systems/gradle/deps")
                   if not (args.defects4j / p).exists()]
        require(not missing, f"Missing local framework artifacts: {missing}")
        modules = {"DBD::CSV": "0.48", "DBI": "1.63", "JSON": "2.97", "JSON::Parse": "0.55",
                   "List::Util": "1.33", "String::Interpolate": "0.33", "URI": "1.72"}
        env = {**os.environ, **({"PERL5LIB": str(args.perl_lib)} if args.perl_lib else {})}
        missing = [f"{m}>={v}" for m, v in modules.items() if subprocess.run(["perl", f"-M{m} {v}", "-e", "1"],
                   env=env, capture_output=True, timeout=10).returncode]
        require(not missing, f"Missing Defects4J Perl runtime modules: {missing}")
        ant = subprocess.check_output([args.defects4j / "major/bin/ant", "-version"], text=True,
                                      env={**env, "JAVA_HOME": str(args.java_home),
                                           "PATH": f"{args.java_home}/bin:/usr/bin:/bin"}, stderr=subprocess.STDOUT, timeout=20)
        dependency_roots = [args.defects4j / "framework/lib/build_systems/gradle",
                            args.defects4j / "framework/projects", args.defects4j / "major"]
        if args.build_cache:
            dependency_roots.append(args.build_cache)
        inventory = []
        for path in dependency_roots:
            if path.is_dir():
                files = [p for p in path.rglob("*") if p.is_file() and p.suffix in {".jar", ".zip", ".pom"}]
                inventory.append({"root": str(path), "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
        return {"revision": D4J, "repositories": "scoped commits only", "ant": ant.strip(), "local_dependencies": inventory}

    def gpu():
        rows = nvidia_inventory()
        require(rows, "No visible NVIDIA A100 with compute capability 8.0")
        return rows

    def encoder():
        from rave_fl.graft.encode import checkpoint_hash
        from rave_fl.graft.topology import BGE_M3_HEAD_FILES, is_checkpoint_weight
        files = {p.relative_to(args.encoder).as_posix(): checkpoint_hash(p)
                 for p in sorted(args.encoder.rglob("*")) if p.is_file()
                 and (is_checkpoint_weight(p) or p.name in BGE_M3_HEAD_FILES or p.suffix in {".json", ".model", ".txt"})}
        require(fingerprint(files) == BGE_FILE_MAP_SHA256, "BGE-M3 checkpoint differs from original eight-file snapshot")
        return files

    def serving():
        require(platform.system() == "Linux", "Linux serving binary must be audited on the deployment host")
        weights, model, template, build = verify_local_inputs(args.binary, args.manifest, args.models)
        distribution = args.binary.parents[1]
        cuda = sorted((distribution / "lib/ollama").glob("**/libggml-cuda.so*"))
        require(cuda, "No native GGML CUDA library in the audited distribution")
        linkage = subprocess.check_output(["ldd", args.binary, cuda[0]], text=True, stderr=subprocess.STDOUT, timeout=10,
                                         env={**os.environ, "LD_LIBRARY_PATH": str(cuda[0].parent), "LD_PRELOAD": ""})
        require("not found" not in linkage, f"Unresolved native Ollama dependencies: {linkage}")
        from types import SimpleNamespace
        # Read-only attestation using the same endpoint implementation, without a
        # fabricated schedule certificate or a generation request.
        probe = AttestedOllama({"backend": "ollama", "model": MODEL, "endpoint": args.endpoint, "timeout": 30},
                              SimpleNamespace(identity={"serving_version": "0.17.7", "model_manifest_sha256": MODEL_DIGEST,
                                                        "template_sha256": file_hash(template)}),
                              args.binary, args.manifest, args.models)
        probe.process = probe.check_listener()
        gpu_uuid = probe.process["linux_settings"]["cuda_visible_devices"]
        require(any(row["uuid"] == gpu_uuid for row in nvidia_inventory()), "Dedicated listener is not bound to a visible full A100")
        probe.model_paths = (weights,)
        probe.model_state = probe.local_model_state()
        from rave_fl.graft.selection import TokenizerCounter
        from scripts.build_graft_settlement_certificate import rendered
        require(all(file_hash(args.tokenizer / name) == sha for name, sha in TOKENIZER_FILES.items()),
                "Generation tokenizer files differ from the frozen audited artifact")
        counter = TokenizerCounter(args.tokenizer)
        test = [{"role": "system", "content": "GRAFT-FL tokenizer probe"},
                {"role": "user", "content": "Å\n[] 東京"}]
        proc = subprocess.run([args.binary, "graft-tokenize", weights, template],
                              input=(json.dumps(test) + "\n").encode(), capture_output=True, check=True, timeout=90,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OLLAMA_NO_CLOUD": "1"})
        native = json.loads(proc.stdout)
        require(b"vocab only - skipping tensors" in proc.stderr and native["rendered"].encode() == rendered(test)
                and native["token_ids"] == counter.token_ids(test), "Linux native tokenizer probe differs")
        return {"build": build, "model_blob_sha256": model, "endpoint": probe.verify_endpoint(),
                "native_probe_tokens": len(native["token_ids"]), "physical_generation_verified": False}

    for key, fn in (("host", host), ("python_packages", packages), ("tools_and_network_namespace", tools),
                    ("java", java), ("defects4j_runtime", d4j), ("a100_inventory", gpu),
                    ("encoder", encoder), ("serving_and_tokenizer", serving)):
        check(key, fn)
    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "BLOCKED",
            "generation_requests": 0, "checks": checks}


def portable(args, *tail):
    cmd = [sys.executable, "-m", "rave_fl.graft.portable_experiment", *map(str, tail)]
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **OFFLINE}, check=True)


def safe_build_path(path):
    return all(ch.isascii() and (ch.isalnum() or ch in "/._-") for ch in str(path))


def host_flags(args):
    return ["--profile", HOST, "--serving-binary", args.binary, "--generation-manifest", args.manifest,
            "--ollama-models-dir", args.models]


def group_settings(args, group, rows, members, labels):
    dataset = group / "dataset.json"
    vectors = group / "embeddings"
    vectors.mkdir(exist_ok=True)
    for row in rows:
        src = Path(row["embeddings"])
        dst = vectors / src.name
        if not dst.exists():
            os.link(src, dst)
        require(src.samefile(dst), "Group embeddings differ from frozen source")
    write_json(dataset, {"version": f"defects4j@{D4J}", "instances": [row["row"] for row in rows]})
    path = group / "base-config.json"
    write_json(path, {"benchmark": {"backend": "ollama", "dataset": "defects4j", "device": "a100",
                      "endpoint": args.endpoint, "bugs": sorted(members), "model": MODEL, "model_digest": MODEL_DIGEST,
                      "dataset_file": str(dataset), "candidate_provider": "flexfl", "candidate_count": 24,
                      "top_per_method": 10, "seed": 0, "temperature": 0, "context_window": 8192,
                      "output_tokens": 160, **CAPS},
                      "input_allowance": 1024, "tokenizer_dir": str(args.tokenizer),
                      "labels_file": str(labels), "embeddings_dir": str(vectors), **ARM_CAPS})
    return path


def certify(args, group, rows):
    members = [r["defect"] for r in rows]
    receipt = group / "group-receipt.json"
    if receipt.exists():
        value = unseal(receipt)
        require(value["members"] == sorted(members) and value["id"] == group.name
                and value["deployment_sha256"] == deployment_identity() and value["implementation"] == fingerprint(implementation_identity())
                and file_hash(value["config"]) == value["config_sha256"]
                and read_lock(Path(value["lock"]))["sha256"] == value["lock_sha256"],
                "Interrupted preparation has incompatible sealed group artifacts")
        return {k: v for k, v in value.items() if k not in {"sha256", "deployment_sha256", "implementation"}}
    if group.exists():
        require(not (args.work / "runs" / group.name).exists(), "Partial certification already has a run directory")
        group.rename(group.with_name(f"{group.name}.partial-{time.time_ns()}"))
    group.mkdir(parents=True, exist_ok=False)
    # Label contents live under the offline tree only.
    labels = args.work / "offline/group-labels" / (group.name + ".json")
    label_files = [args.work / "offline/labels" / (bug + ".json") for bug in members]
    write_json(labels, {"instances": [read(path) for path in label_files if path.is_file()]})
    base = group_settings(args, group, rows, members, labels)
    audit = group / "audit"
    with (group / "capture.log").open("wb") as log:
        subprocess.run([sys.executable, "-m", "scripts.audit_graft_schedule_tokens", "--config", str(base),
                        "--serving-binary", str(args.binary), "--generation-manifest", str(args.manifest),
                        "--ollama-models-dir", str(args.models), "--output", str(audit)], cwd=ROOT,
                       env={**os.environ, **OFFLINE}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    cert, cfg = group / "certificate.json", group / "config.json"
    build_schedule(audit, base, cert, cfg, native_sha256=file_hash(audit / "native-comparison.json"),
                   log_sha256=file_hash(audit / "native-tokenizer.log"))
    portable(args, "preflight", "--config", cfg, *host_flags(args), "--attest-live", "--output", group / "sealed")
    cohort = group / "sealed/cohort.json"
    value = read_lock(cohort)
    require(value["ids"] == sorted(members) and len(value["shards"]) == 1 and value["shards"][0]["admitted"],
            "Sealed group differs from its prepared members")
    result = {"id": group.name, "members": value["ids"], "config": str(cfg), "config_sha256": file_hash(cfg),
            "lock": str(cohort), "lock_sha256": value["sha256"], "calls": value["shards"][0]["planned_calls"],
            "peak": value["shards"][0]["planned_physical_tokens"],
            "settlement_sha256": file_hash(cert), "output": str(args.work / "runs" / group.name),
            "offline_recall": {row["instance_id"]: row["offline_labels"]
                               for row in read(group / "sealed/preflight.json")["inventory"]}}
    sealed(receipt, {**result, "deployment_sha256": deployment_identity(), "implementation": fingerprint(implementation_identity())})
    return result


def gate(phase, run):
    try:
        return run(), None
    except ERRORS as exc:
        return None, f"{phase}: {exc}"


def stage(args, selection, name):
    cases = selection[name]
    folder = args.work / name
    folder.mkdir(exist_ok=True)
    manifest = folder / "batch.json"
    if manifest.exists():
        saved = unseal(manifest)
        require(saved["selection_sha256"] == selection["sha256"] and saved["implementation"] == implementation_identity()
                and saved["deployment_sha256"] == deployment_identity(),
                "Existing batch has another selection or implementation")
        save_disposition(folder, saved["dispositions"])
        return saved
    gt = args.work / "offline/ground-truth.json"
    label_spec = selection["label_reference_offline_only"]
    _, label_failure = gate("pinned FlexFL label reference", lambda: rank_blob(label_spec, gt, args.flexfl))
    labels = {} if label_failure else read(gt)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda case: prepare_one_defect(args, case, labels, args.work), cases))
    if label_failure:
        for row in rows:
            row.setdefault("label_reason", label_failure)
    ready = [r for r in rows if r["status"] == "PREPARED"]
    groups = []
    locations = folder / "groups"
    locations.mkdir(exist_ok=True)
    # One deterministic smoke uses the first fully admissible defect. It counts
    # in the same holdout, and a failed preparation does not replace any ID.
    pending = []
    if name == "core":
        for index, candidate in enumerate(ready):
            group, error = gate("smoke certification", lambda: certify(args, locations / f"core-smoke-{candidate['defect']}", [candidate]))
            if group:
                candidate.update(status="READY", group=group["id"], **recall_for(group, candidate))
                groups.append(group)
                pending = ready[index + 1:]
                break
            candidate.update(status="BLOCKED", reason=error)
    else:
        pending = ready.copy()
    for index in range(0, len(pending), 3):
        chunk = pending[index:index + 3]
        base = f"{name}-{index // 3:03d}"
        group, error = gate("group certification", lambda: certify(args, locations / base, chunk))
        if group:
            groups.append(group)
            for row in chunk:
                row.update(status="READY", group=group["id"], **recall_for(group, row))
        else:
            # Isolate pre-generation certification failures; preserve failed
            # group artifacts and keep each original selected defect in place.
            for row in chunk:
                singleton, issue = gate("singleton certification",
                                         lambda: certify(args, locations / f"{base}-{row['defect']}", [row]))
                row["group_failure"] = error
                if singleton:
                    groups.append(singleton)
                    row.update(status="READY", group=singleton["id"], **recall_for(singleton, row))
                else:
                    row.update(status="BLOCKED", reason=issue)
    ids = [r["defect"] for r in cases]
    value = {"schema": "graft-a100-batch-v1", "segment": name, "selection_sha256": selection["sha256"],
             "selected": ids, "groups": groups, "dispositions": rows,
             "implementation": implementation_identity(), "deployment_sha256": deployment_identity(),
             "commit": git(ROOT, "rev-parse", "HEAD").decode().strip(),
             "max_calls": len(ids) * 48, "aggregate_physical_cap": len(ids) * 72000}
    require(sum(g["calls"] for g in groups) <= value["max_calls"]
            and sum(g["peak"] for g in groups) <= value["aggregate_physical_cap"], "Aggregate admission cap exceeded")
    sealed(manifest, value)
    save_disposition(folder, rows)
    print(name, dict(Counter(r["status"] for r in rows)), flush=True)
    return unseal(manifest)


def save_disposition(folder, rows):
    write_csv(folder / "disposition.csv", [{k: r.get(k) for k in
              ("defect", "status", "stage", "reason", "group", "candidates", "label_status", "label_reason",
               "candidate_recall_status", "candidate_recall")} for r in rows], [])


def blocked_report(output, rows, inventory):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "host-inventory.json", inventory)
    write_csv(output / "disposition.csv", rows, [])
    write_json(output / "batch-summary.json", {"status": "BLOCKED", "selected": len(rows),
               "prepared_ready": 0, "request_count": 0, "no_additional_generation_calls": True})
    (output / "REPORT.md").write_text("# GRAFT-FL A100 holdout\n\nHost checks blocked preparation. "
                                     "See host-inventory.json and disposition.csv.\n")


def deployment_identity():
    names = [*sorted((ROOT / "scripts/graft_server").glob("*")),
             ROOT / "scripts/audit_graft_schedule_tokens.py", ROOT / "scripts/build_graft_settlement_certificate.py"]
    return fingerprint({str(p.relative_to(ROOT)): file_hash(p) for p in names if p.is_file()})


def recall_for(group, row):
    measured = group["offline_recall"][row["defect"]]
    return {"candidate_recall_status": measured["status"], "candidate_recall": measured.get("candidate_recall")}


def plans(args):
    selection = read_selection()
    segments = ("core",) if args.count == 100 else ("core", "extension")
    result = [unseal(args.work / name / "batch.json") for name in segments]
    seen = set()
    baseline = None
    group_ids = set()
    for segment, plan in zip(segments, result, strict=True):
        ids = [r["defect"] for r in selection[segment]]
        require(plan["schema"] == "graft-a100-batch-v1" and plan["selection_sha256"] == selection["sha256"]
                and plan["segment"] == segment
                and plan["selected"] == ids and [r["defect"] for r in plan["dispositions"]] == ids
                and plan["max_calls"] == len(ids) * 48 and plan["aggregate_physical_cap"] == len(ids) * 72000,
                "Frozen batch scope differs")
        for g in plan["groups"]:
            source = args.work / segment / "groups" / g["id"]
            require(g["id"] not in group_ids and Path(g["output"]) == args.work / "runs" / g["id"]
                    and Path(g["config"]) == source / "config.json" and Path(g["lock"]) == source / "sealed/cohort.json"
                    and not seen.intersection(g["members"]) and set(g["members"]) <= set(ids),
                    "Duplicate or foreign defect in batch group")
            group_ids.add(g["id"])
            seen.update(g["members"])
            require(file_hash(g["config"]) == g["config_sha256"], "Group config digest drift")
            cfg = fixed.configuration(read(g["config"]))
            proof = SettlementCertificate(cfg)
            value = read_lock(Path(g["lock"]))
            require(value["sha256"] == g["lock_sha256"] and tuple(g["members"]) == proof.cohort == tuple(value["ids"])
                    and proof.identity["tokenizer_files"] == TOKENIZER_FILES
                    and proof.identity["certificate_sha256"] == g["settlement_sha256"]
                    and value.get("live_attestation", {}).get("prompt_settlement_identity") == proof.identity
                    and g["calls"] == proof.calls == value["shards"][0]["planned_calls"]
                    and g["peak"] == value["shards"][0]["planned_physical_tokens"],
                    "Group certificate, lock or allocation differs")
            condition = value["condition"]
            serving = {k: v for k, v in proof.identity.items() if k not in
                       {"certificate_sha256", "capture_plan_sha256", "native_audit_sha256", "native_tokenizer_log_sha256"}}
            science = {k: v for k, v in condition.items() if k not in
                       {"dataset_sha256", "labels_sha256", "execution_admission"}}
            baseline = baseline or (science, serving)
            require((science, serving) == baseline, "Independent groups differ in scientific or serving condition")
        require({d["defect"] for d in plan["dispositions"] if d["status"] == "READY"} == seen.intersection(ids)
                and sum(g["calls"] for g in plan["groups"]) <= plan["max_calls"]
                and sum(g["peak"] for g in plan["groups"]) <= plan["aggregate_physical_cap"], "Batch disposition or aggregate budget drift")
    return result


def audit_group(group):
    run = Path(group["output"])
    rows, evidence, profile, manifest, spending, labels = audit_saved_run(run, "java_strict_v2")
    cohort = read_lock(Path(group["lock"]))
    require(manifest["cohort_binding"] == binding(cohort, 0), "Result belongs to another group")
    validate_shard(cohort, 0, manifest["plan"], labels, manifest["cohort_binding"]["split_manifest_sha256"])
    require(spending["new_calls"] <= group["calls"] and spending["known_provider_tokens"] <= group["peak"],
            "Actual batch charges exceed the admitted group bound")
    return rows, evidence, profile, manifest, spending, labels


def claim_batch(args, selection_sha):
    root = args.models.parent / "graft-fl-live-claims"
    root.mkdir(mode=0o700, exist_ok=True)
    claim = root / f"holdout-{selection_sha}.json"
    value = {"selection_sha256": selection_sha, "workspace": str(args.work)}
    with lock(root / f".holdout-{selection_sha}.lock"):
        if claim.exists():
            require(unseal(claim)["workspace"] == value["workspace"],
                    "This holdout is already bound to another workspace; resume its original journals")
        else:
            sealed(claim, value)
            with claim.open("rb") as stream:
                os.fsync(stream.fileno())
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def execute(args, batches):
    require(args.allow_inference, "Explicit --allow-inference required")
    require(all(b["implementation"] == implementation_identity() and b["deployment_sha256"] == deployment_identity()
                for b in batches), "Batch implementation changed after sealing")
    groups = [g for b in batches for g in b["groups"]]
    require(groups and groups[0]["id"].startswith("core-smoke-") and len(groups[0]["members"]) == 1,
            "No certified core smoke; generation stays disabled")
    claim_batch(args, batches[0]["selection_sha256"])
    smoke_id = groups[0]["id"]
    status_file = args.work / "batch-status.json"
    status = read(status_file) if status_file.exists() else {}
    for group in groups:
        ident = group["id"]
        output = Path(group["output"])
        print(ident, "checking", flush=True)
        status[ident] = {"status": "running", "members": group["members"]}
        write_json(status_file, status)
        try:
            if not (output / "results.json").exists():
                command = "resume" if output.exists() else "run"
                try:
                    portable(args, command, "--config", group["config"], *host_flags(args),
                             "--cohort", group["lock"], "--output", output, "--allow-inference")
                except subprocess.CalledProcessError as exc:
                    # The collector writes a complete audit record for paid
                    # invalid verdicts and returns 3 to signal an incomplete pair.
                    if exc.returncode != 3 or not (output / "results.json").is_file():
                        raise
            rows, _, _, _, spending, _ = audit_group(group)
            require(ident != smoke_id or (len(rows) == 2 and all(r["pair_status"] == "complete" for r in rows)
                                          and spending["new_calls"] == group["calls"]),
                    "Certified Linux smoke failed completeness or accounting")
            status[ident].update(status="sealed", spending=spending)
        except ERRORS + (KeyboardInterrupt,) as exc:
            status[ident].update(status="halted", reason=str(exc))
            write_json(status_file, status)
            raise
        write_json(status_file, status)


def journal_snapshot(path):
    journal = path / "journal.sqlite3"
    if not journal.exists():
        return []
    # mode=ro includes the WAL if present. This is a snapshot for failed or
    # interrupted operations; it does not substitute for a sealed result audit.
    with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        return [dict(zip(("key", "defect", "status", "reservation", "input", "output", "error"), row, strict=True))
                for row in db.execute("SELECT key,instance,status,reservation,input,output,error FROM calls ORDER BY rowid")]


def outcome_details(result, manifest):
    details = []
    for pair, plan in zip(result["pairs"], manifest["plan"]["pairs"], strict=True):
        for i, name in enumerate(fixed.ARMS):
            arm = pair["arms"][name]
            calls = {call["target"]: call for call in arm["calls"]}
            comparison = {row["target"]: {n["id"]: n for n in row["evidence_nodes"]}
                          for row in plan["evidence_comparisons"]}
            for row in plan["arms"][name]:
                target = row["target"]
                evidence = json.loads(row["request"]["messages"][1]["content"])
                nodes = comparison[target]
                code = [unit for unit in evidence["units"] if nodes[unit["id"]]["kind"] == "code"]
                own = [unit for unit in code if unit["owner"] == target]
                call = calls.get(target)
                frames = evidence["frame_references_in_failure"]
                failure_lines = evidence["failure"].splitlines()
                repeated = sum(max(0, count - 1) for count in Counter(
                    failure_lines[ref["failure_line"] - 1].strip()
                    for ref in frames if 0 < ref["failure_line"] <= len(failure_lines)).values())
                details.append({"defect": pair["instance_id"], "arm": "T0" if i == 0 else "T1", "target": target,
                                "target_body_units": len(own), "target_body_lines": len({n for unit in own for n in
                                     range(unit["lines"][0], unit["lines"][1] + 1)}),
                                "source_code_units": len(code), "stack_references": len(frames),
                                "repeated_stack_references": repeated,
                                "suspicion": call["observations"][0]["suspicion"] if call and call["valid"] else None,
                                "valid": call["valid"] if call else None})
    return details


def report_batch(args, batches):
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.work / "runs"),
            "Choose a new report directory outside paid run artifacts")
    rows, evidence, details, usage, paid_keys = [], [], [], [], set()
    dispositions = {r["defect"]: {k: r.get(k) for k in ("defect", "status", "stage", "reason", "group", "label_status", "label_reason",
                                                      "candidate_recall_status", "candidate_recall")}
                    for batch in batches for r in batch["dispositions"]}
    hashes = {}
    all_plans = []
    saved_status = read(args.work / "batch-status.json") if (args.work / "batch-status.json").exists() else {}
    for batch in batches:
        for g in batch["groups"]:
            directory = Path(g["output"])
            for defect in g["members"]:
                dispositions[defect]["execution_status"] = "unstarted"
            if not directory.exists():
                continue
            if (directory / "results.json").exists():
                try:
                    arms, ev, _, manifest, spending, _ = audit_group(g)
                    result = read(directory / "results.json")
                    keys = {c["request_key"] for p in result["pairs"] for arm in p["arms"].values() for c in arm["calls"]}
                    require(not (keys & paid_keys), "Duplicate paid request across groups")
                    expanded = outcome_details(result, manifest)
                    sources = {n: file_hash(directory / n) for n in ("results.json", "manifest.json", "accounting.json",
                                                                    "journal.sqlite3", "raw_ledgers/requests.jsonl")}
                    paid_keys.update(keys)
                    rows.extend(arms)
                    evidence.extend(ev)
                    details.extend(expanded)
                    all_plans.append(manifest["plan"])
                    for p in result["pairs"]:
                        dispositions[p["instance_id"]]["execution_status"] = p["pair_status"]
                    hashes[g["id"]] = sources
                    usage.append({"group": g["id"], "status": "audited", **spending})
                    continue
                except ERRORS as exc:
                    for defect in g["members"]:
                        dispositions[defect]["execution_status"] = "result_audit_failed"
                        dispositions[defect]["execution_reason"] = str(exc)
            snapshot = journal_snapshot(directory)
            for record in snapshot:
                require(record["key"] not in paid_keys, "Duplicate paid or held request across groups")
                paid_keys.add(record["key"])
            for defect in g["members"]:
                row = dispositions[defect]
                if row["execution_status"] == "unstarted":
                    row.update(execution_status="interrupted_or_blocked",
                               execution_reason=saved_status.get(g["id"], {}).get("reason"))
            usage.append({"group": g["id"], "status": "unsealed", "new_calls": len(snapshot),
                          "known_provider_tokens": sum((r["input"] or 0) + (r["output"] or 0) for r in snapshot),
                          "reserved_or_unknown_tokens": sum(r["reservation"] for r in snapshot if r["status"] != "complete"),
                          "journal_status": dict(Counter(r["status"] for r in snapshot)), "journal_rows": snapshot})
    aggregate_calls = sum(r["new_calls"] for r in usage)
    actual = sum(r["known_provider_tokens"] for r in usage)
    holds = sum(r["reserved_or_unknown_tokens"] for r in usage)
    ceilings = sum(b["aggregate_physical_cap"] for b in batches)
    caps_reconciled = (actual + holds <= ceilings
                       and aggregate_calls <= sum(b["max_calls"] for b in batches))
    args.output.mkdir(parents=True, exist_ok=False)
    for row in dispositions.values():
        row.setdefault("execution_status", "not_prepared")
        row.setdefault("execution_reason", None)
    write_csv(args.output / "disposition.csv", list(dispositions.values()), [])
    write_csv(args.output / "response_evidence.csv", details, ["defect", "arm", "target"])
    write_json(args.output / "usage.json", usage)
    score_summary = []
    for (bug, arm), members in group_details(details).items():
        values = [x["suspicion"] for x in members if x["suspicion"] is not None]
        counts = Counter(values)
        score_summary.append({"defect": bug, "arm": arm, "valid_scores": len(values),
                              "suspicion_spread": max(values) - min(values) if values else None,
                              "unique_scores": len(counts), "largest_tie": max(counts.values(), default=0),
                              "target_body_exposure": sum(bool(x["target_body_units"]) for x in members),
                              "stack_reference_count": sum(x["stack_references"] for x in members),
                              "repeated_stack_references": sum(x["repeated_stack_references"] for x in members)})
    write_csv(args.output / "score_and_exposure.csv", score_summary, ["defect", "arm"])
    statuses = Counter(r["execution_status"] for r in dispositions.values())
    prepared = sum(r["status"] == "READY" for r in dispositions.values())
    labeled = {r["instance_id"] for r in rows if r["pair_status"] == "complete" and r["MRR"] is not None}
    wins = Counter()
    per_bug = {r["instance_id"]: r["MRR"] for r in rows if r["arm"] == "T0" and r["instance_id"] in labeled}
    for row in rows:
        if row["arm"] == "T1" and row["instance_id"] in labeled:
            a, b = per_bug[row["instance_id"]], row["MRR"]
            wins["win" if b > a else "loss" if b < a else "tie"] += 1
    if all_plans:
        ordered = sorted(rows, key=lambda r: (r["instance_id"], r["arm"]))
        spending = {"synthetic": False, "new_calls": aggregate_calls, "known_provider_tokens": actual,
                    "physical_inference_tokens": actual, "reserved_or_unknown_tokens": holds,
                    "pending": any(u.get("journal_status", {}).get("pending", 0) for u in usage),
                    "usage_unknown": any(u.get("status") == "unsealed" for u in usage)}
        write_report(args.output / "matched-results", ordered, evidence, "java_strict_v2",
                     {**all_plans[0], "pairs": [p for plan in all_plans for p in plan["pairs"]]},
                     spending, None, hashes)
    input_tokens = sum(r["provider_reported_input_tokens"] for r in rows) + sum(
        (r["input"] or 0) for u in usage for r in u.get("journal_rows", []))
    output_tokens = sum(r["provider_reported_output_tokens"] for r in rows) + sum(
        (r["output"] or 0) for u in usage for r in u.get("journal_rows", []))
    summary = {"selected": len(dispositions), "prepared_ready": prepared, "execution_status": dict(statuses),
               "pair_complete": statuses["complete"],
               "defects_with_dispatched_requests": len({r["instance_id"] for r in rows if r["calls"] > 0}
                   | {r["defect"] for u in usage for r in u.get("journal_rows", [])}),
               "candidate_recall": dict(Counter(r["candidate_recall_status"] for r in rows if r["arm"] == "T0")),
               "invalid_paid_responses": sum(r["invalid_calls"] for r in rows),
               "pair_complete_labeled": len(labeled), "paired_MRR_wins_ties_losses": dict(wins),
               "request_count": aggregate_calls, "known_provider_tokens": actual, "unresolved_holds": holds,
               "known_provider_input_tokens": input_tokens, "known_provider_output_tokens": output_tokens,
               "provider_counters_reconcile": input_tokens + output_tokens == actual,
               "aggregate_physical_cap": ceilings, "actual_plus_holds_within_cap": actual + holds <= ceilings,
               "call_cap": sum(b["max_calls"] for b in batches), "no_additional_generation_calls": True}
    write_json(args.output / "batch-summary.json", summary)
    (args.output / "REPORT.md").write_text(
        "# GRAFT-FL A100 fixed T0/T1 holdout\n\n"
        f"Selected {summary['selected']}, ready {prepared}, pair-complete {statuses['complete']}, "
        f"complete and labeled {len(labeled)}. Known provider tokens {actual:,}, unresolved holds {holds:,}.\n\n"
        "The disposition table retains all selected defects. Accuracy requires a sealed, pair-complete, labeled result. "
        "Complete labeled candidate misses remain zero scores. Unsealed usage is a journal snapshot.\n\n"
        + ("[Matched accuracy and cost report](matched-results/REPORT.md)\n" if all_plans else "No audited generation results are available.\n"))
    print(json.dumps(summary, indent=2))
    return 0 if caps_reconciled and summary["provider_counters_reconcile"] and not statuses["result_audit_failed"] else 3


def group_details(details):
    groups = {}
    for row in details:
        groups.setdefault((row["defect"], row["arm"]), []).append(row)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ("prepare-linux", "prepare-batch", "run-batch", "report-batch"):
        sub = commands.add_parser(cmd)
        if cmd != "report-batch":
            sub.add_argument("--host-config", type=Path, required=True)
        if cmd != "prepare-linux":
            sub.add_argument("--work", type=Path, required=True)
            sub.add_argument("--count", type=int, choices=(100, 200), required=True)
        if cmd == "prepare-linux":
            sub.add_argument("--output", type=Path)
        if cmd == "prepare-batch":
            sub.add_argument("--workers", type=int, default=2)
            sub.add_argument("--trigger-timeout", type=int, default=900)
        if cmd == "run-batch":
            sub.add_argument("--allow-inference", action="store_true")
        if cmd == "report-batch":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(OFFLINE)
    if args.command == "run-batch":
        require(args.allow_inference, "Explicit --allow-inference required")
    if args.command != "report-batch":
        host_settings(args)
    if args.command == "prepare-linux":
        inventory = host_inventory(args)
        if args.output:
            require(not args.output.exists(), "Choose a new inventory output")
            write_json(args.output, inventory)
        print(json.dumps(inventory, indent=2))
        return 0 if inventory["status"] == "PASS" else 3
    args.work = args.work.resolve()
    if args.command == "prepare-batch":
        require(1 <= args.workers <= 8 and 0 < args.trigger_timeout <= 3600, "Bound preparation resource use")
        require(safe_build_path(args.work), "Workspace path must be safe for the pinned Defects4J shell wrappers")
        args.work.mkdir(parents=True, exist_ok=True)
    require(args.work.is_dir(), "Batch workspace unavailable")
    with lock(args.work / ".batch.lock", shared=args.command == "report-batch"):
        if args.command == "prepare-batch":
            selection = read_selection()
            inventory = host_inventory(args)
            write_json(args.work / "host-inventory.json", inventory)
            if inventory["status"] != "PASS":
                for name in (("core",) if args.count == 100 else ("core", "extension")):
                    path = args.work / f"{name}-host-blocked.csv"
                    write_csv(path, [{"defect": r["defect"], "status": "BLOCKED", "stage": "host preflight",
                                     "reason": "See host-inventory.json"} for r in selection[name]], [])
                return 3
            for name in (("core",) if args.count == 100 else ("core", "extension")):
                stage(args, selection, name)
            return 0 if unseal(args.work / "core/batch.json")["groups"] else 3
        if args.command == "report-batch":
            names = ("core",) if args.count == 100 else ("core", "extension")
            if any(not (args.work / name / "batch.json").is_file() for name in names):
                require(all((args.work / f"{name}-host-blocked.csv").is_file() for name in names),
                        "Requested batch has no sealed preparation or host-blocked disposition")
                inventory = read(args.work / "host-inventory.json")
                require(inventory["status"] == "BLOCKED", "Host disposition and inventory disagree")
                selected = read_selection()
                unavailable = [{"defect": row["defect"], "status": "BLOCKED", "stage": "host preflight",
                                "reason": "See host-inventory.json"} for name in names for row in selected[name]]
                blocked_report(args.output, unavailable, inventory)
                return 3
        batches = plans(args)
        if args.command == "run-batch":
            execute(args, batches)
        else:
            return report_batch(args, batches)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ERRORS as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3)

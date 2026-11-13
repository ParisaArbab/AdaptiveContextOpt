"""Scoped holdout acquisition, strict source validation and authentic trigger replay."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from rave_fl.eval.schema import fingerprint, safe_name
from rave_fl.eval.representations import java_tree
from rave_fl.expanded_pool import union_pool
from rave_fl.source_spans import _find_java_file, extract_method
from rave_fl.graft.evaluation import strict_matches
from rave_fl.graft.prompt_settlement import BGE_REVISION, require
from rave_fl.pilot import write_json
from scripts.verify_graft_real_sources import method_spans
from scripts.graft_server.selection import D4J, FLEX, RANKERS, ROOT, git

UPSTREAMS = {name: f"apache/commons-{name.lower()}" for name in
             ("Cli", "Codec", "Compress", "Csv", "JxPath", "Lang", "Math")}
UPSTREAMS.update(Closure="google/closure-compiler", Gson="google/gson", JacksonCore="FasterXML/jackson-core",
                 JacksonDatabind="FasterXML/jackson-databind", Jsoup="jhy/jsoup", Mockito="mockito/mockito",
                 Time="JodaOrg/joda-time")
REPO_NAMES = {name: slug.split("/")[1] for name, slug in UPSTREAMS.items()}
TEST_HEADER = re.compile(r"(?m)^--- ([\w.$]+::[\w$]+)\s*$")
ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
       "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "GIT_ALLOW_PROTOCOL": "https:file"}
ERRORS = (ValueError, OSError, subprocess.SubprocessError, KeyError, ImportError, RuntimeError, IndexError)


def write_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(body)
    tmp.replace(path)


def blob_sha(body):
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


def rank_blob(spec, path, flex=None):
    oid, name = spec["git_blob"], spec["path"]
    require(re.fullmatch("[0-9a-f]{40}", oid), "Invalid pinned FlexFL blob")
    url = f"https://raw.githubusercontent.com/ParisaArbab/FlexFL_OriginalReplication/{FLEX}/{name}"
    if not path.exists():
        body = None
        if flex:
            try:
                body = git(flex, "cat-file", "blob", oid)
            except subprocess.CalledProcessError:
                pass
        if body is None:
            curl = ["curl", "--fail", "--silent", "--show-error", "--proto", "=https",
                    "--max-time", "60", "--max-filesize", "20000000", "--retry", "2"]
            try:
                body = subprocess.check_output([*curl, url], timeout=210)
            except subprocess.CalledProcessError:
                api_url = f"https://api.github.com/repos/ParisaArbab/FlexFL_OriginalReplication/git/blobs/{oid}"
                response = json.loads(subprocess.check_output([*curl, api_url], timeout=210))
                require(response["encoding"] == "base64" and response["sha"] == oid, "GitHub blob identity mismatch")
                body = base64.b64decode("".join(response["content"].split()), validate=True)
        require(blob_sha(body) == oid, "Downloaded FlexFL blob differs from pinned tree")
        write_bytes(path, body)
    body = path.read_bytes()
    require(blob_sha(body) == oid, "Staged FlexFL blob differs from pinned tree")
    return {"url": url, "git_blob": oid, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def shallow(repo, project, revision):
    require(re.fullmatch("[0-9a-f]{40}", revision), "Expected an exact Git SHA")
    url = "https://github.com/" + UPSTREAMS[project] + ".git"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=repo.parent, prefix=".fetch-") as tmp:
            staged = Path(tmp) / "repo"
            subprocess.run(["git", "init", "-q", str(staged)], env=ENV, check=True)
            subprocess.run(["git", "-C", str(staged), "-c", "http.followRedirects=false", "fetch", "-q",
                            "--depth=1", "--no-tags", "--no-recurse-submodules", url, revision],
                           env=ENV, timeout=180, check=True)
            require(git(staged, "rev-parse", "FETCH_HEAD").decode().strip() == revision,
                    "Upstream returned an unexpected commit")
            subprocess.run(["git", "-C", str(staged), "-c", "core.autocrlf=false", "checkout", "-q",
                            "--detach", revision], env=ENV, check=True)
            git(staged, "fsck", "--strict", "--no-reflogs", revision)
            staged.rename(repo)
    require(git(repo, "rev-parse", "HEAD").decode().strip() == revision
            and (repo / ".git/shallow").read_text().strip() == revision
            and not git(repo, "status", "--porcelain", "--untracked-files=all"),
            "Pinned checkout revision or cleanliness failed")
    require(git(repo, "rev-list", "--all", "HEAD").decode().splitlines() == [revision], "Unexpected Git history")
    git(repo, "fsck", "--strict", "--no-reflogs", revision)
    files = git(repo, "ls-tree", "-r", "-z", revision)
    require(not any(row.startswith(b"160000 ") for row in files.split(b"\0")), "Gitlinks require a separate preparation policy")
    for row in files.split(b"\0"):
        if not row:
            continue
        spec, name = row.split(b"\t", 1)
        mode, kind, oid = spec.split()
        path = repo / os.fsdecode(name)
        require(path.resolve().is_relative_to(repo.resolve()), "Pinned source symlink leaves its checkout")
        require(path.is_symlink() == (mode == b"120000"), "Checkout file mode differs from the pinned tree")
        body = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        require(kind == b"blob" and blob_sha(body).encode() == oid, f"Checkout byte drift at {path}")
    return {"url": url, "commit": revision, "tree": git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
            "tree_listing_sha256": hashlib.sha256(files).hexdigest(), "fetch": "exact SHA, depth 1, no tags/submodules"}


def source_lines(path):
    source = path.read_text(encoding="utf-8")
    spans = method_spans(source)
    body = bytearray(source.encode("utf-8"))
    nodes = [java_tree(source)]
    while nodes:
        node = nodes.pop()
        if node.type in {"block_comment", "line_comment"}:
            for offset in range(node.start_byte, node.end_byte):
                if body[offset] not in (10, 13):
                    body[offset] = 32
        else:
            nodes.extend(node.named_children)
    return spans, body.splitlines()


def audit_pool(checkout, flex, bug):
    checkout = checkout.resolve()
    candidates, meta = union_pool(flex, bug, 10)
    indexed, audits, failures = {}, [], []
    for rank, candidate in enumerate(candidates, 1):
        target = candidate.function_id
        row = meta[target]
        record = {"target": target, "rank": rank, "selected": rank <= 24, **row}
        try:
            path = _find_java_file(checkout, row["file_class"])
            if path not in indexed:
                indexed[path] = source_lines(path)
            spans, code = indexed[path]
            expected_owner = target.split("(", 1)[0].rpartition(".")[0].replace("$", ".")
            matches = [s for s in spans if (strict_matches(target, s.signature)
                       and s.signature.split("(", 1)[0].rpartition(".")[0] == expected_owner)]
            start, end = row["start_line"], row["end_line"]
            record["file"] = path.relative_to(checkout).as_posix()
            record["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["identity_matches"] = [s.__dict__ for s in matches]
            require(len(matches) == 1, "Missing or ambiguous full Java declaration")
            span = matches[0]
            require(1 <= start <= span.start <= span.end <= end <= len(code),
                    "Frozen CSV span truncates or offsets its declaration")
            require(not any(s.start <= end and s.end >= start and not (span.start <= s.start <= s.end <= span.end)
                            for s in spans), "CSV span intersects an unrelated declaration")
            extra = [i for i in (*range(start, span.start), *range(span.end + 1, end + 1)) if code[i - 1].strip()]
            require(not extra, f"CSV span includes unrelated noncomment lines: {extra}")
            extract_method(checkout, target, row)
            record["status"] = "VERIFIED"
        except ERRORS as exc:
            record.update(status="BLOCKED", reason=str(exc))
            failures.append(target)
        audits.append(record)
    return {"ordered_candidates": [r["target"] for r in audits if r["selected"]], "rows": audits,
            "frozen_order_sha256": fingerprint([r["target"] for r in audits if r["selected"]]), "failures": failures}


def failure_signature(body):
    lines = body.splitlines()
    require(len(lines) >= 2 and TEST_HEADER.fullmatch(lines[0]), "Malformed Defects4J failure")
    project = [line.strip() for line in lines[2:] if line.lstrip().startswith("at ")
               and not line.lstrip().startswith(("at java.", "at java.base/", "at jdk.", "at sun.",
                                                 "at junit.", "at org.junit.", "at org.apache.tools.ant.",
                                                 "at edu.washington.cs.mut."))]
    require(project and not re.search(r"\b(?:NoClassDefFoundError|ClassNotFoundException|UnsupportedClassVersionError|"
                                     r"NoSuchMethodError|OutOfMemoryError)\b",
                                     lines[1]), "Failure lacks project stack frames or has an infrastructure exception")
    return lines[0], lines[1], project


def archive_blocks(text):
    headers = list(TEST_HEADER.finditer(text))
    require(headers, "No archived official triggers")
    return [(m[1], text[m.start():headers[i + 1].start() if i + 1 < len(headers) else len(text)].strip())
            for i, m in enumerate(headers)]


def official_replay(args, case, buggy, fixed, folder):
    bug = case["defect"]
    project, number = bug.split("-")
    d4j = args.defects4j.resolve()
    archive = git(d4j, "show", f"{D4J}:framework/projects/{project}/trigger_tests/{number}").decode()
    require((d4j / f"framework/projects/{project}/trigger_tests/{number}").read_text() == archive,
            "Defects4J archived trigger bytes drifted")
    trials = archive_blocks(archive)
    prefix = ["unshare", "--user", "--map-root-user", "--net", "--"]
    env = {**os.environ, "PATH": f"{args.java_home}/bin:/usr/local/bin:/usr/bin:/bin",
           "JAVA_HOME": str(args.java_home), "TZ": "America/Los_Angeles",
           "GIT_ALLOW_PROTOCOL": "file", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_TERMINAL_PROMPT": "0", "GIT_AUTHOR_NAME": "Defects4J", "GIT_AUTHOR_EMAIL": "d4j@localhost",
           "GIT_COMMITTER_NAME": "Defects4J", "GIT_COMMITTER_EMAIL": "d4j@localhost"}
    for key in list(env):
        if key.lower().endswith("_proxy") or key in {"CLASSPATH", "GRADLE_USER_HOME", "GRADLE_OPTS", "JAVA_TOOL_OPTIONS",
                "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "MAVEN_OPTS", "MAVEN_ARGS", "ANT_OPTS", "ANT_ARGS", "PERL5OPT",
                "MAJOR_ROOT", "TESTGEN_LIB_DIR", "BUILD_SYSTEMS_LIB_DIR"}:
            env.pop(key)
    cli = d4j / "framework/bin/defects4j"
    require(cli.is_file(), "Initialized Defects4J CLI required")
    logdir = folder / "trigger"
    logdir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=logdir, prefix=".checkout-") as scratch:
        root = Path(scratch)
        sources = root / "repos"
        sources.mkdir()
        (sources / "README").write_text("Defects4J version 3, scoped GRAFT-FL trigger reproduction\n")
        # The pinned adapter checks out the fixed snapshot inside the
        # disposable reproduction area and applies its official reverse patch.
        union = sources / (REPO_NAMES[project] + ".git")
        subprocess.run(["git", "init", "--bare", "-q", union], env=env, check=True)
        for role, checkout in (("buggy", buggy), ("fixed", fixed)):
            subprocess.run(["git", "-C", union, "fetch", "-q", "--depth=1", "--no-tags", checkout,
                            f"HEAD:refs/heads/{role}"], env=env, check=True, timeout=60)
        subprocess.run(["git", "-C", union, "symbolic-ref", "HEAD", "refs/heads/fixed"], env=env, check=True)
        work = root / "work"
        env["REPO_DIR"] = str(sources)
        env.update(HOME=str(root / "home"), TMPDIR=str(root / "tmp"))
        Path(env["HOME"]).mkdir()
        Path(env["TMPDIR"]).mkdir()
        if args.perl_lib:
            env["PERL5LIB"] = str(args.perl_lib)
        if args.build_cache:
            # Only the user's explicitly staged dependency cache is readable by builds.
            cache = args.build_cache.resolve()
            for directory in (".m2", ".gradle"):
                if (cache / directory).is_dir():
                    shutil.copytree(cache / directory, Path(env["HOME"]) / directory, dirs_exist_ok=True)
        def run(name, *tail):
            cmd = [*prefix, str(cli), *map(str, tail)]
            with (logdir / f"{name}.log").open("ab") as f:
                try:
                    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=args.trigger_timeout)
                finally:
                    with (logdir / "commands.jsonl").open("a") as receipt:
                        receipt.write(json.dumps({"argv": cmd, "phase": name, "REPO_DIR": env["REPO_DIR"],
                                                  "returncode": proc.returncode if "proc" in locals() else None}) + "\n")
            require(proc.returncode == 0, f"Defects4J {name} failed ({proc.returncode}); inspect {logdir}")
        run("checkout", "checkout", "-p", project, "-v", number + "b", "-w", work)
        # Verify the full production Java path set and bytes, allowing only the
        # pinned adapter's documented top-level module relocation (e.g. Time).
        def production(root):
            result = {}
            for p in root.rglob("*.java"):
                name = p.relative_to(root).as_posix()
                if project == "Time":
                    name = name.removeprefix("JodaTime/")
                if re.search(r"(^|/)(test|tests|testsrc|examples|target|build|generated_sources)(/|$)", name):
                    continue
                require(name not in result and p.resolve().is_relative_to(root.resolve()), "Ambiguous production Java source identity")
                result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            return result
        original = production(buggy)
        materialized = production(work)
        # Some official adapters add generated sources. Existing source bytes and
        # all changes outside the attested path subset fail closed.
        if not original or original != materialized:
            write_json(logdir / "production-drift.json", {"missing": sorted(original.keys() - materialized.keys()),
                       "extra": sorted(materialized.keys() - original.keys()),
                       "changed": sorted(p for p in original.keys() & materialized.keys() if original[p] != materialized[p])})
            raise ValueError("Official preparation production Java bytes/path set differ from pinned buggy source")
        observed = []
        for index, (target, archived) in enumerate(trials, 1):
            run(f"trigger-{index:03}", "test", "-w", work, "-t", target)
            raw = (work / "failing_tests").read_text()
            write_bytes(logdir / f"trigger-{index:03}.observed.txt", raw.encode())
            blocks = archive_blocks(raw)
            actual = failure_signature(blocks[0][1])
            expected = failure_signature(archived)
            require(len(blocks) == 1 and blocks[0][0] == target and actual[:2] == expected[:2]
                    and actual[2] == expected[2], f"Authentic trigger assertion/project frames differ for {target}")
            observed.append(raw.rstrip() + "\n")
        require(production(work) == original, "Production source changed during test")
        text = "".join(observed)
        write_bytes(folder / "observed-failure.txt", text.encode())
        return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "trigger_count": len(trials),
                "archive_sha256": hashlib.sha256(archive.encode()).hexdigest(), "production_java_sha256": fingerprint(original),
                "pinned_wrapper": D4J, "logs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sorted(logdir.iterdir()) if p.is_file()},
                "network": "Linux unshare user+network namespace", "match": "exact trigger, failure signature and project frames"}


def label_case(bug, buggy, fixed, labels):
    from difflib import SequenceMatcher
    expected = labels.get(bug, [])
    expected = [expected] if isinstance(expected, str) else expected
    require(isinstance(expected, list) and expected, "Pinned method gold unavailable")
    paths = {}
    # Compare the two independently staged objects without importing fixed code into online inputs.
    for p in buggy.rglob("*.java"):
        rel = p.relative_to(buggy)
        if re.search(r"(^|/)(test|tests|testsrc)(/|$)", rel.as_posix()):
            continue
        q = fixed / rel
        if not q.is_file() or p.read_bytes() == q.read_bytes():
            continue
        a, b = p.read_text(), q.read_text()
        spans = method_spans(a)
        for tag, i, j, _, _ in SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for span in spans:
                if span.start <= max(i + 1, j) and span.end >= i + 1:
                    paths.setdefault(rel.as_posix(), set()).add(span)
    matches = {label: [(p, s) for p, spans in paths.items() for s in spans if strict_matches(label, s.signature)]
               for label in dict.fromkeys(expected)}
    require(all(len(items) == 1 for items in matches.values()), "Pinned gold method was missing or ambiguous in changed buggy declarations")
    files = sorted({items[0][0] for items in matches.values()})
    receipts = {label: {"path": items[0][0], "declaration": items[0][1].__dict__,
                        "buggy_file_sha256": hashlib.sha256((buggy / items[0][0]).read_bytes()).hexdigest(),
                        "fixed_file_sha256": hashlib.sha256((fixed / items[0][0]).read_bytes()).hexdigest()}
                for label, items in matches.items()}
    return {"instance_id": bug, "gold": {"functions": expected, "files": files,
                                         "semantics": "Pinned FlexFL gold independently matched to changed buggy declarations"},
            "offline_source_spans": receipts}


def prepare_one_defect(args, case, labels, work):
    bug = case["defect"]
    project = bug.split("-")[0]
    folder = work / "preparation" / bug
    folder.mkdir(parents=True, exist_ok=True)
    flex = work / "online/flexfl"
    buggy = work / "online/checkouts" / bug
    fixed = work / "offline/fixed" / bug
    outcome = {"defect": bug, "status": "BLOCKED", "stage": "source and rankings"}
    try:
        ranking = {r: rank_blob(case["rankings"][r], flex / case["rankings"][r]["path"], args.flexfl) for r in RANKERS}
        source = shallow(buggy, project, case["buggy"])
        write_json(folder / "acquisition.json", {"rankings": ranking, "buggy": source})
        outcome["stage"] = "candidate coordinates"
        pool = audit_pool(buggy, flex, bug)
        write_json(folder / "candidates.json", pool)
        outcome["candidates"] = len(pool["ordered_candidates"])
        require(not pool["failures"], f"{len(pool['failures'])} frozen coordinate anomalies; see candidates.json")
        outcome["stage"] = "official triggers"
        fixed_receipt = shallow(fixed, project, case["fixed_offline_only"])
        write_json(work / "offline" / "fixed-receipts" / f"{bug}.json", fixed_receipt)
        evidence = folder / "observed-failure.txt"
        receipt_path = folder / "trigger-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            require(hashlib.sha256(evidence.read_bytes()).hexdigest() == receipt["sha256"]
                    and receipt["buggy_tree"] == source["tree"] and receipt["fixed_tree"] == fixed_receipt["tree"]
                    and receipt["pinned_wrapper"] == D4J and all(
                        hashlib.sha256((folder / "trigger" / n).read_bytes()).hexdigest() == h for n, h in receipt["logs"].items()),
                    "Saved observed failure provenance drift")
        else:
            receipt = official_replay(args, case, buggy, fixed, folder)
            write_json(receipt_path, {**receipt, "buggy_tree": source["tree"], "fixed_tree": fixed_receipt["tree"]})
        row = {"instance_id": bug, "repository": source["url"], "revision": case["buggy"], "language": "java",
               "checkout": str(buggy), "flexfl_repo": str(flex), "evidence_file": str(evidence),
               "evidence_kind": "observed_failure"}
        outcome["stage"] = "encoder"
        from rave_fl.eval.adapters import normalize
        from rave_fl.eval.runner import prepare_one
        from rave_fl.graft.fixed_inputs import fixed_graph
        from rave_fl.graft.fixed_evidence_experiment import configuration
        from rave_fl.graft.topology import EmbeddingBundle
        settings = {"benchmark": {"backend": "ollama", "dataset": "defects4j", "dataset_file": str(folder / "dataset.json"),
                                  "candidate_provider": "flexfl", "top_per_method": 10, "candidate_count": 24},
                    "embeddings_dir": str(folder / "embeddings")}
        cfg = configuration(settings)
        instance = normalize(row, "defects4j", {}, folder, cfg["benchmark"])
        instance, entities, *_ = prepare_one(instance, cfg["benchmark"])
        graph = fixed_graph(instance, entities)
        require([e.entity_id for e in entities] == pool["ordered_candidates"], "Frozen order drift")
        vectors = folder / "embeddings" / (safe_name(bug) + ".json")
        if not vectors.exists():
            # The shared exporter sets offline variables while loading.
            # Isolate each encoder process and bound its native CPU threads.
            graph_path = folder / "graph.json"
            write_json(graph_path, graph.to_dict())
            with (folder / "encoder.log").open("wb") as log:
                subprocess.run([sys.executable, "-m", "rave_fl.graft.encode", "--graph", graph_path,
                                "--model-dir", args.encoder, "--revision", BGE_REVISION, "--output", vectors],
                               cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        EmbeddingBundle.load(vectors).validate_for_graph(graph)
        outcome.update(status="PREPARED", stage="group certification", row=row, embeddings=str(vectors))
        try:
            label = label_case(bug, buggy, fixed, labels)
            write_json(work / "offline/labels" / f"{bug}.json", label)
            outcome["label_status"] = "VERIFIED"
        except ERRORS as exc:
            unavailable = work / "offline/labels" / f"{bug}.json"
            if unavailable.exists():
                unavailable.unlink()
            outcome.update(label_status="UNAVAILABLE", label_reason=str(exc))
    except ERRORS as exc:
        outcome["reason"] = str(exc)
    write_json(folder / "disposition.json", outcome)
    return outcome

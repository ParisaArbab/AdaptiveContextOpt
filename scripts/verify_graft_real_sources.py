"""Independently check a pinned Defects4J checkout, FlexFL spans and offline method gold."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import io
from pathlib import Path
import re

from rave_fl.eval.adapters import load_instances, read_rows
from rave_fl.eval.runner import prepare_one
from rave_fl.eval.schema import CompatibilityError, beneath, fingerprint
from rave_fl.eval.representations import java_tree
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.evaluation import parse_identity, strict_matches
from rave_fl.graft.fixed_evidence_reporting import load_json, require
from rave_fl.pilot import write_json
from scripts.audit_graft_phase6 import git
from wp1.flexfl_data import TRADITIONAL_METHODS, flexfl_root, read_ground_truth


TYPE_DECLARATIONS = {"class_declaration", "interface_declaration", "enum_declaration",
                     "record_declaration", "annotation_type_declaration"}
CALLABLES = {"method_declaration", "constructor_declaration"}
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
SHA = re.compile(r"[a-f0-9]{40}")


@dataclass(frozen=True)
class MethodSpan:
    signature: str
    start: int
    end: int


def method_spans(source):
    root = java_tree(source)
    require(not root.has_error, "Java parse error prevents method-span verification")
    data = source.encode("utf-8")
    # Derive line numbers from byte offsets. On the local arm64 Python 3.11
    # environment, repeated tree-sitter Point accesses crashed on real files.
    newlines = [match.end() for match in re.finditer(b"\n", data)]

    def text(node):
        return data[node.start_byte:node.end_byte].decode("utf-8")

    packages = [text(node) for node in root.named_children if node.type == "package_declaration"]
    require(len(packages) <= 1, "Unexpected Java package declarations")
    package = ()
    if packages:
        match = re.fullmatch(r"package\s+([\w.]+)\s*;", packages[0])
        require(match is not None, "Unresolved Java package declaration")
        package = (match.group(1),)
    methods = []

    def walk(node, parents):
        if node.type in TYPE_DECLARATIONS:
            name = node.child_by_field_name("name")
            if name is not None:
                parents = (*parents, text(name))
        if node.type in CALLABLES:
            name = node.child_by_field_name("name")
            params = node.child_by_field_name("parameters")
            require(name is not None and params is not None, "Missing Java method name or parameters")
            if len(parents) > len(package):
                signature = ".".join((*parents, text(name))) + text(params)
                methods.append(MethodSpan(signature, bisect_right(newlines, node.start_byte) + 1,
                                          bisect_right(newlines, node.end_byte) + 1))
        for child in node.named_children:
            walk(child, parents)

    walk(root, package)
    return methods


def checked_git(checkout, *args):
    result = git(checkout, *args)
    if result.returncode:
        raise CompatibilityError("Offline Git object or diff unavailable for " + " ".join(args))
    return result.stdout


def candidates_verified(instance, entities):
    paths = {}
    verified = []
    for entity in entities:
        if entity.path not in paths:
            source = beneath(instance.checkout, entity.path).read_text(encoding="utf-8")
            archived = checked_git(instance.checkout, "show", f"HEAD:{entity.path}").decode("utf-8")
            require(source.splitlines() == archived.splitlines(),
                    f"Candidate source differs from the pinned Git blob: {entity.path}")
            paths[entity.path] = source.splitlines(), method_spans(source)
        lines, spans = paths[entity.path]
        matches = [span for span in spans if strict_matches(entity.entity_id, span.signature)
                   and entity.start_line <= span.start <= span.end <= entity.end_line]
        require(len(matches) == 1, f"Candidate span or signature mismatch: {entity.entity_id}")
        require(entity.source.splitlines() == lines[entity.start_line - 1:entity.end_line],
                f"Candidate body differs from declared source span: {entity.entity_id}")
        require(not any(other.start <= entity.end_line and other.end >= entity.start_line
                        and not (matches[0].start <= other.start <= other.end <= matches[0].end)
                        for other in spans),
                f"Candidate range includes a separate method: {entity.entity_id}")
        verified.append({"entity_id": entity.entity_id, "path": entity.path, **asdict(matches[0])})
    return verified


def changed_methods(checkout, buggy, fixed_revision, source_root):
    require(SHA.fullmatch(buggy) and SHA.fullmatch(fixed_revision) and buggy != fixed_revision,
            "Official buggy and fixed commit SHAs are required")
    require(source_root and source_root != ".", "An explicit production Java source root is required")
    beneath(checkout, source_root)
    args = ("diff", "--no-ext-diff", "--no-textconv", "--no-renames")
    changed = checked_git(checkout, *args, "--name-only", "-z", buggy, fixed_revision, "--", source_root)
    files = [name for name in changed.decode().split("\0") if name.endswith(".java")]
    require(files, "No changed production Java files under the declared root")
    result = defaultdict(set)
    for path in files:
        source = checked_git(checkout, "show", f"{buggy}:{path}").decode("utf-8")
        spans = method_spans(source)
        patch = checked_git(checkout, *args, "--unified=0", buggy, fixed_revision, "--", path)
        hunks = [match for line in patch.decode().splitlines() if (match := HUNK.match(line))]
        require(hunks, f"No comparable Java patch hunks for {path}")
        for match in hunks:
            start, count = int(match[1]), int(match[2] or "1")
            for line in range(start, start + max(count, 1)):
                matches = [span for span in spans if span.start <= line <= span.end
                           and (count or line < span.end)]
                matches.sort(key=lambda span: span.end - span.start)
                require(matches and (len(matches) == 1 or
                        matches[0].end - matches[0].start < matches[1].end - matches[1].start),
                        f"Changed production line outside a unique buggy method: {path}:{line}")
                result[path].add(matches[0].signature)
    return {path: sorted(names) for path, names in sorted(result.items())}


def verify(args):
    value = load_json(args.config)
    require(value.get("labels_file") is None, "Source audit input must exclude evaluation labels")
    config = fixed.configuration(value)
    opts = config["benchmark"]
    require(opts["dataset"] == "defects4j" and opts["candidate_provider"] == "flexfl",
            "Use the frozen Defects4J/FlexFL condition")
    online, manifest = read_rows(opts["dataset_file"])
    require(not any({"gold", "gold_file", "patch", "fix_patch"} & row.keys() for row in [manifest, *online]),
            "Online inputs must exclude reference patches and labels")
    instances, _ = load_instances({**opts, "bugs": [args.bug], "limit": None})
    instance, = instances
    archival_only = getattr(args, "archival_only", False)
    if archival_only:
        require(instance.evidence.kind == "trigger_description",
                "Archive-only source audits require trigger_description evidence")
        opts = {**opts, "require_observed_failure": False}
    project, ident = args.bug.split("-", 1)
    require(re.fullmatch(r"[A-Za-z]+", project) and ident.isdecimal()
            and SHA.fullmatch(args.metadata_revision) and SHA.fullmatch(args.flexfl_revision),
            "Use a Defects4J bug ID and full pinned reference revisions")
    metadata = checked_git(args.metadata, "show", f"{args.metadata_revision}:framework/projects/{project}/active-bugs.csv")
    rows = [row for row in csv.DictReader(io.StringIO(metadata.decode())) if row["bug.id"] == ident]
    require(len(rows) == 1, "Missing or duplicate pinned Defects4J mapping")
    official = rows[0]
    require(instance.revision == official["revision.id.buggy"], "Manifest revision differs from official mapping")
    reference = args.flexfl_reference.resolve()
    reference_data = flexfl_root(reference).relative_to(reference) / "data"
    local_data = flexfl_root(Path(instance.flexfl_repo)) / "data"
    require(not (local_data / "input/ground_truth").exists(),
            "Use an online ranking tree without the FlexFL ground-truth directory")
    for method in TRADITIONAL_METHODS:
        name = Path("FL_results") / method / "Defects4J" / f"{args.bug}_method-susps.csv"
        archived = checked_git(reference, "show", f"{args.flexfl_revision}:{(reference_data / name).as_posix()}")
        require((local_data / name).read_bytes() == archived, f"FlexFL ranking differs from pinned {method} data")
    instance, entities, schedule, _, _, prepared = prepare_one(instance, opts)
    fixed.fixed_schedule(schedule, entities)
    verified = candidates_verified(instance, entities)
    patch = changed_methods(Path(instance.checkout), instance.revision,
                            official["revision.id.fixed"], instance.source_root)
    derived = [signature for values in patch.values() for signature in values]
    gold_name = Path("input/ground_truth/Defects4J/gt.json")
    archived_gold = checked_git(reference, "show", f"{args.flexfl_revision}:{(reference_data / gold_name).as_posix()}")
    require((flexfl_root(args.historical_labels) / "data" / gold_name).read_bytes() == archived_gold,
            "Historical labels differ from the pinned FlexFL artifact")
    historical = read_ground_truth(args.historical_labels, args.bug)
    require(len(derived) == len(historical) and all(
        sum(strict_matches(signature, label) for label in historical) == 1 for signature in derived
    ) and all(sum(strict_matches(signature, label) for signature in derived) == 1 for label in historical),
        "Independent patch-derived gold disagrees with the historical method labels")
    # Gold remains in a separate file. The candidate audit can be consumed without it.
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "candidate-audit.json", {
        "instance_id": args.bug, "buggy_revision": instance.revision, "pool_sha256": prepared["pool_hash"],
        "verified_spans": verified, "schedule_sha256": fingerprint(
            [c.function_id for c in fixed.fixed_schedule(schedule, entities)]
        ),
        "trigger_replay_status": "requires_separate_verified_run",
        "evidence_kind": instance.evidence.kind,
    })
    write_json(args.output / "offline-gold.json", {"instances": [{
        "instance_id": args.bug, "gold": {"files": list(patch), "functions": historical,
                                         "semantics": "patch-derived buggy methods cross-checked against FlexFL"},
    }]})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "metadata", "flexfl-reference", "historical-labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bug", required=True)
    parser.add_argument("--metadata-revision", required=True)
    parser.add_argument("--flexfl-revision", required=True)
    parser.add_argument("--archival-only", action="store_true",
                        help="Audit checkout and source spans using archived trigger evidence only")
    try:
        return verify(parser.parse_args())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Defects4J source audit: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

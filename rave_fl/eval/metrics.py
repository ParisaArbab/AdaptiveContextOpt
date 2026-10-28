"""Offline-only reference localization extraction and multi-location metrics."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import statistics

from rave_fl.eval.adapters import read_rows
from rave_fl.eval.schema import GoldLabels, CompatibilityError, relative_path


def reference_patch_files(patch):
    """Filtered changed-production-file proxy, not a claim of causal faulty files."""
    files = []
    excluded = []
    chunks = re.split(r"(?=^diff --git )", patch, flags=re.M)
    for chunk in chunks:
        match = re.search(r"^diff --git a/(.+) b/(.+)$", chunk, re.M)
        if not match:
            continue
        old, new = match.groups()
        path = new
        try:
            relative_path(path)
        except CompatibilityError:
            excluded.append({"path": path, "reason": "unsafe path"})
            continue
        parts = {p.lower() for p in Path(path).parts}
        if (
            parts & {"tests", "test", "testing", "generated", "vendor", "dist", "build"}
            or Path(path).name.startswith("test_")
            or Path(path).suffix
            not in {".py", ".java", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".php"}
        ):
            excluded.append({"path": path, "reason": "test/generated/non-source reference change"})
            continue
        if old != new:
            excluded.append(
                {"path": path, "reason": "rename requires explicit localization labels"}
            )
            continue
        removed = [
            s[1:] for s in chunk.splitlines() if s.startswith("-") and not s.startswith("---")
        ]
        added = [s[1:] for s in chunk.splitlines() if s.startswith("+") and not s.startswith("+++")]
        if (
            not removed
            and not added
            or "".join("".join(removed).split()) == "".join("".join(added).split())
        ):
            excluded.append({"path": path, "reason": "no substantive source-line change"})
            continue
        files.append(path)
    return tuple(dict.fromkeys(files)), tuple(excluded)


def load_gold(instance, source):
    """Called after collection. Adapters/controllers never receive this object."""
    rows, meta = read_rows(source["dataset_file"])
    row = None
    for r in rows:
        ident = r.get("instance_id", r.get("bug", "")) or (
            f"{r.get('org')}__{r.get('repo')}-{r.get('number')}"
            if instance.dataset == "multi-swe-bench"
            else ""
        )
        if ident == instance.instance_id:
            row = r
            break
    if row is None:
        raise CompatibilityError("Offline label row no longer present")
    data = row.get("gold")
    if row.get("gold_file"):
        p = (Path(source["path_base"]) / row["gold_file"]).resolve()
        data = json.loads(p.read_text())
    if data is not None:
        if not isinstance(data, dict):
            raise CompatibilityError("Gold must contain explicit files/functions lists")
        files = data.get("files", [])
        functions = data.get("functions", [])
        if (
            not isinstance(files, list)
            or not isinstance(functions, list)
            or any(not isinstance(v, str) for v in files + functions)
        ):
            raise CompatibilityError("Malformed offline localization labels")
        for f in files:
            relative_path(f)
        return GoldLabels(
            tuple(dict.fromkeys(files)),
            tuple(dict.fromkeys(functions)),
            data.get("semantics", "curated_localization"),
        )
    if instance.dataset == "defects4j" and instance.flexfl_repo:
        from wp1.flexfl_data import read_ground_truth

        functions = tuple(read_ground_truth(Path(instance.flexfl_repo), instance.instance_id))
        files = []
        from rave_fl.source_spans import _find_java_file

        for ident in functions:
            try:
                files.append(
                    _find_java_file(
                        Path(instance.checkout), ident.split("(", 1)[0].rsplit(".", 1)[0]
                    )
                    .relative_to(Path(instance.checkout))
                    .as_posix()
                )
            except (ValueError, FileNotFoundError):
                pass
        return GoldLabels(
            tuple(dict.fromkeys(files)), functions, semantics="Defects4J supplied method labels"
        )
    patch = (
        row.get("patch")
        if instance.dataset.startswith("swebench")
        else row.get("fix_patch")
        if instance.dataset == "multi-swe-bench"
        else None
    )
    if patch:
        files, exclusions = reference_patch_files(patch)
        return GoldLabels(
            files,
            (),
            "reference_changed_production_files_proxy",
            exclusions,
            "No reliable method-level labels; explicit validated labels required",
        )
    return GoldLabels(unavailable_reason="No offline localization labels supplied")


def ranking_metrics(ranked, gold, pool=None):
    if not gold:
        return None
    gold = set(gold)
    ranked = list(dict.fromkeys(ranked))
    pool = set(pool if pool is not None else ranked)
    relevant = [n for n, i in enumerate(ranked, 1) if i in gold]
    first = min(relevant) if relevant else None
    result = {
        "first_relevant_rank": first,
        "MRR": 1 / first if first else 0.0,
        "gold_locations": len(gold),
        "pool_gold_count": len(pool & gold),
        "candidate_pool_coverage": len(pool & gold) / len(gold),
        "candidate_pool_any_gold": bool(pool & gold),
        "all_gold_recovered": gold <= set(ranked),
        "all_gold_rank": max(relevant) if gold <= set(ranked) else None,
    }
    for k in [1, 3, 5, 10]:
        hit = bool(set(ranked[:k]) & gold)
        result[f"Hit@{k}"] = int(hit)
        result[f"Top-{k}"] = int(hit)
        result[f"Recall@{k}"] = len(set(ranked[:k]) & gold) / len(gold)
    return result


def evaluate(arm, entities, gold, dataset):
    by_id = {e.entity_id: e for e in entities}
    ranked = arm["ranked_functions"]
    files = list(dict.fromkeys(by_id[i].path for i in ranked))
    pool_files = list(dict.fromkeys(e.path for e in entities))
    # Mapping methods to existing gold identifiers is offline only.
    methods = ranked
    pool = [e.entity_id for e in entities]
    if dataset == "defects4j" and gold.functions:
        from wp1.evaluation import method_matches

        def mapped(ids):
            out = []
            for ident in ids:
                matches = [g for g in gold.functions if method_matches(ident, g)]
                out.extend(matches or [ident])
            return out

        methods, pool = mapped(methods), mapped(pool)
    return {
        "file": ranking_metrics(files, gold.files, pool_files),
        "method": ranking_metrics(methods, gold.functions, pool)
        if not entities or any(e.kind != "file" for e in entities)
        else None,
        "gold_semantics": gold.semantics,
        "gold_unavailable_reason": gold.unavailable_reason,
        "excluded_reference_changes": list(gold.exclusions),
    }


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(
            (
                row["policy"],
                row["fusion"],
                row["admission"],
                row["evaluation_mode"],
                row.get("dataset"),
                row.get("language"),
            ),
            [],
        ).append(row)
    result = []
    for key, group in groups.items():
        for granularity in ["file", "method"]:
            valid = [r for r in group if r.get("metrics", {}).get(granularity) is not None]
            completed = [r for r in valid if r["status"] == "completed"]
            conditional = [
                r for r in completed if r["metrics"][granularity]["candidate_pool_any_gold"]
            ]
            for denominator, selection in [
                ("end_to_end_attempted_with_labels", valid),
                ("conditional_completed_pool_any_gold", conditional),
            ]:
                if not selection:
                    continue
                # End-to-end operational score assigns zero credit to incomplete
                # attempts by definition; the underlying unavailable metrics stay
                # null and status counts remain explicit.
                values = lambda metric: [
                    r["metrics"][granularity][metric] if r["status"] == "completed" else 0.0
                    for r in selection
                ]
                selected_completed = [r for r in selection if r["status"] == "completed"]
                costs = [
                    r["accounted_tokens"]
                    for r in selected_completed
                    if r.get("accounted_tokens") is not None
                ]
                out = {
                    "dataset": key[4],
                    "language": key[5],
                    "policy": key[0],
                    "fusion": key[1],
                    "admission": key[2],
                    "evaluation_mode": key[3],
                    "granularity": granularity,
                    "denominator": denominator,
                    "attempted": len(group),
                    "completed": len(completed),
                    "failed_or_incomplete": sum(r["status"] != "completed" for r in group),
                    "n": len(selection),
                    "MRR": statistics.mean(values("MRR")),
                    "Top-1": statistics.mean(values("Top-1")),
                    "Top-3": statistics.mean(values("Top-3")),
                    "Top-5": statistics.mean(values("Top-5")),
                    "Recall@5": statistics.mean(values("Recall@5")),
                    "candidate_pool_coverage": statistics.mean(
                        [r["metrics"][granularity]["candidate_pool_coverage"] for r in valid]
                    ),
                    "mean_tokens": statistics.mean(costs) if costs else None,
                    "median_tokens": statistics.median(costs) if costs else None,
                    "mean_full_inspections": statistics.mean(
                        [r["full_calls"] for r in selected_completed]
                    )
                    if selected_completed
                    else None,
                    "mean_calls": statistics.mean([r["calls"] for r in selected_completed])
                    if selected_completed
                    else None,
                    "mean_wall_seconds": statistics.mean(
                        [r["wall_seconds"] for r in selected_completed]
                    )
                    if selected_completed
                    else None,
                    "completion_rate": sum(r["status"] == "completed" for r in group) / len(group),
                    "budget_violation_rate": sum(bool(r.get("budget_exceeded")) for r in group)
                    / len(group),
                }
                baseline = {
                    (r["instance_id"], r["fusion"]): r
                    for r in rows
                    if r["policy"] == "full" and r["status"] == "completed"
                }
                savings = [
                    1
                    - r["accounted_tokens"]
                    / baseline[(r["instance_id"], r["fusion"])]["accounted_tokens"]
                    for r in selected_completed
                    if (r["instance_id"], r["fusion"]) in baseline
                    and baseline[(r["instance_id"], r["fusion"])]["accounted_tokens"] > 0
                ]
                out.update(
                    mean_token_savings=statistics.mean(savings) if savings else None,
                    median_token_savings=statistics.median(savings) if savings else None,
                )
                result.append(out)
    return result

"""Gold-free imported, FlexFL, lexical/BM25, saved-pool and plugin providers."""
from dataclasses import replace
import csv
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
from collections import Counter

from rave_fl.eval.schema import CodeEntity, CompatibilityError, beneath, file_hash
from rave_fl.eval.representations import index_source

PROVIDERS = ("flexfl", "imported", "saved", "lexical", "bm25", "semantic", "localizer")


def imported(instance, path):
    path = Path(path)
    if not path.is_file():
        raise CompatibilityError("Candidate file missing: " + str(path))
    if path.suffix == ".csv":
        rows = list(csv.DictReader(path.open()))
    else:
        payload = json.loads(path.read_text())
        rows = payload.get("candidates", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CompatibilityError("Candidate import must be an ordered list")
    out = []
    for n, r in enumerate(rows, 1):
        rel = r.get("path", "")
        source_path = beneath(instance.checkout, rel)
        lines = source_path.read_text().splitlines(keepends=True)
        start = int(r.get("start_line", 1))
        end = int(r.get("end_line", len(lines)))
        if not 1 <= start <= end <= len(lines):
            raise CompatibilityError("Imported candidate span outside buggy file")
        source = "".join(lines[start - 1 : end])
        ident = r.get("entity_id", r.get("function_id", rel))
        if r.get("source") is not None and r["source"] != source:
            raise CompatibilityError("Saved candidate source differs from buggy checkout")
        if r.get("source_sha256") and file_hash(source_path) != r["source_sha256"]:
            raise CompatibilityError("Candidate file checksum mismatch")
        ranks = r.get("ranks", {})
        scores = r.get("scores", {})
        if isinstance(ranks, str):
            ranks = json.loads(ranks or "{}")
        if isinstance(scores, str):
            scores = json.loads(scores or "{}")
        out.append(
            CodeEntity(
                ident,
                rel,
                start,
                end,
                source,
                float(r.get("prior", 1 / (n))),
                ranks,
                scores,
                r.get("kind", "function"),
            )
        )
    return out


def provide(instance, opts):
    provider = opts["candidate_provider"]
    count = opts["candidate_count"]
    if provider in {"imported", "saved"}:
        return (
            imported(instance, opts.get("candidate_file") or instance.candidate_file)[:count],
            {"provider": provider},
        )
    if provider == "flexfl":
        if instance.dataset != "defects4j" or instance.language != "java":
            raise CompatibilityError(
                "FlexFL provider needs Java Defects4J ranking CSVs; choose imported/BM25 for other inputs"
            )
        from rave_fl.expanded_pool import union_pool
        from rave_fl.source_spans import extract_pool_sources

        if not instance.flexfl_repo:
            raise CompatibilityError(
                "FlexFL provider requires flexfl_repo in the instance manifest"
            )
        cs, metadata = union_pool(
            Path(instance.flexfl_repo), instance.instance_id, opts["top_per_method"]
        )
        sources, paths = extract_pool_sources(Path(instance.checkout), metadata)
        entities = []
        for c in cs[:count]:
            m = metadata[c.function_id]
            path = paths[c.function_id]
            entities.append(
                CodeEntity(
                    c.function_id,
                    path,
                    m["start_line"],
                    m["end_line"],
                    sources[c.function_id],
                    c.semantic_score,
                    m["ranks"],
                    kind="method",
                )
            )
        return (
            entities,
            {
                "provider": "flexfl",
                "top_per_method": opts["top_per_method"],
                "rank_scores": "Historical ordinal priors, not measured semantic/coverage scores",
            },
        )
    pool = index_source(
        instance, opts["granularity"], opts["max_source_files"], opts["max_source_bytes"]
    )
    if not pool:
        raise CompatibilityError(
            "No source entities indexed; check source_root/language/production filters"
        )
    if provider in {"semantic", "localizer"}:
        factory = opts.get("candidate_factory")
        if not factory:
            raise CompatibilityError(
                provider
                + " requires --candidate-factory module:function; no implicit embeddings/downloads"
            )
        module, name = factory.split(":", 1)
        result = getattr(importlib.import_module(module), name)(
            instance.to_dict(), [e.__dict__ for e in pool], count
        )
        if not isinstance(result, list):
            raise CompatibilityError(
                "Candidate plugin must return ordered entity IDs, never gold-derived entities"
            )
        by_id = {e.entity_id: e for e in pool}
        if len(set(result)) != len(result) or any(i not in by_id for i in result):
            raise CompatibilityError("Candidate plugin returned unknown/duplicate IDs")
        return (
            [
                replace(by_id[i], prior=1 - n / max(1, len(result)))
                for n, i in enumerate(result[:count])
            ],
            {"provider": provider, "factory": factory},
        )

    def terms(text):
        return re.findall(r"[a-z][a-z0-9_]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())

    query = Counter(terms(instance.issue + "\n" + instance.evidence.text))
    docs = [Counter(terms(e.entity_id + "\n" + e.source)) for e in pool]
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(lengths) or 1
    df = Counter(t for d in docs for t in d)
    scores = []
    for e, d, length in zip(pool, docs, lengths):
        if provider == "lexical":
            score = sum(min(q, d[t]) for t, q in query.items()) / math.sqrt(max(1, length))
        else:
            score = sum(
                math.log(1 + (len(pool) - df[t] + 0.5) / (df[t] + 0.5))
                * d[t]
                * 2.5
                / (d[t] + 1.5 * (0.25 + 0.75 * length / average))
                for t in query
                if d[t]
            )
        scores.append((score, e))
    ordered = sorted(scores, key=lambda pair: (-pair[0], pair[1].entity_id))[:count]
    return (
        [
            replace(e, prior=1 - n / max(1, len(ordered)), scores={provider: score})
            for n, (score, e) in enumerate(ordered)
        ],
        {
            "provider": provider,
            "indexed_entities": len(pool),
            "query": "issue plus allowed failure evidence",
            "prior": "ordinal retrieval rank",
        },
    )

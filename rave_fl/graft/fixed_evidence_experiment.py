"""Fixed T0/T1 offline plans, evidence audits and synthetic collection."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from rave_fl.core import Action
from rave_fl.eval.adapters import load_instances, read_rows
from rave_fl.eval.backend import ExecutionLimit, PromptSettlement, Store, UnknownUsage
from rave_fl.eval.fusion import Fusion
from rave_fl.eval.metrics import load_gold
from rave_fl.eval.runner import implementation_identity, prepare_one, run_lock
from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint
from rave_fl.graft.controller import GraphBackend, GraftConfig, GraftProvider
from rave_fl.graft.evaluation import PROFILES, evaluate_profile
from rave_fl.graft.fixed_inputs import fixed_graph, portable_preparation_sha256
from rave_fl.graft.ollama_live import LiveOllamaPaths
from rave_fl.graft.prompt_settlement import SettlementCertificate, digest
from rave_fl.graft.runner import resource_identity, resources, validate_config
from rave_fl.graft.selection import ByteCounter, SYSTEM, query_text
from rave_fl.graft.topology import TopologyConfig, text_hash, topo_select
from rave_fl.pilot import write_json
from rave_fl.pilot_types import CandidateState


T0 = "dense_sparse_no_propagation"
T1 = "dense_sparse_with_propagation"
ARMS = {T0: TopologyConfig(propagate=False), T1: TopologyConfig(propagate=True)}
ARM_DIFF = {"propagate": {"T0": False, "T1": True}}
ACTION = Action.FULL_VERIFY  # GraphBackend's single-verdict transport.
LEVEL = 2


def configuration(value):
    allowed = {
        "benchmark",
        "input_allowance",
        "embeddings_dir",
        "tokenizer_dir",
        "labels_file",
        "per_arm_physical_budget",
        "per_arm_max_new_calls",
        "per_bug_max_new_calls",
        "prompt_settlement",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise CompatibilityError("Unknown fixed-evidence settings")
    benchmark = dict(value.get("benchmark", {}))
    frozen = {
        "admission": "strict",
        "granularity": "method",
        "representation": "l1",
        "batch_size": 1,
        "max_full": 0,
        "policies": ["fixed_l1"],
        "fusions": ["rrf"],
        "rrf_k": 60,
        "rounding": None,
        "tie": "original",
        "workers": 1,
        "require_observed_failure": True,
        "cache_mode": "off",
        "cache_dir": None,
        "check_provider": False,
        "dry_run": False,
        "allow_inference": False,
        "reservation_spec": None,
        "policy_budgets": {},
    }
    if any(key in benchmark and benchmark[key] != setting for key, setting in frozen.items()):
        raise CompatibilityError("Fixed-evidence settings cannot be overridden")
    # Shared validation handles dataset/provider options and resolves local resource paths.
    config = validate_config(
        {
            "benchmark": {**benchmark, **frozen, "run_id": "graft-fixed-evidence"},
            "graft": {
                "selector": "topo",
                "paths": False,
                "repair": False,
                "max_full": 0,
                "input_budget": value.get("input_allowance", 1024),
            },
            "graft_policies": [],
            "embeddings_dir": value.get("embeddings_dir"),
            "tokenizer_dir": value.get("tokenizer_dir"),
        }
    )
    opts = config["benchmark"]
    opts["cache_dir"] = None  # Shared CLI defaults can read RAVE_CACHE_DIR.
    opts.pop("policies")  # Required by the legacy config validator, unused by this collector.
    if not opts["dataset_file"]:
        raise CompatibilityError("Supply an explicit online dataset manifest")
    if opts["candidate_provider"] in {"semantic", "localizer"} or opts.get("candidate_factory"):
        raise CompatibilityError("Use a provider-free candidate source for fixed evidence")
    labels = value.get("labels_file")
    config["labels_file"] = str(Path(labels).resolve()) if labels else None
    settlement = value.get("prompt_settlement")
    if settlement is not None:
        if (not isinstance(settlement, dict) or set(settlement) != {"file", "sha256"}
                or not isinstance(settlement["file"], str) or not settlement["file"] or not digest(settlement["sha256"])):
            raise CompatibilityError("Prompt settlement requires a pinned certificate file and SHA-256")
        config["prompt_settlement"] = {**settlement, "file": str(Path(settlement["file"]).resolve())}
    if config["labels_file"] == opts["dataset_file"]:
        raise CompatibilityError("Online inputs and evaluation labels need separate files")
    if config["labels_file"] and any(
        config[key] and Path(config["labels_file"]).is_relative_to(config[key])
        for key in ("embeddings_dir", "tokenizer_dir")
    ):
        raise CompatibilityError("Keep evaluation labels outside resource fingerprint directories")
    for key, default in {
        "per_arm_physical_budget": opts["budget"],
        "per_arm_max_new_calls": opts["candidate_count"],
        "per_bug_max_new_calls": 2 * opts["candidate_count"],
    }.items():
        limit = value.get(key, default)
        if type(limit) is not int or limit < 1:
            raise CompatibilityError(f"{key} must be a positive integer")
        config[key] = limit
    return config


def fixed_schedule(candidates, entities):
    ids = [candidate.function_id for candidate in candidates]
    entity_ids = [entity.entity_id for entity in entities]
    if (
        not ids
        or len(ids) != len(set(ids))
        or len(entity_ids) != len(set(entity_ids))
        or set(ids) != set(entity_ids)
    ):
        raise CompatibilityError("Candidate schedule must cover each unique prepared entity once")
    if any(type(candidate.rank) is not int or candidate.rank < 1 for candidate in candidates):
        raise CompatibilityError("Prior ranks must be positive integers")
    return sorted(candidates, key=lambda candidate: candidate.rank)


class FixedGraphBackend(GraphBackend):
    def planning(self, candidates, action, level):
        settlement = getattr(self, "prompt_settlement", None)
        if settlement:
            return settlement.upper, self.opts["context_window"] + settlement.output_cap
        return super().planning(candidates, action, level)

    def request_identity(self, candidates, action, level, payload):
        return {
            **super().request_identity(candidates, action, level, payload),
            "experiment": "graft-fixed-evidence-v1",
            "instance_id": self.instance.instance_id,
        }


@dataclass
class PreparedPair:
    instance: object
    entities: list
    schedule: list
    graph: object
    counter: object
    selections: dict


def arm_config(config, arm):
    return GraftConfig(**{**config["graft"], "propagation": ARMS[arm].propagate})


def physical_envelope(rows):
    """Peak serial exposure: settled prefix upper bound plus the next full hold."""
    prefix = peak = all_windows = 0
    for row in rows:
        hold = row["prompt_accounting"]["physical_reservation_tokens"]
        peak = max(peak, prefix + hold)
        # Execution and journal validation both enforce these per-action bounds.
        prefix += row.get("settled_upper_tokens", min(hold, row["logical_reservation"]))
        all_windows += hold
    return {"serial_peak_tokens": peak, "sum_full_window_reservations_tokens": all_windows}


def prompt_accounting(payload, selection, counter):
    messages = payload["messages"]
    if counter.count(messages) != selection.input_units:
        raise CompatibilityError("Complete request count differs from packed evidence")
    options = payload["options"]
    return {
        "messages_sha256": fingerprint(messages),
        "system_content_utf8_bytes": len(messages[0]["content"].encode("utf-8")),
        "user_content_utf8_bytes": len(messages[1]["content"].encode("utf-8")),
        "byte_proxy_units": ByteCounter().count(messages),
        "local_chat_template_tokens": selection.input_units if counter.exact else None,
        "serving_template_tokens": None,  # No provider-equivalence certificate in Phase 3.
        "generation_output_cap_tokens": options["num_predict"],
        "context_window_tokens": options["num_ctx"],
        "physical_reservation_tokens": options["num_ctx"] + options["num_predict"],
    }


def compare_evidence(graph, target, left, right):
    shared_diagnostics = ("seed_sha256", "adjacency_sha256", "embedding_hash", "seed_mass")
    if any(left.diagnostics[key] != right.diagnostics[key] for key in shared_diagnostics):
        raise CompatibilityError("Fixed arms produced different pre-propagation inputs")
    left_ids, right_ids = set(left.ids), set(right.ids)
    union = left_ids | right_ids
    intersection = left_ids & right_ids
    ranks = [
        {node_id: rank for rank, node_id in enumerate(
            sorted(s.scores, key=lambda node_id: (-s.scores[node_id], node_id)), start=1
        )}
        for s in (left, right)
    ]
    nodes = graph.by_id
    return {
        "target": target,
        "query_sha256": text_hash(query_text(graph, target)),
        "shared_seed_sha256": left.diagnostics["seed_sha256"],
        "shared_adjacency_sha256": left.diagnostics["adjacency_sha256"],
        "common_ids": sorted(intersection),
        "t0_only_ids": sorted(left_ids - right_ids),
        "t1_only_ids": sorted(right_ids - left_ids),
        "jaccard": len(intersection) / len(union),
        "evidence_nodes": [
            {
                "id": node_id,
                "kind": nodes[node_id].kind,
                "owner": nodes[node_id].owner,
                "origin": asdict(nodes[node_id].origin),
                "exact_text_sha256": text_hash(nodes[node_id].text),
                "exact_text_utf8_bytes": len(nodes[node_id].text.encode()),
                "t0_score": left.scores[node_id],
                "t1_score": right.scores[node_id],
                "t0_rank": ranks[0][node_id],
                "t1_rank": ranks[1][node_id],
            }
            for node_id in sorted(union)
        ],
        "source_relations": [
            asdict(edge) for edge in graph.edges if {edge.source, edge.target} <= union
        ],
    }


def prepare_pair(instance, config, certificate=None):
    opts = config["benchmark"]
    instance, entities, candidates, _, _, prepared = prepare_one(instance, opts)
    schedule = fixed_schedule(candidates, entities)
    if any(entity.kind not in {"method", "function"} for entity in entities):
        raise CompatibilityError("Fixed evidence requires method/function candidates")
    graph = fixed_graph(instance, entities)
    counter, embeddings = resources(config, instance.instance_id)
    embeddings.validate_for_graph(graph)
    schedule_rows = [{"target": c.function_id, "prior_rank": c.rank, "prior_score": c.prior} for c in schedule]
    if certificate:
        certificate.verify_pair(instance.instance_id, prepared, graph, embeddings, schedule_rows)
    selections = {}
    arms = {}
    # Planning reads the backend kind only. No provider or Store is instantiated.
    planning_context = SimpleNamespace(kind=opts["backend"])
    for name, topology in ARMS.items():
        backend = FixedGraphBackend(
            instance, entities, planning_context, None, opts, opts["budget"]
        )
        settings = arm_config(config, name)
        rows, selections[name] = [], {}
        for candidate in schedule:
            target = candidate.function_id
            try:
                selected = topo_select(
                    graph,
                    target,
                    embeddings,
                    settings.input_budget,
                    counter=counter,
                    config=topology,
                    audit=True,
                )
                selections[name][target] = selected
                row = {
                    "target": target,
                    "status": selected.status,
                    "reason": selected.reason,
                    "selected_ids": selected.ids,
                    "selected_node_count": len(selected.ids),
                    "selection_sha256": selected.hash,
                    "evidence_message_sha256": text_hash(selected.text),
                    "evidence_message_utf8_bytes": len(selected.text.encode()),
                    "input_units": selected.input_units,
                    "input_allowance": selected.budget,
                    "counter": selected.counter,
                    "exact_template_tokens": selected.exact_template_tokens,
                    "budget_rejected_ids": selected.diagnostics.get("budget_rejected_ids", []),
                    "propagation": {
                        key: selected.diagnostics[key]
                        for key in ("iterations", "converged", "residual", "seed_mass", "score_mass")
                    },
                }
                if selected.status == "ready":
                    backend.configure(graph, selected, settings, counter)
                    payload = backend.payload([candidate], ACTION, LEVEL)
                    backend.prompt_settlement = certificate.for_request(instance.instance_id, name, target, payload, counter) if certificate else None
                    predicted, reserved = backend.planning([candidate], ACTION, LEVEL)
                    row.update(
                        request=payload,
                        prompt_accounting=prompt_accounting(payload, selected, counter),
                        predicted_tokens=predicted,
                        logical_reservation=reserved,
                    )
                    if backend.prompt_settlement:
                        settlement = backend.prompt_settlement
                        row["prompt_settlement"] = asdict(settlement)
                        row["settled_upper_tokens"] = settlement.upper
                        row["prompt_accounting"]["serving_template_tokens"] = settlement.prompt_tokens
                    if predicted > opts["context_window"]:
                        row.update(status="infeasible", reason="Conservative context check")
            except (ValueError, RuntimeError) as exc:
                row = {"target": target, "status": "infeasible", "reason": str(exc)}
            rows.append(row)
        arms[name] = rows
    comparisons = []
    for a, b in zip(arms[T0], arms[T1], strict=True):
        target = a["target"]
        left, right = selections[T0].get(target), selections[T1].get(target)
        if left is not None and right is not None:
            comparison = compare_evidence(graph, target, left, right)
            comparison["packing_units_delta_t1_minus_t0"] = right.input_units - left.input_units
            comparisons.append(comparison)
        if "request" in a and "request" in b:
            first, second = a["request"], b["request"]
            if first != {**second, "messages": [second["messages"][0], first["messages"][1]]}:
                raise CompatibilityError("Paired requests differ outside the rendered evidence")
    provenance = embeddings.value.get("provenance") if not embeddings.synthetic else None
    calls = len(schedule)
    planned = all("request" in row for rows in arms.values() for row in rows)
    per_arm = {name: physical_envelope(rows) for name, rows in arms.items()} if planned else {}
    pair_envelope = physical_envelope(row for name in ARMS for row in arms[name]) if planned else None
    reasons = []
    if any(row["status"] != "ready" for rows in arms.values() for row in rows):
        reasons.append("At least one scheduled target is infeasible")
    if certificate and any(bound["serial_peak_tokens"] > opts["budget"] for bound in per_arm.values()):
        reasons.append("Full schedule exceeds prompt-aware serial per-arm logical envelope")
    if not certificate and any(
        sum(row.get("logical_reservation", 0) for row in rows) > opts["budget"]
        for rows in arms.values()
    ):
        reasons.append("Full schedule exceeds conservative per-arm logical envelope")
    if any(bound["serial_peak_tokens"] > config["per_arm_physical_budget"] for bound in per_arm.values()):
        reasons.append("Full schedule exceeds conservative serial per-arm physical envelope")
    if pair_envelope and pair_envelope["serial_peak_tokens"] > opts["per_bug_physical_budget"]:
        reasons.append("Pair exceeds conservative serial per-bug physical envelope")
    if calls > config["per_arm_max_new_calls"] or 2 * calls > config["per_bug_max_new_calls"]:
        reasons.append("Schedule exceeds an arm or bug call limit")
    record = {
        "instance_id": instance.instance_id,
        "status": "infeasible" if reasons else "ready",
        "reasons": reasons,
        "prepared_sha256": fingerprint(prepared),
        "portable_preparation_sha256": portable_preparation_sha256(prepared, instance),
        "pool_sha256": prepared["pool_hash"],
        "graph_sha256": graph.hash,
        "embeddings_sha256": embeddings.hash,
        "embedding_model_id": embeddings.model_id,
        "embedding_dimension": embeddings.dimension,
        "embedding_text_count": len(embeddings.value["texts"]),
        "embedding_provenance": {
            key: provenance[key] for key in (
                "checkpoint_revision", "checkpoint_files", "graph_sha256", "source_revision",
                "max_length", "max_observed_tokens", "batch_size", "device_requested", "packages",
            ) if key in provenance
        } if provenance is not None else None,
        "embedding_synthetic": embeddings.synthetic,
        "evidence_sha256": instance.evidence.sha256,
        "revision": instance.revision,
        "schedule": schedule_rows,
        "arms": arms,
        "evidence_comparisons": comparisons,
        "max_fresh_calls": 2 * calls,
        "physical_reservation_upper": pair_envelope["serial_peak_tokens"] if pair_envelope else None,
        "physical_envelope_by_arm": per_arm,
        "physical_envelope": pair_envelope,
    }
    return record, PreparedPair(instance, entities, schedule, graph, counter, selections)


def prepare(config):
    opts = config["benchmark"]
    dataset_hash = file_hash(opts["dataset_file"])
    rows, metadata = read_rows(opts["dataset_file"])
    if any({"gold", "gold_file", "patch", "fix_patch"} & row.keys() for row in [metadata, *rows]):
        raise CompatibilityError("Remove labels and reference patches from the online manifest")
    arm_values = {name: asdict(topology) for name, topology in ARMS.items()}
    a, b = arm_values[T0], arm_values[T1]
    if (
        a["propagate"] is not False
        or b["propagate"] is not True
        or {k for k in a | b if a.get(k) != b.get(k)} != {"propagate"}
    ):
        raise CompatibilityError("The fixed arms must differ only in propagation")
    if a["typed"] or any(config["graft"][k] for k in ("paths", "repair", "typed_topology", "max_full")):
        raise CompatibilityError("Paths, repair, typed topology and method Full must stay disabled")
    instances, source = load_instances(opts)
    if source["dataset_sha256"] != dataset_hash:
        raise CompatibilityError("Online manifest changed during validation")
    certificate = SettlementCertificate(config) if config.get("prompt_settlement") else None
    if certificate and tuple(i.instance_id for i in instances) != certificate.cohort:
        raise CompatibilityError("The selected cohort differs from this schedule's settlement certificate")
    records, pairs = [], {}
    for instance in instances:
        try:
            if instance.provenance.get("skip_reason"):
                raise CompatibilityError(instance.provenance["skip_reason"])
            record, pair = prepare_pair(instance, config, certificate)
            pairs[instance.instance_id] = pair
        except (ValueError, OSError, ImportError) as exc:
            record = {
                "instance_id": instance.instance_id,
                "status": "infeasible",
                "reasons": [str(exc)],
                "schedule": [],
                "arms": {},
                "max_fresh_calls": 0,
                "physical_reservation_upper": 0,
            }
        records.append(record)
    ready = [r for r in records if r["status"] == "ready"]
    calls = sum(r["max_fresh_calls"] for r in ready)
    envelope = physical_envelope(
        row for record in ready for name in ARMS for row in record["arms"][name]
    )
    reserve = envelope["serial_peak_tokens"]
    admitted = calls <= opts["max_new_calls"] and reserve <= opts["physical_budget"]
    if not admitted:
        for record in ready:
            record["status"] = "infeasible"
            record["reasons"].append("Combined schedule exceeds whole-run token or call envelope")
    if certificate and admitted and len(ready) != len(records):
        admitted = False
        for record in ready:
            record["status"] = "infeasible"
            record["reasons"].append("The other audited defect did not pass preflight")
    if certificate and not admitted:
        calls, reserve = 0, 0
        envelope = physical_envelope([])
    plan = {
        "schema": "graft-fixed-evidence-plan-v1",
        "config": config,
        "dataset": source,
        "implementation": implementation_identity(),
        "resources": resource_identity(config),
        "arm_configs": arm_values,
        "arm_diff": ARM_DIFF,
        "system_prompt_sha256": text_hash(SYSTEM),
        "pairs": records,
        "planned_fresh_calls_upper": calls,
        "physical_reservation_upper": reserve,
        "physical_envelope": envelope,
        "run_admitted": admitted,
        "new_model_calls": 0,
    }
    if certificate:
        plan["settlement_serving_identity"] = certificate.identity
    if file_hash(opts["dataset_file"]) != dataset_hash:
        raise CompatibilityError("Online manifest changed during planning")
    # Round trip tuples in selections so the in-memory and saved manifests compare exactly.
    return json.loads(json.dumps(plan, allow_nan=False)), pairs


def requests(config, pairs, plan, provider, store):
    result = {}
    opts = config["benchmark"]
    for record in plan["pairs"]:
        if record["status"] != "ready":
            continue
        pair = pairs[record["instance_id"]]
        for name in ARMS:
            settings = arm_config(config, name)
            backend = FixedGraphBackend(
                pair.instance, pair.entities, provider, store, opts, opts["budget"]
            )
            backend.arm_physical_budget = config["per_arm_physical_budget"]
            actions = []
            for candidate, row in zip(pair.schedule, record["arms"][name], strict=True):
                selection = pair.selections[name][candidate.function_id]
                backend.configure(pair.graph, selection, settings, pair.counter)
                payload = backend.payload([candidate], ACTION, LEVEL)
                settlement = PromptSettlement(**row["prompt_settlement"]) if "prompt_settlement" in row else None
                backend.prompt_settlement = settlement
                if settlement:
                    settlement.verify(payload, provider)
                if (
                    payload != row["request"] or row["target"] != candidate.function_id
                    or backend.planning([candidate], ACTION, LEVEL)
                    != (row["predicted_tokens"], row["logical_reservation"])
                ):
                    raise CompatibilityError("Prepared request changed after planning")
                key = fingerprint(backend.request_identity([candidate], ACTION, LEVEL, payload))
                actions.append((candidate, selection, key, payload, row["logical_reservation"], settlement))
            result[pair.instance.instance_id, name] = (backend, settings, actions)
    return result


def journal_identity(store, actions, *, reconcile=False, live_model=None, linux_attestation=None):
    expected = {
        key: (bug, payload, logical, settlement)
        for (bug, _), (_, _, group) in actions.items()
        for _, _, key, payload, logical, settlement in group
    }
    if len(expected) != sum(len(group) for _, _, group in actions.values()):
        raise CompatibilityError("Fixed plan produced duplicate request identities")
    rows = store.db.execute(
        "SELECT key,instance,status,request,response,reservation,input,output FROM calls ORDER BY key"
    ).fetchall()
    requests_logged = {}
    responses_logged = {}
    if store.ledger.exists():
        for line in store.ledger.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "request":
                key = event["fingerprint"]
                if key in requests_logged:
                    raise CompatibilityError("Duplicate fresh request in raw ledger")
                requests_logged[key] = event
            elif event["event"] == "response":
                key = event["fingerprint"]
                if key in responses_logged:
                    raise CompatibilityError("Duplicate response in raw ledger")
                responses_logged[key] = event
    if (
        set(requests_logged) != {row[0] for row in rows}
        or responses_logged.keys() - requests_logged.keys()
    ):
        raise CompatibilityError("Journal and raw request ledger disagree")
    if list(requests_logged) != list(expected)[:len(rows)]:
        raise CompatibilityError("Journal requests are not a prefix of the fixed schedule")
    bindings = {}
    if any(item[-1] for item in expected.values()):
        bindings = dict(store.db.execute("SELECT key,binding FROM prompt_settlements"))
        if set(bindings) != {row[0] for row in rows}:
            raise CompatibilityError("Journal settlement bindings differ from paid or pending requests")
    linux = any(item[-1] and item[-1].serving_identity.get("system") == "Linux" for item in expected.values())
    gpu_rows = dict(store.db.execute("SELECT key,receipt FROM gpu_attestations")) if linux else {}
    if linux and set(gpu_rows) != {row[0] for row in rows if row[2] == "complete"}:
        raise CompatibilityError("Journal GPU receipts differ from completed Linux requests")
    recovered = []
    for key, bug, status, raw_request, raw_response, reserve, inp, out in rows:
        if status != "complete":
            raise UnknownUsage(f"Journal has unresolved or unsafe usage for {bug}: {status}")
        payload, response = json.loads(raw_request), json.loads(raw_response)
        event = requests_logged[key]
        limits = payload["options"]
        planned_bug, planned_payload, logical, settlement = expected[key]
        logged = responses_logged.get(key)
        if (
            (planned_bug, planned_payload) != (bug, payload)
            or event["request"] != payload
            or event["instance"] != bug
            or event["reservation"] != reserve
            or reserve != limits["num_ctx"] + limits["num_predict"]
            or not (
                type(inp) is type(out) is int
                and 0 <= inp <= limits["num_ctx"]
                and 0 <= out <= limits["num_predict"]
                and inp + out <= logical
            )
            or response.get("done") is not True
            or (live_model is not None and response.get("model") != live_model)
            or (response.get("prompt_eval_count"), response.get("eval_count")) != (inp, out)
            or (logged is not None and logged["response"] != response)
        ):
            raise CompatibilityError(
                "Saved request, response or accounting differs from the fixed plan"
            )
        if settlement:
            binding = asdict(settlement)
            if (json.loads(bindings[key]) != binding or event.get("prompt_settlement") != binding
                    or fingerprint(payload) != settlement.request_sha256
                    or event.get("model_digest") != settlement.serving_identity["model_manifest_sha256"]
                    or inp != settlement.prompt_tokens or inp + out > settlement.upper
                    or (logged is not None and logged.get("settled_upper_tokens") != settlement.upper)):
                raise CompatibilityError("Journal settled usage or provenance differs from the verified request")
            if logged is None:
                if not reconcile or key != next(reversed(requests_logged)):
                    raise CompatibilityError("Missing settlement response event")
                recovered.append({"event": "response", "fingerprint": key, "instance": bug,
                                  "response": response, "settled_upper_tokens": settlement.upper,
                                  **({"gpu_attestation": json.loads(gpu_rows[key])} if linux else {}),
                                  "recovered_from_durable_journal": True})
            if linux:
                gpu = json.loads(gpu_rows[key])
                placement = gpu.get("compute_processes") or []
                listener = gpu.get("listener") or {}
                if (type(gpu.get("model_size")) is not int or gpu["model_size"] <= 0
                        or gpu.get("size_vram") != gpu["model_size"] or gpu.get("context_length") != limits["num_ctx"]
                        or not placement or type(listener.get("pid")) is not int
                        or (logged is not None and logged.get("gpu_attestation") != gpu)):
                    raise CompatibilityError("Missing or invalid Linux GPU attestation in paid response ledger")
                if linux_attestation:
                    static = {k: v for k, v in listener.items() if k not in {"pid", "started"}}
                    baseline = {k: v for k, v in linux_attestation["listener"].items() if k not in {"pid", "started"}}
                    allowed = {row["uuid"] for row in linux_attestation["a100_inventory"]}
                    if static != baseline or any(row.get("gpu_uuid") not in allowed
                            or type(row.get("allocated_mib")) is not int or row["allocated_mib"] <= 0
                            or not row.get("cuda_libraries") for row in placement):
                        raise CompatibilityError("Linux GPU receipt differs from the sealed serving or device identity")
    for event in recovered:
        store.event(event)
    return fingerprint({"calls": rows, "settlements": bindings, **({"gpu_attestations": gpu_rows} if linux else {})}
                       ) if bindings or linux else fingerprint(rows)


def collect_arm(pair, name, action_group):
    backend, settings, actions = action_group
    states = {}
    calls = []
    for candidate, selection, key, _, _, settlement in actions:
        backend.configure(pair.graph, selection, settings, pair.counter)
        backend.prompt_settlement = settlement
        result = backend.execute([candidate], ACTION, LEVEL)
        valid = (
            result.error is None
            and len(result.observations) == 1
            and result.observations[0].function_id == candidate.function_id
            and result.observations[0].valid
        )
        calls.append(
            {"target": candidate.function_id, "request_key": key, "valid": valid, **asdict(result)}
        )
        if valid:
            states[candidate.function_id] = CandidateState(1, result.observations[0])
    complete = len(states) == len(pair.schedule)
    ranking = None
    if complete:
        fusion = Fusion("rrf", rrf_k=60, tie="original")
        ranking = [
            c.function_id for c in fusion.order(pair.schedule, fusion.scores(pair.schedule, states))
        ]
    physical = sum(event["actual_tokens"] for event in backend.events if event["reuse"] in {"fresh", "journal"})
    synthetic = backend.provider.metadata["synthetic"]
    return {
        "arm": name,
        "execution_status": "complete" if complete else "incomplete",
        "logical_tokens": backend.spent,
        "provider_reported_input_tokens": sum(call["input_tokens"] for call in calls),
        "provider_reported_output_tokens": sum(call["output_tokens"] for call in calls),
        # Shared cache is off. Journal reuse was paid once in an earlier invocation.
        "simulated_physical_tokens": physical if synthetic else 0,
        "real_physical_tokens": 0 if synthetic else physical,
        "calls": calls,
        "accounting": backend.events,
        "ranked_functions": ranking,
    }


def evaluate_pairs(results, pairs, labels_file):
    if not labels_file:
        return None
    labels_sha256 = file_hash(labels_file)
    labels, _ = read_rows(labels_file)
    if any("gold" not in label or "gold_file" in label for label in labels):
        raise CompatibilityError("Evaluation manifest requires inline gold objects")
    ids = [label.get("instance_id") or label.get("bug") for label in labels]
    if any(not isinstance(ident, str) or not ident for ident in ids) or len(ids) != len(set(ids)):
        raise CompatibilityError("Evaluation manifest needs unique defect IDs")
    source = {"dataset_file": labels_file, "path_base": str(Path(labels_file).parent)}
    for row in results:
        if row["pair_status"] != "complete" or row["instance_id"] not in ids:
            continue
        pair = pairs[row["instance_id"]]
        gold = load_gold(pair.instance, source)
        profiles = PROFILES if pair.instance.dataset == "defects4j" else ("legacy_v1",)
        for arm in row["arms"].values():
            arm["metrics"] = {
                profile: evaluate_profile(arm, pair.entities, gold, pair.instance.dataset, profile)
                for profile in profiles
            }
    if labels_sha256 != file_hash(labels_file):
        raise CompatibilityError("Evaluation labels changed during evaluation")
    return labels_sha256


def paired_differences(row, record):
    """Retain signed T1 minus T0 differences, with explicit missing measurements."""
    left, right = (row["arms"][name] for name in (T0, T1))
    expected = len(record["schedule"])
    both_collected = expected > 0 and len(left["calls"]) == len(right["calls"]) == expected
    usage_keys = (
        "provider_reported_input_tokens", "provider_reported_output_tokens", "logical_tokens",
        "simulated_physical_tokens", "real_physical_tokens",
    )
    metrics = {}
    for profile, a in left.get("metrics", {}).items():
        b = right["metrics"][profile]
        metrics[profile] = {
            level: {key: b[level][key] - a[level][key] for key in ("MRR", "Top-1", "Top-5")}
            if a[level] is not None and b[level] is not None else None
            for level in ("method", "file")
        }
    return {
        "direction": "T1_minus_T0",
        "usage": {key: right[key] - left[key] for key in usage_keys} if both_collected else None,
        "accuracy": metrics or None,
    }


def run(config, directory, *, resume=False, allow_inference=False, cohort_binding=None,
        expected_plan_sha256=None, live_paths=None, prepared=None):
    opts = config["benchmark"]
    live = opts["backend"] != "fake"
    if live:
        if not allow_inference:
            raise CompatibilityError("Real inference requires explicit --allow-inference")
        if (opts["backend"] != "ollama" or not config.get("prompt_settlement")
                or not isinstance(live_paths, LiveOllamaPaths) or not cohort_binding
                or not cohort_binding.get("live_attestation") or expected_plan_sha256 is None):
            raise CompatibilityError("Real inference requires an attested portable cohort and its settlement certificate")
    plan, pairs = prepared if prepared is not None else prepare(config)
    if expected_plan_sha256 is not None and fingerprint(plan) != expected_plan_sha256:
        raise CompatibilityError("Prepared plan changed after cohort validation")
    if cohort_binding is not None:
        labels = config["labels_file"]
        if (file_hash(labels) if labels else None) != cohort_binding["labels_sha256"]:
            raise CompatibilityError("Offline labels differ from the frozen cohort")
        split = opts.get("split_manifest")
        if (file_hash(split) if split else None) != cohort_binding["split_manifest_sha256"]:
            raise CompatibilityError("Split manifest differs from the frozen cohort")
    if live and (not plan["run_admitted"] or any(pair["status"] != "ready" for pair in plan["pairs"])):
        raise CompatibilityError("Live collection requires the complete admitted schedule")
    provider = live_paths.provider(config) if live else GraftProvider(opts)
    if live:
        provider.preflight()
        if not provider.matches_cohort(cohort_binding["live_attestation"]):
            raise CompatibilityError("Active serving identity differs from the sealed cohort")
    directory = Path(directory)
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    elif not directory.is_dir():
        raise CompatibilityError("Resume requires an existing run directory")
    with run_lock(directory):
        if not live:
            provider.preflight()
        identity = {"plan": plan, "provider": cohort_binding["live_attestation"] if live else provider.metadata}
        if cohort_binding is not None:
            identity["cohort_binding"] = cohort_binding
        manifest = directory / "manifest.json"
        result_path = directory / "results.json"
        if live:
            provider.claim_collection(cohort_binding["cohort_sha256"], directory, resume=resume)
        if resume:
            if json.loads(manifest.read_text()) != identity:
                raise CompatibilityError("Resume manifest differs from current fixed plan")
            if not (directory / "journal.sqlite3").is_file():
                raise CompatibilityError("Resume journal is missing")
        else:
            write_json(manifest, identity)
        actions = requests(config, pairs, plan, provider, None)
        settlements = {key: settlement for _, _, group in actions.values()
                       for _, _, key, _, _, settlement in group if settlement}
        store = Store(directory, opts, settlements=settlements)
        try:
            for backend, _, _ in actions.values():
                backend.store = store
            live_model = opts["model"] if live else None
            linux_attestation = cohort_binding["live_attestation"] if live and opts["device"] == "a100" else None
            before_hash = journal_identity(store, actions, reconcile=resume, live_model=live_model,
                                           linux_attestation=linux_attestation)
            before_calls = store.snapshot()["new_calls"]
            if resume and result_path.exists():
                saved = json.loads(result_path.read_text())
                if not isinstance(saved, dict):
                    raise CompatibilityError("Completed result is malformed")
                result_hash = saved.pop("sha256", None)
                if (
                    result_hash != fingerprint(saved)
                    or saved.get("journal_sha256") != before_hash
                    or saved.get("manifest_sha256") != fingerprint(identity)
                ):
                    raise CompatibilityError("Completed result or journal checksum mismatch")
                return {**saved, "new_calls_this_invocation": 0}
            results = []
            for record in plan["pairs"]:
                bug = record["instance_id"]
                row = {
                    "instance_id": bug,
                    "pair_status": "incomplete",
                    "arms": {},
                    "reasons": record["reasons"],
                }
                if record["status"] == "ready":
                    row["arms"] = {
                        name: collect_arm(pairs[bug], name, actions[bug, name]) for name in ARMS
                    }
                    if all(arm["execution_status"] == "complete" for arm in row["arms"].values()):
                        row["pair_status"] = "complete"
                else:
                    row["arms"] = {
                        name: {
                            "arm": name,
                            "execution_status": "infeasible",
                            "logical_tokens": 0,
                            "provider_reported_input_tokens": 0,
                            "provider_reported_output_tokens": 0,
                            "simulated_physical_tokens": 0,
                            "real_physical_tokens": 0,
                            "calls": [],
                            "accounting": [],
                            "ranked_functions": None,
                        }
                        for name in ARMS
                    }
                for arm in row["arms"].values():
                    arm["paired_status"] = row["pair_status"]
                results.append(row)
            usage = store.snapshot()
            collected = [arm for row in results for arm in row["arms"].values()]
            if sum(arm["simulated_physical_tokens"] for arm in collected) != usage["simulated_provider_tokens"]:
                raise CompatibilityError("Collected arm usage and journal charges differ")
            if sum(arm["real_physical_tokens"] for arm in collected) != usage["physical_inference_tokens"]:
                raise CompatibilityError("Collected real usage and journal charges differ")
            if settlements and sum(event["settled_upper_tokens"] for arm in collected
                                   for event in arm["accounting"]) != usage["settled_upper_tokens"]:
                raise CompatibilityError("Collected arm settlement and journal bounds differ")
            labels_hash = evaluate_pairs(results, pairs, config["labels_file"])
            if cohort_binding is not None and labels_hash != cohort_binding["labels_sha256"]:
                raise CompatibilityError("Offline labels changed during the run")
            for row, record in zip(results, plan["pairs"], strict=True):
                row["paired_differences"] = paired_differences(row, record)
            result = {
                "schema": "graft-fixed-evidence-result-v1",
                "synthetic": not live,
                "manifest_sha256": fingerprint(identity),
                "pairs": results,
                "spending": usage,
                "journal_sha256": journal_identity(store, actions, live_model=live_model, linux_attestation=linux_attestation),
                "labels_sha256": labels_hash,
            }
            write_json(result_path, {**result, "sha256": fingerprint(result)})
            return {**result, "new_calls_this_invocation": usage["new_calls"] - before_calls}
        finally:
            write_json(directory / "accounting.json", store.snapshot())
            store.db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run", "resume"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = configuration(json.loads(args.config.read_text()))
        if args.command == "dry-run":
            if config.get("prompt_settlement"):
                raise CompatibilityError("Use the portable dry-run to verify local serving provenance")
            if args.output.exists():
                raise CompatibilityError("Choose a new plan file")
            plan, _ = prepare(config)
            write_json(args.output, plan)
            ready = all(pair["status"] == "ready" for pair in plan["pairs"])
            print(json.dumps({"plan": str(args.output), "ready": ready, "new_model_calls": 0}))
            return 0 if ready else 3
        result = run(
            config,
            args.output,
            resume=args.command == "resume",
            allow_inference=args.allow_inference,
        )
        print(
            json.dumps(
                {
                    "results": str(args.output / "results.json"),
                    "pairs": {p["instance_id"]: p["pair_status"] for p in result["pairs"]},
                    "spending": result["spending"],
                    "new_calls_this_invocation": result["new_calls_this_invocation"],
                }
            )
        )
        return 0 if all(p["pair_status"] == "complete" for p in result["pairs"]) else 3
    except (
        ValueError,
        OSError,
        ImportError,
        sqlite3.DatabaseError,
        ExecutionLimit,
        UnknownUsage,
    ) as exc:
        parser.exit(2, f"Fixed-evidence experiment: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit evidence stages and a bounded, untrained experimental router.

Legacy numeric level 2 still means method Full. Graph inspections use a single-
verdict transport parser but are projected to level 1 in the historical ranker.
They are never advertised as a superset of a complete target method.
"""
from dataclasses import dataclass, asdict, replace
from enum import Enum
import json
import math
import re
from rave_fl.core import Action
from rave_fl.pilot_types import CandidateState
from rave_fl.eval.backend import Backend, ExecutionLimit, Provider
from rave_fl.eval.schema import CompatibilityError, fingerprint
from rave_fl.graft.selection import SYSTEM, ByteCounter, relevance_select, package
from rave_fl.graft.focused import focus_select
from rave_fl.graft.topology import topo_select, TopologyConfig
from rave_fl.graft.paths import preserve_paths
from rave_fl.graft.repair import repair_support, missing_support


class EvidenceStage(str, Enum):
    SKETCH = "sketch"
    GRAPH = "graph"
    METHOD_FULL = "method_full"


class GraftProvider(Provider):
    """The fixture supplies the required graph schema; real transport is unchanged."""

    def preflight(self):
        metadata = super().preflight()
        if self.kind == "fake":
            metadata["provider_version"] = "graft-fake-v1"
        return metadata

    def execute(self, payload, ids, action):
        result = super().execute(payload, ids, action)
        if self.kind == "fake" and payload["messages"][0]["content"] == SYSTEM:
            value = json.loads(result["message"]["content"])
            value.update(uncertainty=0.3, missing_evidence="")
            result["message"]["content"] = json.dumps(value)
            from rave_fl.core import approx_tokens

            result["eval_count"] = min(
                payload["options"]["num_predict"],
                max(1, approx_tokens(result["message"]["content"])),
            )
        return result


@dataclass(frozen=True)
class GraftConfig:
    selector: str = "bm25"
    input_budget: int = 4096
    paths: bool = False
    repair: bool = True
    max_graph: int = 2
    max_full: int = 1
    uncertainty_threshold: float = 0.45
    top_k: int = 5
    typed_topology: bool = False
    propagation: bool = True
    guard_unsupported_graph_downgrade: bool = True
    failure_name_anchor: bool = False

    def __post_init__(self):
        if self.selector not in {"bm25", "topo"}:
            raise ValueError("Explicit bm25 or topo selector required")
        if any(type(v) is not int or v < 0 for v in [self.max_graph, self.max_full]):
            raise ValueError("Nonnegative action caps required")
        if (
            type(self.input_budget) is not int
            or self.input_budget < 1
            or type(self.top_k) is not int
            or self.top_k < 1
        ):
            raise ValueError("Positive input allowance and top-k required")
        if (
            not math.isfinite(self.uncertainty_threshold)
            or not 0 <= self.uncertainty_threshold <= 1
        ):
            raise ValueError("Invalid uncertainty threshold")
        if any(
            type(v) is not bool
            for v in [
                self.paths, self.repair, self.typed_topology, self.propagation,
                self.guard_unsupported_graph_downgrade, self.failure_name_anchor,
            ]
        ):
            raise ValueError("Boolean ablation flags required")


def select_evidence(graph, target, config, counter=None, embeddings=None):
    if config.selector == "topo":
        result = topo_select(
            graph,
            target,
            embeddings,
            config.input_budget,
            counter,
            TopologyConfig(typed=config.typed_topology, propagate=config.propagation),
        )
    else:
        result = relevance_select(graph, target, config.input_budget, counter)
    if config.paths:
        result = preserve_paths(graph, result, counter)
    if config.repair:
        result = repair_support(graph, result, counter)
    return result


class GraphBackend(Backend):
    """Reuse the durable journal and strict accounting with a distinct identity."""

    def configure(self, graph, selection, config, counter=None):
        if self.opts["admission"] != "strict":
            raise CompatibilityError("GRAFT requires strict admission")
        if graph.revision != self.instance.revision or selection.graph_hash != graph.hash:
            raise CompatibilityError("Graph, selection and buggy revision differ")
        if selection.target not in {e.entity_id for e in self.entities}:
            raise CompatibilityError("Graph target outside candidate pool")
        for e in self.entities:
            source = graph.sources.get(e.entity_id)
            if source is None or source["source"] != e.source or source["path"] != e.path:
                raise CompatibilityError("Graph source differs from supplied candidate")
        failures = [n for n in graph.nodes if n.kind == "failure"]
        if len(failures) != 1 or failures[0].text != self.instance.evidence.text:
            raise CompatibilityError("Graph failure evidence differs")
        checked = package(
            graph,
            selection.target,
            selection.ids,
            selection.budget,
            counter or ByteCounter(),
            selection.selector,
            selection.scores,
        )
        if (
            checked.status != "ready"
            or checked.text != selection.text
            or checked.input_units != selection.input_units
        ):
            raise CompatibilityError("Infeasible, altered or differently counted evidence package")
        self.graph, self.selection, self.graft_config = graph, selection, config

    def payload(self, cs, action, level):
        if len(cs) != 1 or cs[0].function_id != self.selection.target:
            raise CompatibilityError("Graph inspection must target its selected candidate")
        value = super().payload(cs, action, level)
        value["messages"] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.selection.text},
        ]
        return value

    def request_identity(self, cs, action, level, payload):
        identity = super().request_identity(cs, action, level, payload)
        return {
            **identity,
            "prompt_version": "graft-graph-verdict-v2-short-strings",
            "evidence_stage": EvidenceStage.GRAPH.value,
            "graph_hash": self.graph.hash,
            "selection_hash": self.selection.hash,
            "graft_config": asdict(self.graft_config),
        }

    def execute(self, cs, action=Action.FULL_VERIFY, level=2):
        # FULL_VERIFY here selects the existing single-verdict wire parser only.
        result = super().execute(cs, Action.FULL_VERIFY, 2)
        if result.observations[0].valid:
            value = json.loads(result.raw_response)
            schema_error = ('Graph response omitted required uncertainty' if 'uncertainty' not in value else
                            'Graph explanation/missing_evidence must be strings' if
                            not isinstance(value.get('explanation'), str) or
                            not isinstance(value.get('missing_evidence'), str) else None)
            if schema_error:
                # Preserve the single-verdict legacy parser but do not convert
                # absent graph uncertainty into a confident observed judgment.
                result.observations[0] = replace(
                    result.observations[0],
                    valid=False,
                    uncertainty=1.0,
                    reason=schema_error,
                )
                result.error = schema_error
                self.events[-1]["error"] = result.error
            else:
                result.observations[0] = replace(
                    result.observations[0], missing_evidence=str(value.get("missing_evidence", ""))
                )
        return result


class Session:
    """One policy's logical spend across both legacy and graph transports."""

    def __init__(
        self, instance, entities, provider, store, opts, budget, graph, config, counter=None
    ):
        if opts["admission"] != "strict":
            raise CompatibilityError("GRAFT requires strict admission")
        self.legacy = Backend(instance, entities, provider, store, opts, budget)
        self.graph_backend = GraphBackend(instance, entities, provider, store, opts, budget)
        self.graph, self.config, self.counter = graph, config, counter
        self.spent, self.events = 0, []
        self.budget = budget

    def backend(self, stage, selection=None):
        backend = self.graph_backend if stage == EvidenceStage.GRAPH else self.legacy
        if stage == EvidenceStage.GRAPH:
            backend.configure(self.graph, selection, self.config, self.counter)
        backend.spent = self.spent
        return backend

    def planning(self, candidates, stage, selection=None):
        backend = self.backend(stage, selection)
        return backend.planning(
            candidates,
            Action.BATCH_SCREEN if stage == EvidenceStage.SKETCH else Action.FULL_VERIFY,
            1 if stage == EvidenceStage.SKETCH else 2,
        )

    def execute(self, candidates, stage, selection=None):
        backend = self.backend(stage, selection)
        before = len(backend.events)
        try:
            return backend.execute(
                candidates,
                Action.BATCH_SCREEN if stage == EvidenceStage.SKETCH else Action.FULL_VERIFY,
                1 if stage == EvidenceStage.SKETCH else 2,
            )
        finally:
            self.spent = backend.spent
            self.events.extend(
                {
                    **event,
                    "stage": stage.value,
                    "candidate_ids": [c.function_id for c in candidates],
                    "selection_hash": selection.hash if selection else None,
                }
                for event in backend.events[before:]
            )


def failing_test_name_anchors(failure_text, function_ids):
    """Exact name cues from observed failing TEST headers, never gold labels.

    Full method names only; this is not proof of execution or calibrated risk.
    """
    tests = set()
    for line in failure_text.splitlines():
        if line.startswith("TEST:"):
            name = line[5:].strip().split("(", 1)[0]
            parts = re.split(r"[^a-zA-Z0-9]+", name)
            tests.update(part.lower() for part in parts if part)
            # Also handle common Java testGetOffsetFromLocal naming.
            tests.update(
                part[4:].lower()
                for part in parts
                if part.startswith("test") and len(part) > 8
            )
    result = {}
    for function_id in function_ids:
        method = function_id.rsplit(".", 1)[-1].split("(", 1)[0]
        name = re.sub(r"[^a-z0-9]", "", method.lower())
        result[function_id] = len(name) >= 8 and name in tests
    return result


class Router:
    """Transparent development heuristic; no calibrated gain or trained policy."""

    def __init__(self, candidates, session, fusion, policy="graft_adaptive", embeddings=None):
        if policy not in {"l1_anchor", "l1_graph", "l1_graph_suspicion", "l1_graph_focused", "graft_adaptive"}:
            raise ValueError("Unknown GRAFT policy")
        if len({c.function_id for c in candidates}) != len(candidates) or not candidates:
            raise ValueError("Nonempty unique candidate pool required")
        self.candidates, self.session, self.fusion, self.policy = (
            candidates,
            session,
            fusion,
            policy,
        )
        self.config, self.graph = session.config, session.graph
        self.embeddings = embeddings
        self.states = {c.function_id: CandidateState() for c in candidates}
        self.history = {c.function_id: [] for c in candidates}
        self.packages, self.trace, self.calls = {}, [], []
        failures = [n.text for n in self.graph.nodes if n.kind == "failure"]
        self.failure_anchors = failing_test_name_anchors(
            "\n".join(failures), [c.function_id for c in candidates]
        )

    def scores(self):
        scores = self.fusion.scores(self.candidates, self.states)
        if self.config.failure_name_anchor and self.fusion.name == "rrf":
            # One quarter of an RRF rank contribution, a transparent heuristic.
            # No gold label, patch or future outcome is used.
            bonus = 0.25 / (self.fusion.rrf_k + 1)
            scores = {
                ident: value + (bonus if self.failure_anchors[ident] else 0.0)
                for ident, value in scores.items()
            }
        return scores

    def order(self):
        return self.fusion.order(self.candidates, self.scores())


    def observe(self, candidates, stage, selection=None):
        result = self.session.execute(candidates, stage, selection)
        self.calls.append(
            {
                "stage": stage.value,
                "candidate_ids": [c.function_id for c in candidates],
                "selection": selection.to_dict() if selection else None,
                **asdict(result),
            }
        )
        if {o.function_id for o in result.observations} != {
            c.function_id for c in candidates
        } or len(result.observations) != len(candidates):
            raise CompatibilityError("Paid backend observation IDs differ from selected targets")
        for obs in result.observations:
            previous = self.states[obs.function_id].observation
            # A graph response that admits missing evidence is not enough to
            # override stronger, valid L1 evidence with a lower suspicion.
            # Preserve the paid response in history for audit and ablation.
            guard = (
                self.config.guard_unsupported_graph_downgrade
                and stage == EvidenceStage.GRAPH
                and previous is not None and previous.valid
                and obs.valid and obs.suspicion < previous.suspicion
                and obs.missing_evidence.strip().lower() not in {"", "none", "n/a"}
            )
            event = {
                "stage": stage.value,
                "evidence_ids": list(selection.ids) if selection else [],
                **asdict(obs),
                "applied_to_ranking": not guard,
                "guard_reason": "unsupported_graph_downgrade" if guard else None,
            }
            self.history[obs.function_id].append(event)
            # Historical Full bonus applies only to actual complete methods.
            self.states[obs.function_id] = CandidateState(
                2 if stage == EvidenceStage.METHOD_FULL else 1,
                previous if guard else obs,
                list(self.history[obs.function_id]),
            )

    def admitted(self, candidates, stage, selection=None):
        predicted, reserved = self.session.planning(candidates, stage, selection)
        return (
            predicted,
            reserved,
            reserved <= self.session.budget - self.session.spent
            and predicted <= self.session.legacy.opts["context_window"],
        )

    def execute(self):
        stop = "action_caps_or_no_eligible_action"
        batch = self.session.legacy.opts["batch_size"]
        try:
            for start in range(0, len(self.candidates), batch):
                cs = self.candidates[start : start + batch]
                predicted, reserved, allowed = self.admitted(cs, EvidenceStage.SKETCH)
                self.trace.append(
                    {
                        "phase": "initial_l1",
                        "candidate_ids": [c.function_id for c in cs],
                        "predicted_tokens": predicted,
                        "reserved_tokens": reserved,
                        "selected": allowed,
                    }
                )
                if not allowed:
                    stop = "incomplete_l1_budget_or_context"
                    return self.result(stop)
                self.observe(cs, EvidenceStage.SKETCH)
            if self.policy == "l1_anchor":
                # Pure L1 plus the configured name anchor. Graph construction
                # supplies the same initial evidence but NO graph or Full call
                # is allowed, ensuring a clean ranking-only ablation.
                return self.result("l1_anchor_only")
            for step in range(self.config.max_graph + self.config.max_full + 1):
                scores = self.scores()
                ordered = self.order()
                differences = [
                    abs(scores[a.function_id] - scores[b.function_id])
                    for a, b in zip(ordered, ordered[1:])
                ]
                scale = max(differences, default=1.0) or 1.0
                counts = {
                    s.value: sum(call["stage"] == s.value for call in self.calls)
                    for s in EvidenceStage
                }
                rows = []
                for position, c in enumerate(ordered):
                    ident = c.function_id
                    past = {v["stage"] for v in self.history[ident]}
                    obs = self.states[ident].observation
                    adjacent = differences[max(0, position - 1) : position + 1]
                    closeness = 1 / (1 + min(adjacent) / scale) if adjacent else 0.0
                    uncertain = (
                        not obs.valid
                        or obs.uncertainty >= self.config.uncertainty_threshold
                        or bool(obs.missing_evidence)
                    )
                    stages = []
                    if (
                        EvidenceStage.GRAPH.value not in past
                        and EvidenceStage.METHOD_FULL.value not in past
                        and counts["graph"] < self.config.max_graph
                    ):
                        stages.append(EvidenceStage.GRAPH)
                    if (
                        self.policy == "graft_adaptive"
                        and EvidenceStage.METHOD_FULL.value not in past
                        and counts["method_full"] < self.config.max_full
                    ):
                        # Prefer a graph inspection first when it is feasible. Full
                        # remains available for invalid/uncertain graph judgments.
                        if EvidenceStage.GRAPH.value in past or self.config.max_graph == 0:
                            stages.append(EvidenceStage.METHOD_FULL)
                    for stage in stages:
                        selection = None
                        if stage == EvidenceStage.GRAPH:
                            if ident not in self.packages:
                                if self.policy == "l1_graph_focused":
                                    self.packages[ident] = focus_select(
                                        self.graph, ident, self.config.input_budget,
                                        self.session.counter,
                                        missing_evidence=obs.missing_evidence if obs else "",
                                    )
                                else:
                                    self.packages[ident] = select_evidence(
                                        self.graph,
                                        ident,
                                        self.config,
                                        self.session.counter,
                                        self.embeddings,
                                    )
                            selection = self.packages[ident]
                            if selection.status != "ready":
                                rows.append(
                                    {
                                        "candidate": ident,
                                        "stage": stage.value,
                                        "eligible": False,
                                        "reason": selection.reason,
                                        "package_status": selection.status,
                                    }
                                )
                                if (
                                    self.policy == "graft_adaptive"
                                    and counts["method_full"] < self.config.max_full
                                ):
                                    stages.append(EvidenceStage.METHOD_FULL)
                                continue
                        predicted, reserved, admitted = self.admitted([c], stage, selection)
                        seen = {i for h in self.history[ident] for i in h["evidence_ids"]}
                        evidence = (
                            set(selection.ids)
                            if selection
                            else {n.id for n in self.graph.nodes if n.owner == ident}
                        )
                        novelty = len(evidence - seen) / max(1, len(evidence))
                        value = (
                            0.35 * obs.suspicion
                            + 0.30 * obs.uncertainty
                            + 0.20 * closeness
                            + 0.15 * novelty
                            + (0.40 if self.config.failure_name_anchor and self.failure_anchors[ident] else 0.0)
                        )
                        priority = (
                            value
                            * (
                                0.5
                                + 0.5
                                * (self.session.budget - self.session.spent)
                                / max(1, self.session.budget)
                            )
                            / max(1, predicted)
                        )
                        gate = (
                            self.policy in {"l1_graph", "l1_graph_suspicion", "l1_graph_focused"} or position < self.config.top_k or uncertain
                        )
                        rows.append(
                            {
                                "candidate": ident,
                                "stage": stage.value,
                                "eligible": admitted and gate,
                                "reason": "eligible"
                                if admitted and gate
                                else "budget_or_context"
                                if not admitted
                                else "outside_cheap_gate",
                                "predicted_tokens": predicted,
                                "reserved_tokens": reserved,
                                "current_score": scores[ident],
                                "failure_name_anchor": self.failure_anchors[ident],
                                "current_rank": position + 1,
                                "reported_uncertainty": obs.uncertainty,
                                "suspicion": obs.suspicion,
                                "adjacent_closeness": closeness,
                                "novelty": novelty,
                                "missing_support": len(missing_support(self.graph, selection.ids))
                                if selection
                                else None,
                                "priority": priority,
                                "calibrated": False,
                            }
                        )
                eligible = [row for row in rows if row["eligible"]]
                selected = (
                    (
                        max(
                            eligible,
                            key=lambda r: (
                                r["priority"],
                                -r["current_rank"],
                                r["candidate"],
                                r["stage"],
                            ),
                        )
                        if self.policy == "graft_adaptive"
                        else max(
                            eligible,
                            key=lambda r: (
                                # Deliberately simple, gold-free graph-selection
                                # ablation; retain the old l1_graph as control.
                                r["suspicion"],
                                -r["reported_uncertainty"],
                                -r["predicted_tokens"],
                                -r["current_rank"],
                                r["candidate"],
                            ),
                        )
                        if self.policy in {"l1_graph_suspicion", "l1_graph_focused"}
                        else min(
                            eligible,
                            key=lambda r: (
                                next(
                                    c.rank
                                    for c in self.candidates
                                    if c.function_id == r["candidate"]
                                ),
                                r["candidate"],
                            ),
                        )
                    )
                    if eligible
                    else None
                )
                self.trace.append(
                    {
                        "phase": "route",
                        "step": step,
                        "spent_before": self.session.spent,
                        "remaining_before": self.session.budget - self.session.spent,
                        "stage_counts": counts,
                        "options": rows,
                        "selected": selected,
                    }
                )
                if selected is None:
                    break
                c = next(c for c in self.candidates if c.function_id == selected["candidate"])
                stage = EvidenceStage(selected["stage"])
                self.observe(
                    [c],
                    stage,
                    self.packages[c.function_id] if stage == EvidenceStage.GRAPH else None,
                )
        except ExecutionLimit as exc:
            stop = "execution_limit: " + str(exc)
            self.trace.append({"phase": "stop", "reason": stop, "paid_tokens": self.session.spent})
        return self.result(stop)

    def result(self, stop):
        return {
            "schema": "graft-router-v1",
            "policy": self.policy,
            "graft_config": asdict(self.config),
            "ranker": self.fusion.config(),
            "ranked_functions": [c.function_id for c in self.order()],
            "ranking_scores": self.scores(),
            "failure_name_anchors": self.failure_anchors,
            "failure_anchor_is_calibrated": False,
            "actual_tokens": self.session.spent,
            "tokens_remaining": self.session.budget - self.session.spent,
            "budget_exceeded": self.session.spent > self.session.budget,
            "strict_admission": True,
            "calls": self.calls,
            "accounting_events": self.session.events,
            "states": {i: asdict(s) for i, s in self.states.items()},
            "stage_histories": self.history,
            "trace": self.trace,
            "stop_reason": stop,
            "utility_is_calibrated": False,
            "quality_is_validated": False,
            "graph_is_method_full_superset": False,
        }

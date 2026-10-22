"""Strict, zero-inference replay of independently prompted measured actions.

Only execute() reveals a selected stored answer. estimate() reconstructs the
original preflight estimate and never reads future response token counts.
Replay is a conditional counterfactual, not a fresh model experiment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path

from rave_fl.adaptive import fused_score
from rave_fl.core import Action
from rave_fl.pilot import prepare_case, write_json
from rave_fl.pilot_backend import OllamaPilotBackend, build_prompt, parse_response
from rave_fl.pilot_evaluation import evaluate
from rave_fl.pilot_types import CallResult, CandidateState, Observation, PolicyConfig
from rave_fl.rank_repair import RankRepairPilot, RepairConfig


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ReplayCorpus:
    def __init__(self, run_path: Path, manifest_case: dict):
        run = json.loads(run_path.read_text())
        require(run.get("status") == "complete" and not run.get("dry_run"), "Require a completed measured run")
        matches = [c for c in run["cases"] if c["bug"] == manifest_case["bug"]]
        require(len(matches) == 1, "Missing or ambiguous replay bug")
        case = matches[0]
        self.candidates, self.context, provenance = prepare_case(manifest_case, case["top_per_method"])
        self.by_id = {c.function_id: c for c in self.candidates}
        require(provenance["sources_sha256"] == case["sources_sha256"], "Replay source hash mismatch")
        require(provenance["evidence"]["sha256"] == case["evidence"]["sha256"], "Replay evidence hash mismatch")
        self.references = {name: deepcopy(case["arms"][name]) for name in ("full", "fixed_batch")}
        require(self.references["full"]["full_verified_count"] == len(self.candidates)
                and not self.references["full"]["failed_full_ids"], "Replay needs a complete valid Full reference")
        require(self.references["fixed_batch"]["screened_count"] == len(self.candidates), "Replay needs complete L1 batches")
        self._entries = {}
        digest_values, settings, ledger_hashes = set(), None, {}
        self.groups = []
        for name in ("full", "fixed_batch"):
            arm = self.references[name]
            digest_values.add(arm["model_metadata"]["digest"])
            path = run_path.parent / case["bug"] / (name + "_calls.jsonl")
            ledger_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require(all(e["event"] in {"request", "response"} for e in events), "Replay ledger contains a failed/unknown event")
            requests = [e for e in events if e["event"] == "request"]
            responses = [e for e in events if e["event"] == "response"]
            require(len(requests) == len(responses) == arm["call_count"], "Incomplete replay request/response pairs")
            seen = set()
            total = 0
            for req, response, call in zip(requests, responses, arm["calls"]):
                require(req["call_id"] == response["call_id"], "Replay call ID mismatch")
                payload = req["request"]
                require(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == req["request_sha256"],
                        "Replay request fingerprint mismatch")
                require(payload["model"] == run["model"] and payload["stream"] is False
                        and payload["think"] is False and payload["format"] == "json", "Unsupported replay request")
                opts = payload["options"]
                ids = req["candidate_ids"]
                require(len(set(ids)) == len(ids) and set(ids) <= set(self.by_id), "Invalid replay candidate IDs")
                require(not (set(ids) & seen), "Candidate appears twice in the same replay reference")
                seen.update(ids)
                candidate_settings = (payload["model"], opts["temperature"], opts["seed"], opts["num_ctx"], opts["num_predict"] // len(ids))
                require(opts["num_predict"] % len(ids) == 0, "Nonuniform output caps")
                if settings is None:
                    settings = candidate_settings
                require(settings == candidate_settings, "Replay references use different generation settings")
                action, level = Action(req["action"]), req["level"]
                require((action, level) == ((Action.FULL_VERIFY, 2) if name == "full" else (Action.BATCH_SCREEN, 1)),
                        "Unexpected replay action or evidence level")
                selected = [self.by_id[i] for i in ids]
                system, user = build_prompt(selected, self.context, action, level)
                require(payload["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "Prompt changed or contains action history; cached replay is invalid")
                provider = response["response"]
                require(provider.get("done") is True, "Incomplete provider response")
                observations, error = parse_response(provider["message"]["content"], selected, action)
                require(error is None, "Cannot build a complete reference from invalid cached responses")
                result = CallResult(observations, provider["prompt_eval_count"], provider["eval_count"],
                                    call["duration_seconds"], provider["message"]["content"], error)
                require(call["observations"] == asdict(result)["observations"] and call["raw_response"] == result.raw_response,
                        "Stored parsed response disagrees with raw provider output")
                require(call["input_tokens"] == result.input_tokens and call["output_tokens"] == result.output_tokens,
                        "Cached usage disagrees with provider counters")
                require(call["candidate_ids"] == ids and call["level"] == level and call["action"] == action.value,
                        "Stored call differs from replay ledger")
                key = (action.value, level, tuple(ids))
                self._entries[key] = result
                total += result.total_tokens
                if name == "fixed_batch":
                    self.groups.append(tuple(ids))
            require(seen == set(self.by_id) and total == arm["actual_tokens"], "Incomplete cached coverage or usage")
        require(len(digest_values) == 1, "Model weights differ across replay references")
        self.settings = settings
        self.provenance = {**provenance, "source_run": str(run_path), "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                           "ledger_sha256": ledger_hashes, "model": settings[0], "model_digest": next(iter(digest_values)),
                           "seed": settings[2], "context_window": settings[3], "output_per_candidate": settings[4],
                           "prompt_independence_checked": True}

    def backend(self):
        return ReplayBackend(self)

    def observed_full(self, ident):
        """Offline evaluator/oracle only. Policy implementations never call this."""
        return deepcopy(self._entries[(Action.FULL_VERIFY.value, 2, (ident,))])


class ReplayBackend:
    def __init__(self, corpus):
        self._corpus = corpus
        model, temperature, seed, window, cap = corpus.settings
        require(temperature == 0, "Replay expects the frozen temperature-zero protocol")
        self._planner = OllamaPilotBackend(model, corpus.context, output_per_candidate=cap,
                                           seed=seed, context_window=window)
        self.context_window = window
        self.replayed_calls = 0

    def estimate(self, candidates, action, level):
        return self._planner.estimate(candidates, action, level)

    def execute(self, candidates, action, level):
        key = (action.value, level, tuple(c.function_id for c in candidates))
        require(key in self._corpus._entries, "No exact cached action; refuse to invent an observation or call the network")
        self.replayed_calls += 1
        return deepcopy(self._corpus._entries[key])


def oracle_frontier(corpus, truths, max_full=4, budget=16000):
    """Gold/cached-cost diagnostic bound. This is never a deployed policy."""
    from wp1.evaluation import method_matches
    fixed = corpus.references["fixed_batch"]
    relevant = {c.function_id for c in corpus.candidates if any(method_matches(c.function_id, t) for t in truths)}
    original = {c.function_id: c.rank for c in corpus.candidates}
    scores = fixed["ranking_scores"]
    full_scores = {i: corpus.observed_full(i).observations[0].suspicion for i in scores}
    full_costs = {i: corpus.observed_full(i).total_tokens for i in scores}
    def quality(selected):
        s = {**scores, **{i: full_scores[i] for i in selected}}
        order = sorted(s, key=lambda i: (-s[i], original[i]))
        rank = next((n for n, i in enumerate(order, 1) if i in relevant), None)
        return (1 / rank if rank else 0), rank
    base, _ = quality(())
    singleton_gain = {i: quality((i,))[0] - base for i in scores}
    rows, interactions = [], []
    for size in range(max_full + 1):
        best = None
        feasible = 0
        for combo in itertools.combinations(scores, size):
            cost = fixed["actual_tokens"] + sum(full_costs[i] for i in combo)
            if cost > budget:
                continue
            feasible += 1
            mrr, rank = quality(combo)
            candidate = {"full_count": size, "selected": list(combo), "accounted_tokens": cost,
                         "MRR": mrr, "rank": rank, "top5": bool(rank and rank <= 5)}
            if best is None or (mrr, -cost) > (best["MRR"], -best["accounted_tokens"]):
                best = candidate
            if size == 2:
                interactions.append({"selected": list(combo),
                                     "interaction_MRR": (mrr - base) - sum(singleton_gain[i] for i in combo)})
        rows.append({"exact_full_count": size, "feasible_subsets": feasible, "best": best})
    strongest = max(interactions, key=lambda r: abs(r["interaction_MRR"])) if interactions else None
    return {"uses_gold_and_future_observed_costs": True, "deployable_policy": False,
            "budget": budget, "frontier": rows, "strongest_pair_interaction": strongest}


def replay_experiment(run_path, manifest_path, protocol_path, output):
    require(not output.exists(), "Replay output must be a new directory")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest["cases"]) == 1, "Run one independently specified replay case at a time")
    protocol = json.loads(protocol_path.read_text())
    corpus = ReplayCorpus(run_path, manifest["cases"][0])
    require(protocol["batch_size"] == len(corpus.groups[0]), "Replay batch size changed")
    results = []
    for budget in protocol["budgets"]:
        for limit in protocol["max_full_values"]:
            for mode in protocol["policies"]:
                cfg = PolicyConfig(token_budget=budget, batch_size=protocol["batch_size"], mode=mode)
                arm = RankRepairPilot(corpus.candidates, corpus.backend(), cfg,
                                      RepairConfig(limit, 5)).execute()
                arm.update(execution_kind="offline_exact_action_replay", new_llm_calls=0,
                           accounted_replayed_tokens=arm["actual_tokens"], fresh_inference_tokens=0)
                results.append(arm)
    # No gold is read until every policy decision has been made.
    from wp1.flexfl_data import read_ground_truth
    case = manifest["cases"][0]
    truths = read_ground_truth(Path(case["flexfl_repo"]), case["bug"])
    for arm in results:
        arm["accuracy"] = evaluate(arm["ranked_functions"], truths, arm["states"])
    oracle = oracle_frontier(corpus, truths, max(protocol["max_full_values"]), protocol["primary_budget"])
    report = {"schema": 1, "execution_kind": "offline_exact_action_replay", "new_llm_calls": 0,
              "fresh_inference_tokens": 0, "protocol": protocol, "provenance": corpus.provenance,
              "ranker": "shared_prior_evidence_fusion_v1", "references": corpus.references,
              "policy_runs": results, "diagnostic_oracle": oracle,
              "limitations": ["Previously inspected development bug; not a held-out result.",
                              "Cached responses and their token totals are conditional counterfactuals, not new inference.",
                              "Untrained utility is a ranking proxy; no calibrated quality or novelty claim.",
                              "Independent candidate prompts and exact original batch composition are required."]}
    output.mkdir(parents=True)
    write_json(output / "replay.json", report)
    lines = ["# L1-first rank-repair replay", "", "No new model calls. Recorded-action counterfactuals on a known development bug.", "",
             "| Budget | Max Full | Policy | Full calls | Replayed tokens | Gold rank | MRR | Stop |",
             "|---:|---:|---|---:|---:|---:|---:|---|"]
    for a in results:
        lines.append(f"| {a['config']['token_budget']} | {a['repair_config']['max_full']} | {a['mode']} | {a['full_verified_count']} | {a['actual_tokens']} | {a['accuracy']['first_relevant_rank']} | {a['accuracy']['MRR']:.3f} | {a['stop_reason']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    primary = [a for a in results if a["config"]["token_budget"] == protocol["primary_budget"]
               and a["repair_config"]["max_full"] == protocol["primary_max_full"]]
    from rave_fl.rank_repair_labels import action_value_rows
    labels = action_value_rows(corpus, truths, primary, case["bug"], protocol["primary_budget"])
    (output / "action_values.jsonl").write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in labels))
    write_json(output / "action_value_profile.json", {
        "rows": len(labels), "independent_bugs": 1,
        "unique_states": len({r["state_id"] for r in labels}),
        "positive_MRR_gain": sum(r["targets"]["delta_MRR"] > 0 for r in labels),
        "negative_MRR_gain": sum(r["targets"]["delta_MRR"] < 0 for r in labels),
        "training_claim": "none; all rows belong to one known development bug"})
    return {"output": str(output), "new_llm_calls": 0,
            "primary": [{"policy": a["mode"], "tokens": a["actual_tokens"],
                         "rank": a["accuracy"]["first_relevant_rank"], "full_calls": a["full_verified_count"],
                         "selected": [c["candidate_ids"][0] for c in a["calls"] if c["level"] == 2]} for a in primary]}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("configs/rave_rank_repair_protocol.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(replay_experiment(a.run, a.manifest, a.protocol, a.output), indent=2))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()

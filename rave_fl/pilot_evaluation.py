"""Offline-only evaluation. Never imported by the allocation controller."""
from __future__ import annotations

import random
import statistics
from wp1.evaluation import first_relevant_rank, method_matches


def evaluate(ranking, truths, states):
    if not truths:
        raise ValueError("Missing ground truth: refuse to report fake zero accuracy")
    rank = first_relevant_rank(ranking, truths)
    def recall(ids):
        return sum(any(method_matches(i, t) for i in ids) for t in truths) / len(truths)
    screened = [i for i, s in states.items() if s["level"] >= 0]
    verified = [i for i, s in states.items() if s["level"] == 2 and s["observation"]["valid"]]
    retained = [i for i, s in states.items() if s["observation"] and s["observation"]["valid"]
                and s["observation"]["suspicion"] >= .5]
    return {"top1": bool(rank and rank <= 1), "top3": bool(rank and rank <= 3),
            "top5": bool(rank and rank <= 5), "first_relevant_rank": rank,
            "MRR": 1 / rank if rank else 0,
            "candidate_pool_recall": recall(ranking), "screened_gold_recall": recall(screened),
            "full_verified_gold_recall": recall(verified), "positive_evidence_gold_recall": recall(retained)}


def risk_rows(bug, run, truths):
    full = run["arms"].get("full")
    if full is None or full["full_verified_count"] != len(full["states"]) or full["failed_full_ids"]:
        raise ValueError("Risk labels require a complete valid full baseline")
    rows = []
    original_order = {ident: i for i, ident in enumerate(full["states"])}
    for name, level, reusable in (("probe_l0", 0, "confidence"), ("probe_l1", 1, "fixed_batch")):
        arm_name = name if name in run["arms"] else reusable
        arm = run["arms"].get(arm_name)
        if arm is None:
            raise ValueError("Risk labels require complete L0 discovery and L1 fixed-batch probes")
        snapshots = [s for s in arm["risk_snapshots"] if s["level"] == level]
        ids = [s["function_id"] for s in snapshots]
        if len(ids) != len(full["states"]) or set(ids) != set(full["states"]):
            raise ValueError("Risk probes must cover every candidate exactly once at each cheap level")
        from types import SimpleNamespace
        from rave_fl.adaptive import fused_score
        from rave_fl.pilot_types import CandidateState, Observation
        cheap_scores = {s["function_id"]: fused_score(SimpleNamespace(prior=s["features"]["prior"]),
                        CandidateState(s["level"], Observation(**s["observation"]))) for s in snapshots}
        cheap_ranking = sorted(cheap_scores, key=lambda i: (-cheap_scores[i], original_order[i]))
        cheap_rank = first_relevant_rank(cheap_ranking, truths)
        for snapshot in snapshots:
            ident = snapshot["function_id"]
            full_obs = full["states"][ident]["observation"]
            cheap = snapshot["observation"]
            is_gold = any(method_matches(ident, t) for t in truths)
            full_positive = full_obs["suspicion"] >= .5
            cheap_negative = not cheap["valid"] or cheap["suspicion"] < .5
            # Marginal action outcome: retain other candidates' cheap evidence,
            # replace just this candidate with its measured full observation.
            # This labels whether continuing cheaply loses MRR versus verifying
            # this candidate now. Gold/full values never enter features.
            scores = dict(cheap_scores)
            scores[ident] = full["ranking_scores"][ident]
            hybrid_ranking = sorted(scores, key=lambda i: (-scores[i], original_order[i]))
            hybrid_rank = first_relevant_rank(hybrid_ranking, truths)
            ranking_harm = bool(hybrid_rank and (cheap_rank is None or cheap_rank > hybrid_rank))
            rows.append({"bug": bug, "function_id": ident, "level": snapshot["level"],
                         "features": snapshot["features"],
                         "label": int(is_gold and full_positive and cheap_negative),
                         "ranking_harm_label": int(ranking_harm),
                         "ranking_label_definition": "counterfactual_cheap_context_MRR_loss_relative_to_full_action",
                         "cheap_first_relevant_rank": cheap_rank,
                         "counterfactual_first_relevant_rank": hybrid_rank,
                         "full_disagreement": int(cheap_negative and full_positive),
                         "label_definition": "gold_false_negative_introduced_relative_to_full",
                         "inference_fingerprint": run["inference_fingerprint"],
                         "source_arm": arm_name,
                         "source": "real_ollama_paired_probes"})
    return rows


def interval(values, seed=0, samples=2000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [boot[int(.025 * samples)], boot[min(samples - 1, int(.975 * samples))]]


def summarize(runs):
    names = sorted({n for r in runs for n in r["arms"] if not n.startswith("probe_")})
    summary = {}
    for name in names:
        paired = [(r["arms"]["full"], r["arms"][name]) for r in runs
                  if "full" in r["arms"] and name in r["arms"]
                  and "accuracy" in r["arms"]["full"] and "accuracy" in r["arms"][name]]
        if not paired:
            continue
        deltas = {k: [float(a["accuracy"][k]) - float(f["accuracy"][k]) for f, a in paired]
                  for k in ("top1", "top5", "MRR")}
        baseline_tokens = sum(f["actual_tokens"] for f, a in paired)
        arm_tokens = sum(a["actual_tokens"] for f, a in paired)
        full_seconds = sum(f["llm_duration_seconds"] for f, a in paired)
        arm_seconds = sum(a["llm_duration_seconds"] for f, a in paired)
        complete = all(f["full_verified_count"] == len(f["states"]) and not f["failed_full_ids"] for f, a in paired)
        summary[name] = {
            "n_bugs": len(paired), "baseline_complete": complete,
            "actual_tokens": arm_tokens, "full_actual_tokens": baseline_tokens,
            "llm_seconds": arm_seconds, "full_llm_seconds": full_seconds,
            "llm_time_savings_fraction": 1 - arm_seconds / full_seconds if full_seconds else None,
            "estimated_cost_usd_at_supplied_rates": (
                sum(a["hypothetical_cost_usd_at_supplied_rates"] for f, a in paired)
                if all("hypothetical_cost_usd_at_supplied_rates" in a for f, a in paired) else None),
            "token_savings_fraction": 1 - arm_tokens / baseline_tokens if baseline_tokens else None,
            "mean_accuracy": {k: statistics.mean(float(a["accuracy"][k]) for f, a in paired)
                              for k in ("top1", "top3", "top5", "MRR", "screened_gold_recall", "full_verified_gold_recall")},
            "paired_delta": {k: {"mean": statistics.mean(v), "bootstrap_95_ci": interval(v)}
                             for k, v in deltas.items()},
            "interpretation": "exploratory; a small pilot cannot establish noninferiority or novelty",
        }
    return summary

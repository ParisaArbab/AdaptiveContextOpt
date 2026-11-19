"""Gold-blind L1 preservation vs Code Cards, using saved live Qwen observations.

For each candidate, select L1 if it is among the top k *prior-ranked methods*,
and Code Card otherwise. This choice uses neither gold, LLM scores, nor Full
responses. This exploratory replay DOES NOT train or claim performance improvement.
No inference occurs. The original 24-candidate pool is unchanged.
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_method_agent"))
import research_method_agent as agent

COHORT = ROOT / "results" / "egav_validation_10bugs"
VALUES = (5, 8, 12)
ARMS = ("l1", "code_cards", "card_egav")


def replay(bug, observations, k, gold_files):
    """Select evidence by pre-existing BM25 prior only, before reading any reply."""
    candidates = agent.sorted_candidates(bug)
    kinds = {c["id"]: ("l1" if i < k else "card") for i, c in enumerate(candidates)}
    state, tokens, actions = {}, 0, []
    for c in candidates:
        kind = kinds[c["id"]]
        o = observations.get(agent.observation_key(bug["id"], c["id"], kind))
        if o is None or o.get("status") != "ok":
            return {"status": "missing_real_observation", "missing": [c["id"], kind]}
        score = 0.30 * float(c["prior"]) + 0.70 * float(o["score"])
        cost = int(o["input_tokens"]) + int(o["output_tokens"])
        state[c["id"]] = score
        tokens += cost
        actions.append({"candidate": c["id"], "kind": kind, "tokens": cost})
    order = sorted(candidates, key=lambda c: (-state[c["id"]], c["id"]))
    files = list(dict.fromkeys(c["file"] for c in order))
    result = agent.metric(files, set(gold_files))
    return {"status": "complete", "tokens": tokens, "llm_calls": len(actions),
            "k": k, "actions": actions, **result}


def main():
    rows = []
    for batch in ("01", "02"):
        folder = COHORT / ("batch_" + batch)
        manifest = agent.load(folder / "manifest.json")
        observations = agent.load(folder / "observations.json")
        labels = agent.load(folder / "gold_offline.json")["bugs"]
        original = agent.load(folder / "evaluation/results.json")["results"]
        if observations.get("manifest_hash") != agent.sha(manifest):
            raise ValueError("Manifest/observation hash mismatch: " + batch)
        if observations.get("provenance") != "live_ollama":
            raise ValueError("Requires saved real Qwen observations")
        indexed = agent.obs_index(observations)
        by_bug_arm = {(x["bug"], x["arm"]): x for x in original}
        for bug in manifest["bugs"]:
            bid = bug["id"]
            gold_files = labels[bid].get("files", [])
            if not gold_files:
                raise ValueError("Missing file-level labels for " + bid)
            for arm in ARMS:
                original_row = by_bug_arm[(bid, arm)]
                if original_row.get("gold_level") != "file_proxy":
                    raise ValueError("Non-file-level result")
                rows.append({"bug": bid, "arm": arm, "status": original_row["status"],
                             "mrr": original_row["mrr"],
                             "gold_rank": original_row["gold_rank"],
                             "tokens": original_row["tokens"]})
            for k in VALUES:
                outcome = replay(bug, indexed, k, gold_files)
                rows.append({"bug": bid, "arm": "prior_l1_" + str(k) + "_cards_rest",
                             **{z: outcome.get(z) for z in
                                ("status", "mrr", "gold_rank", "tokens")}})
    bugs = {r["bug"] for r in rows}
    if len(bugs) != 10:
        raise ValueError("Expected 10 distinct development bugs")
    summaries = []
    for arm in ARMS + tuple("prior_l1_" + str(k) + "_cards_rest" for k in VALUES):
        items = [r for r in rows if r["arm"] == arm and r["status"] == "complete"]
        summary = {"arm": arm, "n": len(items)}
        if items:
            summary.update(mean_mrr=statistics.mean(r["mrr"] for r in items),
                           mean_tokens=statistics.mean(r["tokens"] for r in items))
        summaries.append(summary)
    COHORT.mkdir(parents=True, exist_ok=True)
    document = {"schema": "mixed-l1-card-development-audit-v1",
                "no_new_model_calls": True,
                "policy_selection": "Top K by original BM25 prior, not gold or reply",
                "study": "10 development bugs, not held-out validation",
                "summary": summaries,
                "rows": rows}
    (COHORT / "mixed_l1_cards_audit.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("\n10-BUG DEVELOPMENT DIAGNOSTIC: PRESERVE L1 ON PRIOR-TOP-K, CARDS ELSEWHERE")
    print(f"{'Policy':28} {'n':>3} {'MRR':>8} {'Tokens':>9}")
    for s in summaries:
        if "mean_mrr" in s:
            print(f"{s['arm']:28} {s['n']:>3} {s['mean_mrr']:>8.4f} {s['mean_tokens']:>9.0f}")
        else:
            print(f"{s['arm']:28} {s['n']:>3} INCOMPLETE")
    print("\nPer-bug ranks (None means the gold file was absent):")
    labels = ("l1", "code_cards") + tuple(
        "prior_l1_" + str(k) + "_cards_rest" for k in VALUES)
    for bug in sorted(bugs):
        rank_map = {r["arm"]: r["gold_rank"] for r in rows if r["bug"] == bug}
        vals = ["MISS" if rank_map[a] is None else str(rank_map[a]) for a in labels]
        print(bug.ljust(14), " ".join(f"{x:>5}" for x in vals))
    print("\nWARNING: Both card formatting and prompt label differ from L1.")
    print("The two conditions' scores may be uncalibrated; this is exploratory.")
    print("The original source retrieval pool remains 24 methods.")
    print("Saved:", COHORT / "mixed_l1_cards_audit.json")


if __name__ == "__main__":
    main()

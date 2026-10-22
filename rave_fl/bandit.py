"""Guarded offline contextual-bandit experiment for RAVE-FL.

This is NOT activated by the normal pilot. Real randomized action logs with
observed outcomes and known logging propensities are required. Deterministic
replay or counterfactual Full responses are not valid bandit log data.

Rows (JSONL): {"bug": "Time-...", "project": "Time",
 "features": {"prior": 0.5, ...all FEATURE_NAMES...},
 "action": "FULL_VERIFY", "reward": 0.1, "propensity": 0.25}
reward: *observed* quality improvement minus normalized token penalty, using a
declared versioned reward definition, never a guessed counterfactual outcome.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rave_fl.code_sketch import FEATURE_NAMES

ACTIONS = ("BATCH_SCREEN", "EXPAND_CONTEXT", "FULL_VERIFY", "STOP")


def _feature_vector(row):
    import numpy as np
    values = row.get("features")
    if not isinstance(values, dict) or set(values) != set(FEATURE_NAMES):
        raise ValueError("Bandit rows must contain the full versioned feature schema")
    x = [float(values[k]) for k in FEATURE_NAMES]
    if not all(math.isfinite(v) for v in x):
        raise ValueError("Invalid nonfinite bandit feature")
    return np.asarray([1.0, *x], dtype=float)


def validate_logs(rows, split, min_train_bugs=10, min_test_bugs=5):
    if set(split) != {"train", "calibration", "test"}:
        raise ValueError("Bandit split must have train/calibration/test")
    sets = {part: set(ids) for part, ids in split.items()}
    if any(not v for v in sets.values()) or any(sets[a] & sets[b]
         for a in sets for b in sets if a < b):
        raise ValueError("Overlapping or empty bug splits")
    if not rows:
        raise ValueError("No real logged actions")
    per = {part: [] for part in sets}
    project_partition = {}
    nontrivial_propensities = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Action log row must be an object")
        bug, project, action = row.get("bug"), row.get("project"), row.get("action")
        if not isinstance(bug, str) or not isinstance(project, str) or not project:
            raise ValueError("Every logged action needs a bug and project identifier")
        if action not in ACTIONS:
            raise ValueError("Unknown action in bandit log")
        prob, reward = row.get("propensity"), row.get("reward")
        if (isinstance(prob, bool) or not isinstance(prob, (int, float))
                or not math.isfinite(prob) or not 0 < prob <= 1):
            raise ValueError("Actual logging probability must be in (0,1]")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or not -1 <= reward <= 1):
            raise ValueError("Observed normalized reward must be in [-1,1]")
        if row.get("reward_source") != "observed_randomized_action":
            raise ValueError("Only real randomized observed rewards are supported")
        _feature_vector(row)
        part = next((k for k, ids in sets.items() if bug in ids), None)
        if part is None:
            raise ValueError("Logged bug is absent from the frozen split")
        if project in project_partition and project_partition[project] != part:
            raise ValueError("Repository leakage across bandit partitions")
        project_partition[project] = part
        per[part].append(row)
        nontrivial_propensities += prob < 1
    if len({r["bug"] for r in per["train"]}) < min_train_bugs:
        raise ValueError("Insufficient independent TRAIN bugs for a bandit")
    if len({r["bug"] for r in per["test"]}) < min_test_bugs:
        raise ValueError("Insufficient independent held-out TEST bugs")
    if len({r["bug"] for r in per["calibration"]}) < min_test_bugs:
        raise ValueError("Insufficient independent CALIBRATION bugs")
    if nontrivial_propensities == 0:
        raise ValueError("Deterministic logs cannot identify an alternative bandit policy")
    if set(r["action"] for r in per["train"]) != set(ACTIONS):
        raise ValueError("Training lacks some actions; collect randomized coverage")
    if len({r["reward_definition"] for r in rows}) != 1 or not rows[0].get("reward_definition"):
        raise ValueError("Specify one identical, nonempty normalized reward definition")
    return per


def fit(rows, ridge=1.0):
    """Per-action ridge reward models used with an upper-confidence action score."""
    import numpy as np
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dimension = len(FEATURE_NAMES) + 1
    models = {}
    for action in ACTIONS:
        a = ridge * np.eye(dimension, dtype=float)
        b = np.zeros(dimension, dtype=float)
        count = 0
        for r in rows:
            if r["action"] != action:
                continue
            x = _feature_vector(r)
            # IPW mitigates a logged exploration policy that varies by context;
            # large weights are capped, an explicitly biased safety tradeoff.
            weight = min(20.0, 1.0 / r["propensity"])
            a += weight * np.outer(x, x)
            b += weight * r["reward"] * x
            count += 1
        if not count:
            raise ValueError(f"No observed training outcomes for {action}")
        inv = np.linalg.inv(a)
        models[action] = {"theta": (inv @ b).tolist(), "inverse": inv.tolist(), "n": count}
    return models


def choose(models, features, alpha):
    import numpy as np
    if alpha < 0:
        raise ValueError("alpha cannot be negative")
    x = _feature_vector({"features": features})
    scores = {}
    for action in ACTIONS:
        m = models[action]
        theta, inv = np.asarray(m["theta"]), np.asarray(m["inverse"])
        scores[action] = float(x @ theta + alpha * math.sqrt(max(0.0, x @ inv @ x)))
    return max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))


def off_policy_evaluate(models, rows, alpha, min_ess=5):
    """Self-normalized importance sampling, only for supported logged actions."""
    weights, weighted_rewards = [], []
    for r in rows:
        if choose(models, r["features"], alpha) == r["action"]:
            w = min(20.0, 1.0 / r["propensity"])
            weights.append(w)
            weighted_rewards.append(w * r["reward"])
    if not weights:
        raise ValueError("No logged support for the chosen policy")
    ess = sum(weights)**2 / sum(w*w for w in weights)
    if ess < min_ess:
        raise ValueError(f"Off-policy support too small, effective sample size {ess:.2f}")
    return {"snips_reward": sum(weighted_rewards) / sum(weights),
            "matched_actions": len(weights), "effective_sample_size": ess,
            "importance_weight_clip": 20.0,
            "note": "Observational off-policy estimate, not deployment validation; clipping adds bias."}


def train(rows, split, alpha_grid=(0.0, 0.1, 0.5, 1.0),
          min_train_bugs=10, min_test_bugs=5):
    per = validate_logs(rows, split, min_train_bugs, min_test_bugs)
    models = fit(per["train"])
    candidates = []
    for alpha in alpha_grid:
        try:
            estimate = off_policy_evaluate(models, per["calibration"], alpha)
        except ValueError:
            continue
        candidates.append((estimate["snips_reward"], -alpha, alpha, estimate))
    if not candidates:
        raise ValueError("No calibration policy has enough randomized logging support")
    _, _, alpha, calibration = max(candidates)
    test = off_policy_evaluate(models, per["test"], alpha)
    return {"schema": 1, "kind": "offline_logged_contextual_bandit",
            "feature_names": list(FEATURE_NAMES), "actions": list(ACTIONS),
            "models": models, "alpha": alpha, "calibration": calibration,
            "held_out_test": test,
            "reward_definition": rows[0]["reward_definition"],
            "train_bugs": sorted(split["train"]),
            "calibration_bugs": sorted(split["calibration"]),
            "test_bugs": sorted(split["test"]),
            "online_enabled": False,
            "limitations": ["Requires observed randomized actions and known propensities.",
                            "Not an online RAVE-FL controller, not a safe-stop guarantee.",
                            "No contextual bandit model is available until real logs pass all checks."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    result = train(rows, json.loads(args.split.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()

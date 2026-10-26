"""Stable names wrap existing engines. No new verification controller."""
from rave_fl.adaptive import AdaptivePilot
from rave_fl.pilot_types import PolicyConfig
from rave_fl.rank_repair import RankRepairPilot, RepairConfig
from rave_fl.selection_controls import SelectionControl

POLICIES = (
    "full",
    "rank_gate",
    "fixed_l1",
    "confidence",
    "v1_staged",
    "v1_cost_aware",
    "highest_suspicion",
    "ranker_disagreement",
    "ranking_ambiguity",
    "v2_utility",
    "uncertainty_aware",
)
DEFAULT_POLICIES = ("full", "fixed_l1", "highest_suspicion", "v2_utility", "uncertainty_aware")


def engine(candidates, backend, policy, opts, metadata, depths, fusion):
    if policy not in POLICIES:
        raise ValueError("Unsupported policy " + policy)
    mode = {
        "full": "full",
        "rank_gate": "rank_threshold",
        "confidence": "confidence",
        "v1_staged": "adaptive",
        "v1_cost_aware": "adaptive",
    }.get(policy, "repair_utility")
    cfg = PolicyConfig(
        token_budget=opts["budget"],
        batch_size=opts["batch_size"],
        mode=mode,
        max_steps=opts["max_steps"],
    )
    repair = RepairConfig(opts["max_full"], 5)
    if policy in {"full", "rank_gate", "confidence", "v1_staged", "v1_cost_aware"}:
        result = AdaptivePilot(candidates, backend, cfg)
    elif policy == "v2_utility":
        result = RankRepairPilot(candidates, backend, cfg, repair)
    else:
        result = SelectionControl(candidates, backend, cfg, repair, policy, metadata, depths)
    result.fusion = fusion
    result.cost_aware = policy != "v1_staged"
    return result

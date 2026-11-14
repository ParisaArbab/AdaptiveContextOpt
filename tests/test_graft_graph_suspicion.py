"""Regression checks for the new graph-selection treatment."""
from test_graft_controller import setup
from rave_fl.eval.fusion import Fusion
from rave_fl.graft.controller import GraftConfig, Router


def test_new_graph_policy_chooses_highest_observed_suspicion(tmp_path):
    candidates, session, _ = setup(
        tmp_path, GraftConfig(max_graph=1, max_full=0)
    )
    arm = Router(candidates, session, Fusion("rrf"), "l1_graph_suspicion").execute()
    routes = [event for event in arm["trace"] if event.get("phase") == "route"]
    first = next(event for event in routes if event["selected"])
    options = [row for row in first["options"] if row["eligible"]]
    assert first["selected"]["candidate"] == max(
        options,
        key=lambda r: (
            r["suspicion"],
            -r["reported_uncertainty"],
            -r["predicted_tokens"],
            -r["current_rank"],
            r["candidate"],
        ),
    )["candidate"]
    assert first["selected"]["stage"] == "graph"
    assert len([c for c in arm["calls"] if c["stage"] == "graph"]) == 1
    assert all(c["stage"] != "method_full" for c in arm["calls"])


def test_historical_graph_policy_still_selects_original_top_rank(tmp_path):
    candidates, session, _ = setup(
        tmp_path, GraftConfig(max_graph=1, max_full=0)
    )
    arm = Router(candidates, session, Fusion("rrf"), "l1_graph").execute()
    routes = [event for event in arm["trace"] if event.get("phase") == "route"]
    first = next(event for event in routes if event["selected"])
    assert first["selected"]["candidate"] == min(
        (row for row in first["options"] if row["eligible"]),
        key=lambda r: (
            next(c.rank for c in candidates if c.function_id == r["candidate"]),
            r["candidate"],
        ),
    )["candidate"]

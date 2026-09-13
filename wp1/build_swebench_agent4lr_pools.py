from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text()
    )


def require_top5(
    name: str,
    values,
):
    if not isinstance(values, list):
        raise ValueError(
            f"{name}: Top-5 is not a list"
        )

    if len(values) < 5:
        raise ValueError(
            f"{name}: expected at least 5 "
            f"candidates, found {len(values)}"
        )

    result = []

    for value in values[:5]:
        value = str(value).strip()

        if not value:
            raise ValueError(
                f"{name}: empty candidate"
            )

        result.append(value)

    return result


def append_source(
    pool,
    source,
    candidates,
):
    for source_rank, entity in enumerate(
        candidates,
        1,
    ):
        pool.append(
            {
                "position": len(pool) + 1,
                "source": source,
                "source_rank": source_rank,
                "entity": entity,
            }
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--instance",
        required=True,
    )

    args = ap.parse_args()

    root = (
        Path.home()
        / "AdaptiveContextOpt"
    )

    work = (
        root
        / "data"
        / "swebench_workspaces"
        / args.instance
    )

    outputs = (
        work / "outputs"
    )

    out = (
        outputs / "agent4lr"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load traditional localization methods.
    # These are shared by RAW and LeanCTX.
    # --------------------------------------------------

    sbir_data = load_json(
        outputs
        / "sbir"
        / "sbir.json"
    )

    ochiai_data = load_json(
        outputs
        / "ochiai"
        / "ochiai.json"
    )

    boostn_data = load_json(
        outputs
        / "boostn"
        / "boostn.json"
    )

    sr_data = load_json(
        outputs
        / "agent4sr_pair.json"
    )

    sbir = require_top5(
        "SBIR",
        sbir_data.get("top5"),
    )

    ochiai = require_top5(
        "Ochiai",
        ochiai_data.get("top5"),
    )

    boostn = require_top5(
        "BoostN",
        boostn_data.get("top5"),
    )

    raw_sr = require_top5(
        "RAW Agent4SR",
        sr_data.get(
            "raw",
            {},
        ).get("predictions"),
    )

    lean_sr = require_top5(
        "LeanCTX Agent4SR",
        sr_data.get(
            "leanctx",
            {},
        ).get("predictions"),
    )

    # --------------------------------------------------
    # Build exactly like FlexFL combination:
    #
    # 5 SBIR
    # 5 Ochiai
    # 5 BoostN
    # 5 Agent4SR
    #
    # IMPORTANT:
    # No deduplication.
    # --------------------------------------------------

    raw_pool = []

    append_source(
        raw_pool,
        "SBIR",
        sbir,
    )

    append_source(
        raw_pool,
        "Ochiai",
        ochiai,
    )

    append_source(
        raw_pool,
        "BoostN",
        boostn,
    )

    append_source(
        raw_pool,
        "Agent4SR_RAW",
        raw_sr,
    )

    lean_pool = []

    append_source(
        lean_pool,
        "SBIR",
        sbir,
    )

    append_source(
        lean_pool,
        "Ochiai",
        ochiai,
    )

    append_source(
        lean_pool,
        "BoostN",
        boostn,
    )

    append_source(
        lean_pool,
        "Agent4SR_LEANCTX",
        lean_sr,
    )

    if len(raw_pool) != 20:
        raise ValueError(
            f"RAW pool has {len(raw_pool)} "
            "positions instead of 20"
        )

    if len(lean_pool) != 20:
        raise ValueError(
            f"LeanCTX pool has {len(lean_pool)} "
            "positions instead of 20"
        )

    common_raw = [
        row["entity"]
        for row in raw_pool[:15]
    ]

    common_lean = [
        row["entity"]
        for row in lean_pool[:15]
    ]

    if common_raw != common_lean:
        raise ValueError(
            "Traditional candidate lists differ "
            "between RAW and LeanCTX."
        )

    result = {
        "instance_id": args.instance,

        "combination": [
            "SBIR Top-5",
            "Ochiai Top-5",
            "BoostN Top-5",
            "condition-specific Agent4SR Top-5",
        ],

        "deduplicated": False,

        "traditional_candidates_shared":
            True,

        "raw": {
            "candidate_count":
                len(raw_pool),

            "unique_candidate_count":
                len({
                    row["entity"]
                    for row in raw_pool
                }),

            "candidates":
                raw_pool,
        },

        "leanctx": {
            "candidate_count":
                len(lean_pool),

            "unique_candidate_count":
                len({
                    row["entity"]
                    for row in lean_pool
                }),

            "candidates":
                lean_pool,
        },
    }

    pair_path = (
        out
        / "candidate_pools.json"
    )

    raw_path = (
        out
        / "raw_candidates.json"
    )

    lean_path = (
        out
        / "leanctx_candidates.json"
    )

    pair_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    raw_path.write_text(
        json.dumps(
            result["raw"],
            indent=2,
        )
    )

    lean_path.write_text(
        json.dumps(
            result["leanctx"],
            indent=2,
        )
    )

    print()
    print(
        "============================================"
    )
    print(
        "RAW AGENT4LR CANDIDATES"
    )
    print(
        "============================================"
    )

    for row in raw_pool:
        print(
            f"{row['position']:2}. "
            f"[{row['source']:<12}] "
            f"{row['entity']}"
        )

    print()
    print(
        "============================================"
    )
    print(
        "LEANCTX AGENT4LR CANDIDATES"
    )
    print(
        "============================================"
    )

    for row in lean_pool:
        print(
            f"{row['position']:2}. "
            f"[{row['source']:<16}] "
            f"{row['entity']}"
        )

    print()
    print(
        "============================================"
    )
    print(
        "POOL SUMMARY"
    )
    print(
        "============================================"
    )

    print(
        "RAW positions       :",
        len(raw_pool),
    )

    print(
        "RAW unique entities :",
        result[
            "raw"
        ][
            "unique_candidate_count"
        ],
    )

    print(
        "Lean positions      :",
        len(lean_pool),
    )

    print(
        "Lean unique entities:",
        result[
            "leanctx"
        ][
            "unique_candidate_count"
        ],
    )

    print(
        "Common first 15     :",
        common_raw == common_lean,
    )

    print()
    print(
        "Saved:",
        pair_path,
    )


if __name__ == "__main__":
    main()

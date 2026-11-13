"""Audit and report a sealed fixed T0/T1 run without execution."""
import argparse
from collections import Counter
from contextlib import closing
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from statistics import mean
from types import SimpleNamespace

from rave_fl.eval.fusion import Fusion
from rave_fl.eval.backend import PromptSettlement
from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint
from rave_fl.graft.evaluation import PROFILES
from rave_fl.graft.fixed_evidence_experiment import ARMS, T0, T1, journal_identity
from rave_fl.pilot import write_json
from rave_fl.pilot_types import CandidateState, Observation, PilotCandidate


SYNTHETIC_LABEL = "SYNTHETIC SOFTWARE TEST — NOT RESEARCH RESULTS"
LIVE_LABEL = "REAL-MODEL SMOKE — DESCRIPTIVE PAIRED RESULTS"
METRICS = ("Top-1", "Top-5", "MRR")
COSTS = (
    "provider_reported_input_tokens",
    "provider_reported_output_tokens",
    "logical_tokens",
    "simulated_physical_tokens",
    "real_physical_tokens",
)
ARMLABEL = {T0: "T0", T1: "T1"}


def require(condition, message):
    if not condition:
        raise CompatibilityError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_arm(arm, schedule, planned, ready, bug, logical_budget, paid, journal, synthetic):
    calls = arm["calls"]
    targets = [candidate.function_id for candidate in schedule]
    require([c["target"] for c in calls] == (targets if ready else []),
            f"{bug}: observed targets differ from the complete scheduled arm")
    require(len(calls) == len(arm["accounting"]), f"{bug}: call accounting length mismatch")
    states, actions = {}, []
    input_tokens = output_tokens = settled = 0
    remaining = logical_budget
    for call, event, row in zip(calls, arm["accounting"], planned if ready else [], strict=True):
        target, key = call["target"], call["request_key"]
        require(key in journal and key not in paid and key == event["fingerprint"],
                f"{bug}: missing, duplicated or mismatched journal request")
        paid.add(key)
        response, inp, out = journal[key]
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        raw = content if isinstance(content, str) else ""
        remaining -= inp + out
        prompt_audit = row["prompt_accounting"]
        require(row["target"] == target and raw == call["raw_response"]
                and fingerprint(row["request"]["messages"]) == prompt_audit["messages_sha256"]
                and (inp, out) == (call["input_tokens"], call["output_tokens"])
                == (event["input_tokens"], event["output_tokens"])
                and event["actual_tokens"] == inp + out
                and event["reserved_tokens"] == row["logical_reservation"]
                and event["predicted_tokens"] == row["predicted_tokens"]
                and event["remaining"] == remaining
                and event["reuse"] in {"fresh", "journal"},
                f"{bug}: saved call, journal reply or accounting mismatch")
        input_tokens += inp
        output_tokens += out
        if call["valid"]:
            verdict = json.loads(raw)
            observation = Observation(**call["observations"][0])
            require(call["error"] is None and len(call["observations"]) == 1
                    and observation.valid and observation.function_id == target
                    and observation.suspicion == verdict["suspicion"]
                    and observation.uncertainty == verdict["uncertainty"]
                    and observation.reason == verdict["explanation"]
                    and observation.missing_evidence == verdict["missing_evidence"],
                    f"{bug}: valid observation does not match the raw verdict")
            states[target] = CandidateState(1, observation)
        settlement = PromptSettlement(**row["prompt_settlement"]) if "prompt_settlement" in row else None
        if settlement:
            require(settled + row["logical_reservation"] <= logical_budget
                    and event["settled_upper_tokens"] == settlement.upper == row["settled_upper_tokens"]
                    and inp == settlement.prompt_tokens and out <= settlement.output_cap,
                    f"{bug}: prompt settlement differs from the verified bound")
            settled += settlement.upper
            require(event["admission_remaining_tokens"] == logical_budget - settled,
                    f"{bug}: inconsistent admission remainder")
        actions.append((None, None, key, row["request"], row["logical_reservation"], settlement))
    complete = ready and len(states) == len(schedule)
    expected_status = "complete" if complete else "incomplete" if ready else "infeasible"
    require(arm["execution_status"] == expected_status, f"{bug}: inconsistent arm completion")
    ranking = None
    if complete:
        fusion = Fusion("rrf", rrf_k=60, tie="original")
        ranking = [c.function_id for c in fusion.order(schedule, fusion.scores(schedule, states))]
    require(arm["ranked_functions"] == ranking, f"{bug}: frozen ranking reconstruction differs")
    actual = input_tokens + output_tokens
    usage = dict(zip(COSTS, (input_tokens, output_tokens, actual,
                            actual if synthetic else 0, 0 if synthetic else actual), strict=True))
    require(all(arm[k] == v for k, v in usage.items()), f"{bug}: arm spending mismatch")
    return usage, actions


def read_journal(source):
    journal = source / "journal.sqlite3"
    wal = journal.with_name(journal.name + "-wal")
    require(journal.is_file(), "Saved journal is missing")
    require(not wal.exists() or wal.stat().st_size == 0,
            "Report requires a closed, checkpointed journal with no pending WAL data")
    return sqlite3.connect(journal.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def audit_saved_run(source, profile):
    result_path = source / "results.json"
    require(result_path.is_file(), "Finalized results are missing. Inspect the journal and accounting snapshot.")
    manifest = load_json(source / "manifest.json")
    result = load_json(result_path)
    spending = load_json(source / "accounting.json")
    result_hash = result.pop("sha256")
    require(fingerprint(result) == result_hash and fingerprint(manifest) == result["manifest_sha256"],
            "Result or manifest checksum mismatch")
    synthetic = result["synthetic"]
    require(result["schema"] == "graft-fixed-evidence-result-v1" and type(synthetic) is bool
            and manifest["provider"]["synthetic"] is synthetic and spending == result["spending"]
            and spending["synthetic"] is synthetic, "Report requires a sealed fixed-evidence result")
    plan = manifest["plan"]
    benchmark = plan["config"]["benchmark"]
    require(benchmark["backend"] == ("fake" if synthetic else "ollama") and plan["arm_configs"] == {
        name: asdict(config) for name, config in ARMS.items()
    } and benchmark["fusions"] == ["rrf"] and benchmark["rrf_k"] == 60
        and benchmark["tie"] == "original" and benchmark["rounding"] is None
        and benchmark["cache_mode"] == "off" and benchmark["admission"] == "strict"
        and benchmark["workers"] == 1 and benchmark["batch_size"] == 1
        and not any(plan["config"]["graft"][flag] for flag in ("paths", "repair", "typed_topology", "max_full")),
        "Manifest does not use the frozen T0/T1 protocol")
    if not synthetic:
        require(plan["run_admitted"] and all(p["status"] == "ready" for p in plan["pairs"])
                and manifest["provider"] == manifest["cohort_binding"]["live_attestation"]
                and manifest["provider"]["prompt_settlement_identity"] == plan["settlement_serving_identity"],
                "Live result requires the admitted cohort and certified serving identity")
    profile = profile or ("java_strict_v2" if benchmark["dataset"] == "defects4j" else "legacy_v1")
    require(profile in PROFILES, "Unknown evaluation profile")
    ledger = source / "raw_ledgers" / "requests.jsonl"
    rows, evidence, actions = [], [], {}
    paid = set()
    with closing(read_journal(source)) as db:
        journal = {key: (json.loads(raw), inp, out) for key, raw, inp, out in db.execute(
            "SELECT key,response,input,output FROM calls WHERE status='complete'"
        )}
        ids = [pair["instance_id"] for pair in plan["pairs"]]
        require(len(ids) == len(set(ids)) and [p["instance_id"] for p in result["pairs"]] == ids,
                "Saved results differ from the manifest defect schedule")
        for record, pair in zip(plan["pairs"], result["pairs"], strict=True):
            bug = pair["instance_id"]
            ready = record["status"] == "ready"
            schedule = [PilotCandidate(item["target"], "", item["prior_score"], item["prior_rank"], ("", "", ""), {})
                        for item in record["schedule"]]
            require(len(schedule) == len({c.function_id for c in schedule}) and (not ready or schedule),
                    f"{bug}: invalid candidate schedule")
            require(set(pair["arms"]) == set(ARMS), f"{bug}: result is missing a fixed arm")
            if ready:
                for a, b in zip(record["arms"][T0], record["arms"][T1], strict=True):
                    left, right = a["request"], b["request"]
                    require(a["target"] == b["target"] and a["input_allowance"] == b["input_allowance"]
                            and left == {**right, "messages": [right["messages"][0], left["messages"][1]]},
                            f"{bug}: paired requests differ outside the rendered evidence")
            complete = all(a["execution_status"] == "complete" for a in pair["arms"].values())
            require(pair["pair_status"] == ("complete" if complete else "incomplete"),
                    f"{bug}: paired completeness mismatch")
            coverage = []
            for name in ARMS:
                arm = pair["arms"][name]
                planned = record["arms"].get(name, [])
                usage, arm_actions = audit_arm(arm, schedule, planned, ready, bug, benchmark["budget"], paid, journal, synthetic)
                if ready:
                    actions[bug, name] = (None, None, arm_actions)
                evaluation = arm.get("metrics", {}).get(profile, {})
                metric = evaluation.get("method")
                require(complete or not arm.get("metrics"), f"{bug}: incomplete pair has accuracy metrics")
                recall = metric["candidate_pool_coverage"] if metric is not None else None
                if metric is not None:
                    require(metric["gold_locations"] > 0
                            and 0 <= metric["pool_gold_count"] <= metric["gold_locations"]
                            and recall == metric["pool_gold_count"] / metric["gold_locations"]
                            and (recall > 0 or all(metric[key] == 0 for key in METRICS)),
                            f"{bug}: inconsistent method metrics or candidate recall")
                coverage.append(recall)
                label_status = "not_evaluated"
                if complete:
                    label_status = ("available" if metric is not None else
                                    "method_labels_unavailable" if evaluation else "missing_for_profile")
                rows.append({
                    "instance_id": bug, "synthetic": synthetic, "arm": ARMLABEL[name], "pair_status": pair["pair_status"],
                    "arm_status": arm["execution_status"], "plan_status": record["status"],
                    "plan_reasons": " | ".join(record["reasons"]),
                    "label_status": label_status,
                    "file_labels_available": evaluation.get("file") is not None,
                    "label_reason": evaluation.get("gold_unavailable_reason") or (
                        "No saved method metric for selected profile" if complete and metric is None else ""
                    ),
                    "scheduled_candidates": len(schedule), "calls": len(arm["calls"]),
                    "invalid_calls": sum(not c["valid"] for c in arm["calls"]),
                    "candidate_recall": recall,
                    "gold_locations": metric["gold_locations"] if metric is not None else None,
                    "gold_in_pool": metric["pool_gold_count"] if metric is not None else None,
                    "first_relevant_rank": metric["first_relevant_rank"] if metric is not None else None,
                    "candidate_recall_status": "unavailable" if recall is None else "miss" if recall == 0
                    else "full" if recall == 1 else "partial",
                    **{k: metric[k] if metric is not None else None for k in METRICS},
                    **usage,
                    "planned_logical_hold_sum_tokens": sum(r["logical_reservation"] for r in planned)
                    if planned and all("logical_reservation" in r for r in planned) else None,
                    "planned_settled_upper_tokens": sum(r["settled_upper_tokens"] for r in planned)
                    if planned and all("settled_upper_tokens" in r for r in planned) else None,
                    **{key: record.get("physical_envelope_by_arm", {}).get(name, {}).get(key)
                       for key in ("serial_peak_tokens", "sum_full_window_reservations_tokens")},
                })
            require(coverage[0] == coverage[1], f"{bug}: candidate recall differs between fixed arms")
            evidence.extend(evidence_rows(record, synthetic=synthetic))
        view = SimpleNamespace(db=db, ledger=ledger)
        require(journal_identity(view, actions, live_model=None if synthetic else benchmark["model"])
                == result["journal_sha256"] and paid == set(journal),
                "Journal integrity or complete charge attribution mismatch")
    require(len(paid) == spending["new_calls"]
            and sum(row["logical_tokens"] for row in rows) == spending["known_provider_tokens"]
            and sum(row["simulated_physical_tokens"] for row in rows) == spending["simulated_provider_tokens"]
            and sum(row["real_physical_tokens"] for row in rows) == spending["physical_inference_tokens"]
            and spending["reserved_or_unknown_tokens"] == 0 and not spending["usage_unknown"]
            and not spending["pending"], "Reconstructed spending differs from the sealed snapshot")
    if "settled_upper_tokens" in spending:
        require(sum(row["planned_settled_upper_tokens"] or 0 for row in rows if row["plan_status"] == "ready")
                == spending["settled_upper_tokens"], "Reconstructed settlement differs from the sealed snapshot")
    return rows, evidence, profile, manifest, spending, result["labels_sha256"]


def evidence_rows(record, *, synthetic=True):
    comparisons = {item["target"]: item for item in record.get("evidence_comparisons", [])}
    rows = [{row["target"]: row for row in record["arms"].get(name, [])} for name in ARMS]
    for candidate in record["schedule"]:
        target = candidate["target"]
        a, b = (side.get(target, {}) for side in rows)
        comparison = comparisons.get(target, {})
        sizes = [row.get("input_units") for row in (a, b)]
        yield {
            "instance_id": record["instance_id"], "synthetic": synthetic, "target": target,
            "embedding_synthetic": record.get("embedding_synthetic"),
            "t0_status": a.get("status"), "t1_status": b.get("status"),
            "t0_reason": a.get("reason"), "t1_reason": b.get("reason"),
            "counter": a.get("counter"), "input_allowance": a.get("input_allowance"),
            "t0_input_units": sizes[0], "t1_input_units": sizes[1],
            "delta_input_units": sizes[1] - sizes[0] if None not in sizes else None,
            "jaccard": comparison.get("jaccard"),
            "common_ids": comparison.get("common_ids"), "t0_only_ids": comparison.get("t0_only_ids"),
            "t1_only_ids": comparison.get("t1_only_ids"), "source_nodes": comparison.get("evidence_nodes"),
            "t0_budget_rejected_ids": a.get("budget_rejected_ids"),
            "t1_budget_rejected_ids": b.get("budget_rejected_ids"),
            "t0_prompt_sha256": a.get("prompt_accounting", {}).get("messages_sha256"),
            "t1_prompt_sha256": b.get("prompt_accounting", {}).get("messages_sha256"),
            "t1_propagation": b.get("propagation"),
        }


def aggregate(rows):
    pairs = []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        matched = a["pair_status"] == "complete" and a["MRR"] is not None and b["MRR"] is not None
        paid_matched = a["calls"] == b["calls"] == a["scheduled_candidates"] and a["calls"] > 0
        pairs.append({
            "instance_id": a["instance_id"], "synthetic": a["synthetic"], "pair_status": a["pair_status"],
            "matched_complete_labeled": matched,
            "matched_full_candidate_recall": matched and a["candidate_recall"] == 1,
            "candidate_recall": a["candidate_recall"],
            **{f"delta_{k}": b[k] - a[k] if matched else None for k in METRICS},
            **{f"delta_{k}": b[k] - a[k] if paid_matched else None for k in COSTS},
        })
    cohorts = {}
    for key in ("matched_complete_labeled", "matched_full_candidate_recall"):
        eligible = {p["instance_id"] for p in pairs if p[key]}
        groups = {name: [r for r in rows if r["arm"] == name and r["instance_id"] in eligible]
                  for name in ARMLABEL.values()}
        cohorts[key] = {
            "n": len(eligible),
            "arms": {name: {k: mean(r[k] for r in group) if group else None
                            for k in (*METRICS, *COSTS, "candidate_recall")}
                     for name, group in groups.items()},
            "mean_T1_minus_T0": {k: mean(p[f"delta_{k}"] for p in pairs if p[key]) if eligible else None
                                for k in (*METRICS, *COSTS)},
        }
    operational = {}
    for name in ARMLABEL.values():
        group = [r for r in rows if r["arm"] == name]
        known_recall = [r["candidate_recall"] for r in group if r["candidate_recall"] is not None]
        operational[name] = {
            "arm_status_counts": dict(sorted(Counter(r["arm_status"] for r in group).items())),
            "label_status_counts": dict(sorted(Counter(r["label_status"] for r in group).items())),
            "recall_known_n": len(known_recall), "mean_candidate_recall": mean(known_recall) if known_recall else None,
            "candidate_recall_status_counts": dict(sorted(Counter(r["candidate_recall_status"] for r in group).items())),
            "largest_planned_arm_serial_peak_tokens": max(
                (r["serial_peak_tokens"] for r in group if r["serial_peak_tokens"] is not None), default=None
            ),
            "sum_full_holds_for_admitted_arms_tokens": sum(
                r["sum_full_window_reservations_tokens"] or 0 for r in group if r["plan_status"] == "ready"
            ),
            **{k: sum(r[k] for r in group) for k in (*COSTS, "calls", "invalid_calls")},
        }
    return pairs, cohorts, operational


def write_csv(path, rows, empty_fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def report(source, output, profile=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.is_relative_to(source), "Choose a separate report directory outside the saved run")
    rows, evidence, profile, manifest, spending, label_hash = audit_saved_run(source, profile)
    hashes = {name: file_hash(source / name) for name in (
        "manifest.json", "results.json", "accounting.json", "journal.sqlite3", "raw_ledgers/requests.jsonl"
    ) if (source / name).is_file()}
    return write_report(output, rows, evidence, profile, manifest["plan"], spending, label_hash, hashes)


def write_report(output, rows, evidence, profile, plan, spending, label_hash, hashes, cohort=None):
    pairs, cohorts, operational = aggregate(rows)
    opts = plan["config"]["benchmark"]
    synthetic = spending["synthetic"]
    summary = {
        "schema": "graft-fixed-evidence-report-v1", "synthetic": synthetic,
        "label": SYNTHETIC_LABEL if synthetic else LIVE_LABEL,
        "profile": profile, "selected_pairs": len(pairs),
        "pair_status_counts": dict(sorted(Counter(p["pair_status"] for p in pairs).items())),
        "model": opts["model"], "dataset": opts["dataset"], "provider": opts["backend"],
        "input_allowance": plan["config"]["graft"]["input_budget"],
        "output_cap_tokens": opts["output_tokens"], "context_window_tokens": opts["context_window"],
        "logical_budget_per_arm_tokens": opts["budget"],
        "physical_budget_per_arm_tokens": plan["config"]["per_arm_physical_budget"],
        "physical_budget_per_bug_tokens": opts["per_bug_physical_budget"],
        "physical_budget_per_run_tokens": opts["physical_budget"],
        "run_planned_physical_envelope": plan.get("physical_envelope"),
        "known_cost_definition": f"{'Synthetic provider' if synthetic else 'Provider-reported'} input plus output, settled verification only",
        "embedding_source_counts": dict(sorted(Counter(
            "synthetic_vectors" if pair.get("embedding_synthetic") is True else
            "real_vectors" if pair.get("embedding_synthetic") is False else "unavailable"
            for pair in plan["pairs"]
        ).items())),
        "operational": operational, "cohorts": cohorts, "spending": spending,
        "input_hashes": hashes,
        "labels_sha256": label_hash,
    }
    if cohort is not None:
        summary["sharding"] = cohort
    from rave_fl.graft.fixed_evidence_figures import render_figures

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_csv(output / "per_bug_arms.csv", rows, ["instance_id", "arm"])
    write_csv(output / "paired_differences.csv", pairs, ["instance_id", "pair_status"])
    write_csv(output / "evidence_selection.csv", evidence, ["instance_id", "target"])
    render_figures(output / "figures", summary, pairs, evidence)
    (output / "REPORT.md").write_text(markdown(summary, pairs), encoding="utf-8")
    return summary


def markdown(summary, pairs):
    synthetic = summary["synthetic"]
    lines = [
        "# Fixed T0/T1 rehearsal" if synthetic else "# Fixed T0/T1 real-model smoke", "", f"**{summary['label']}**", "",
        f"Saved configuration uses {summary['model']} with the {summary['provider']} provider, "
        f"the {summary['dataset']} dataset adapter and {summary['profile']} method evaluation.",
        "", f"Selected defects {summary['selected_pairs']}. Each candidate uses the same "
        f"{summary['input_allowance']:,}-unit packing allowance and "
        f"{summary['output_cap_tokens']:,}-token output cap across arms.",
        "", "## Operational accounting", "",
        "| Arm | Complete | Invalid replies | Provider input | Provider output | Logical | Simulated physical | Real physical |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in summary["operational"].items():
        values = [arm["arm_status_counts"].get("complete", 0), arm["invalid_calls"], *(arm[k] for k in COSTS)]
        lines.append(f"| {name} | " + " | ".join(f"{value:,}" for value in values) + " |")
    lines += ["", "Charges count each settled request once across resume. Logical cost counts its use in the arm. "
              "Per-defect planning holds are available separately in the arm CSV.", "",
              "## Accuracy and candidate recall", "",
              "| Cohort | Arm | Paired n | Top-1 | Top-5 | MRR | Candidate recall | Mean logical tokens |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, cohort in summary["cohorts"].items():
        for name, values in cohort["arms"].items():
            numbers = [values[k] for k in (*METRICS, "candidate_recall", "logical_tokens")]
            lines.append(f"| {key.replace('_', ' ')} | {name} | {cohort['n']} | " +
                         " | ".join("Unavailable" if value is None else f"{value:.3f}" for value in numbers) + " |")
    lines += [
        "", "The primary cohort requires both arms to complete and both method labels to be evaluable. "
        "Known out-of-pool gold contributes zero end-to-end accuracy. The full candidate recall sensitivity "
        "cohort requires all gold methods to be in the fixed candidate pool. Missing labels and incomplete "
        "pairs have unavailable accuracy. Candidate recall is the fraction of labeled gold methods in the pool.",
        "", "## Paired differences", "",
        "| Defect | Status | Candidate recall | Δ logical tokens | Δ MRR |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in pairs:
        cells = [pair[k] for k in ("candidate_recall", "delta_logical_tokens", "delta_MRR")]
        lines.append(f"| {pair['instance_id']} | {pair['pair_status']} | " +
                     " | ".join("Unavailable" if v is None else f"{v:g}" for v in cells) + " |")
    lines += ["", "Differences are T1 minus T0. Costs use all fully attempted paired schedules.",
              "", "## Figures", ""]
    for name in ("accuracy", "token_cost", "pareto", "paired_differences", "evidence_selection"):
        lines.append(f"- [{name.replace('_', ' ').capitalize()}](figures/{name}.svg)")
    lines += ["", "Sources are the sealed manifest, results, accounting snapshot, SQLite journal and raw request ledger. "
              "The report regenerates without model calls or access to the dataset, labels or embeddings.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES)
    args = parser.parse_args(argv)
    try:
        summary = report(args.run, args.output, args.profile)
    except (OSError, ValueError, KeyError, TypeError, ImportError, sqlite3.Error, RuntimeError) as exc:
        parser.exit(2, f"Fixed evidence report: {exc}\n")
    print(f"{summary['selected_pairs']} pairs reported in {args.output}")


if __name__ == "__main__":
    main()

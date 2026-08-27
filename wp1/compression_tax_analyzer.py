"""
compression_tax_analyzer.py — WP1 evaluation framework.

Answers the two questions the project is actually asking, per ablation arm:

  1. TOKENS — how much does the pipeline (and each element of it) save?
     Measured with a real tokenizer, split by pipeline stage, and reported
     both as an absolute count and as a saving relative to the control arm
     (`pure_flexfl`: FlexFL on raw output, no token optimization at all).

  2. ACCURACY — what does that cost in localization quality? FlexFL's own
     metric set (Top-1/3/5 hit rate, MAP, MRR) at method and file level,
     plus precision, so an arm that buys recall by predicting more entities
     doesn't look free.

and then the trade-off between them: `tokens_per_correct_top1` and
`accuracy_per_1k_tokens` make the two commensurable, and
`element_contributions` isolates each element's marginal effect by
differencing the arm that has it against the arm that doesn't.

Provisional results (lean-ctx reference mode, local_fallback captures, or
the key-free heuristic backend) are aggregated separately and excluded from
headline numbers rather than silently mixed in.

Taxonomy for WHY compression cost accuracy (unchanged):
  T1 lost value evidence | T2 lost stack frames | T3 collapsed sub-failures
  T4 dropped non-error lines | T5 over-summarized
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import ablation
import metrics


@dataclass
class InstanceOutcome:
    instance_id: str
    arm: str
    arm_flags: Dict[str, object]
    backend: str
    dataset: str
    language: str
    capture_mode: str          # "docker" | "local_fallback"
    compressor_mode: str       # "n/a" | "daemon" | "reference"
    predicted_functions: List[str]
    predicted_files: List[str]
    ground_truth_functions: List[str]
    ground_truth_files: List[str]
    method_scores: Optional[Dict[str, float]]
    file_scores: Optional[Dict[str, float]]
    token_report: Dict[str, object]
    feedback_rounds_used: int = 0
    feedback_restores: int = 0
    feedback_prunes: int = 0
    feedback_stop_reason: str = ""
    graph_expanded: List[str] = field(default_factory=list)
    taxonomy_tags: List[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    provisional: bool = False

    # -- token conveniences -------------------------------------------------
    @property
    def context_tokens(self) -> Dict[str, int]:
        return dict(self.token_report.get("context_tokens", {}))

    @property
    def agent_input_tokens(self) -> int:
        ctx = self.context_tokens
        return int(ctx.get("agent_input_tokens", ctx.get("tool_output_raw_tokens", 0)))

    @property
    def raw_tokens(self) -> int:
        return int(self.context_tokens.get("tool_output_raw_tokens", 0))

    @property
    def llm_tokens(self) -> int:
        return int(self.token_report.get("llm_total_tokens", 0))

    @property
    def provisional_reasons(self) -> List[str]:
        reasons = []
        if self.compressor_mode == "reference":
            reasons.append("lean-ctx reference mode (real daemon not reachable)")
        if self.capture_mode == "local_fallback":
            reasons.append("local_fallback capture (not the official SWE-bench image)")
        if self.backend == "heuristic":
            reasons.append("key-free heuristic backend (no real LLM reasoning)")
        return reasons

    @property
    def total_tokens(self) -> int:
        """What the pipeline actually consumed end to end. For an LLM backend
        that's every prompt+completion token across all stages. For the
        key-free heuristic backend there are no LLM calls, so the honest
        figure is the context the agent was handed."""
        return self.llm_tokens or self.agent_input_tokens


def score_file_level(predicted: List[str], ground_truth: List[str]) -> bool:
    """Retained for backwards compatibility with older result files. The
    binary "any predicted file is in ground truth" test ignores rank and
    rewards breadth, which is why metrics.score_ranked replaced it."""
    return bool(set(predicted) & set(ground_truth))


def classify_taxonomy(raw_text: str, compressed_text: str) -> List[str]:
    """Lightweight, explainable heuristics for tagging WHY something was
    lost — mirrors the manual classification originally used for
    sympy__sympy-16792 (T1) and astropy__astropy-12907 (T1/T2)."""
    tags = []
    raw_lines, comp_lines = raw_text.splitlines(), compressed_text.splitlines()
    comp_set = set(comp_lines)

    if any(("assert" in l or "AssertionError" in l) for l in raw_lines) and \
       compressed_text.count("...") >= 2:
        tags.append("T1_lost_value_evidence")

    raw_frames = sum(1 for l in raw_lines if l.strip().startswith("File \""))
    comp_frames = sum(1 for l in comp_lines if l.strip().startswith("File \""))
    if raw_frames >= 2 and comp_frames <= 1:
        tags.append("T2_lost_stack_frames")

    if raw_text.count("FAILED") >= 3 and compressed_text.count("FAILED") <= 1:
        tags.append("T3_collapsed_subfailures")

    dropped_nonerror = sum(
        1 for l in raw_lines
        if l not in comp_set and "error" not in l.lower() and "assert" not in l.lower() and l.strip()
    )
    if raw_lines and dropped_nonerror > 0.5 * len(raw_lines):
        tags.append("T4_dropped_nonerror_lines")

    if len(compressed_text) < 0.1 * len(raw_text) and "..." in compressed_text:
        tags.append("T5_over_summarized")

    return tags


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _token_summary(items: List[InstanceOutcome]) -> dict:
    if not items:
        return {}
    stage_totals: Dict[str, int] = {}
    for o in items:
        for stage, usage in (o.token_report.get("by_stage") or {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0) + int(usage.get("total_tokens", 0))
    n = len(items)
    tokenizers = {o.token_report.get("tokenizer", "unknown") for o in items}
    return {
        "mean_raw_capture_tokens": _mean([o.raw_tokens for o in items]),
        "mean_agent_input_tokens": _mean([o.agent_input_tokens for o in items]),
        "mean_llm_tokens": _mean([o.llm_tokens for o in items]),
        "mean_total_tokens": _mean([o.total_tokens for o in items]),
        "mean_tokens_by_stage": {s: round(t / n, 2) for s, t in sorted(stage_totals.items())},
        "mean_llm_calls": _mean([int(o.token_report.get("llm_calls", 0)) for o in items]),
        "tokenizer": sorted(tokenizers),
        "tokenizer_exact": all(o.token_report.get("tokenizer_exact") for o in items),
    }


def _feedback_summary(items: List[InstanceOutcome]) -> dict:
    active = [o for o in items if o.arm_flags.get("use_feedback")]
    if not active:
        return {}
    reasons: Dict[str, int] = {}
    for o in active:
        reasons[o.feedback_stop_reason or "unknown"] = reasons.get(o.feedback_stop_reason or "unknown", 0) + 1
    return {
        "mean_rounds": _mean([o.feedback_rounds_used for o in active]),
        "mean_restores": _mean([o.feedback_restores for o in active]),
        "mean_prunes": _mean([o.feedback_prunes for o in active]),
        "stop_reasons": reasons,
    }


def summarize_arm(items: List[InstanceOutcome]) -> dict:
    method = metrics.aggregate([o.method_scores for o in items])
    file_ = metrics.aggregate([o.file_scores for o in items])
    tokens = _token_summary(items)
    top1_hits = sum(1 for o in items if (o.method_scores or {}).get("top1"))
    total_tokens = sum(o.total_tokens for o in items)
    return {
        "n_instances": len(items),
        "accuracy": {"method_level": method, "file_level": file_},
        "tokens": tokens,
        "feedback": _feedback_summary(items),
        "tradeoff": {
            # cost of one correct top-1 localization, the metric that makes
            # "cheap but wrong" and "accurate but expensive" comparable
            "tokens_per_correct_top1": round(total_tokens / top1_hits, 1) if top1_hits else None,
            "top1_per_1k_tokens": round(1000 * top1_hits / total_tokens, 4) if total_tokens else None,
        },
        "mean_wall_seconds": _mean([o.wall_seconds for o in items]),
        "taxonomy_tag_counts": _tag_counts(items),
    }


def _tag_counts(items: List[InstanceOutcome]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for o in items:
        for tag in o.taxonomy_tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _delta_vs(summary: dict, control: dict) -> dict:
    """Arm vs. control, on the two axes that matter."""
    if not summary or not control:
        return {}
    a_tok = summary["tokens"].get("mean_total_tokens", 0)
    c_tok = control["tokens"].get("mean_total_tokens", 0)
    out = {
        "token_reduction_pct": round(100 * (1 - a_tok / c_tok), 2) if c_tok else None,
        "mean_token_delta": round(a_tok - c_tok, 2),
    }
    for level in ("method_level", "file_level"):
        a = summary["accuracy"].get(level) or {}
        c = control["accuracy"].get(level) or {}
        for key in ("top1", "top3", "top5", "map", "mrr"):
            if key in a and key in c:
                out[f"{level}_{key}_delta"] = round(a[key] - c[key], 4)
    return out


def element_contributions(by_arm: Dict[str, dict]) -> dict:
    """Marginal effect of each element, by differencing the arm that has it
    against the arm that doesn't, holding everything else fixed.

    Reported as (with-element minus without-element), so a negative
    token delta means the element SAVED tokens and a positive accuracy
    delta means it HELPED. Each element gets every comparable arm pair
    available in the result set, because the marginal effect of (say)
    lean-ctx is not guaranteed to be the same with and without graphify —
    if the pairs disagree, the elements interact and the single-number
    summary is the thing that's wrong, not the data.
    """
    out: Dict[str, List[dict]] = {}
    arms = {name: ablation.ARMS[name] for name in by_arm if name in ablation.ARMS}

    for element in ablation.ELEMENTS:
        attr = f"use_{element}"
        pairs = []
        for name_on, arm_on in arms.items():
            if not getattr(arm_on, attr):
                continue
            for name_off, arm_off in arms.items():
                if getattr(arm_off, attr):
                    continue
                others_on = {e: getattr(arm_on, f"use_{e}") for e in ablation.ELEMENTS if e != element}
                others_off = {e: getattr(arm_off, f"use_{e}") for e in ablation.ELEMENTS if e != element}
                if others_on != others_off:
                    continue
                pairs.append({
                    "with": name_on,
                    "without": name_off,
                    "holding": others_on,
                    **_delta_vs(by_arm[name_on], by_arm[name_off]),
                })
        if pairs:
            out[element] = pairs
    return out


def analyze(results_path: Path) -> dict:
    payload = json.loads(results_path.read_text())
    # accept both the new {"config","outcomes"} envelope and a bare list
    raw_results = payload["outcomes"] if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    outcomes = [InstanceOutcome(**r) for r in raw_results]

    by_arm: Dict[str, List[InstanceOutcome]] = {}
    for o in outcomes:
        by_arm.setdefault(o.arm, []).append(o)

    # Provisionality is a property of the RUN, not of individual arms.
    # Arms differ in whether they use lean-ctx, and lean-ctx in reference
    # mode is provisional — so scoring arm-by-arm would silently split the
    # table into a "trusted" half and a "provisional" half and then compare
    # arms that were never comparable. If anything in the run is
    # provisional, the whole comparison is, and the reasons are listed.
    reasons: Dict[str, List[str]] = {}
    for o in outcomes:
        for reason in o.provisional_reasons:
            reasons.setdefault(reason, [])
            if o.arm not in reasons[reason]:
                reasons[reason].append(o.arm)

    basis = "provisional" if reasons else "trusted"
    headline = {arm: summarize_arm(items) for arm, items in by_arm.items()}
    provisional_summary: Dict[str, dict] = {}
    control = headline.get(ablation.CONTROL_ARM)

    vs_control = {arm: _delta_vs(summary, control)
                  for arm, summary in headline.items()
                  if control and arm != ablation.CONTROL_ARM}

    return {
        "config": config,
        "basis": basis,
        "basis_note": (
            "This run is provisional — use it to validate the pipeline, not "
            "to publish. Reasons below."
            if basis == "provisional" else
            "All arms measured under publishable conditions."
        ),
        "provisional_reasons": {r: sorted(arms) for r, arms in sorted(reasons.items())},
        "summary_by_arm": headline,
        "provisional_by_arm": provisional_summary if basis == "trusted" else {},
        "vs_control": {"control_arm": ablation.CONTROL_ARM, "deltas": vs_control},
        "element_contributions": element_contributions(headline),
        "compression_tax_cases": _tax_cases(outcomes),
    }


def _tax_cases(outcomes: List[InstanceOutcome]) -> Dict[str, List[dict]]:
    """Instances the control arm localized correctly but an optimized arm
    did not — the actual compression tax, with the taxonomy tag explaining
    why. (This is where the old version had an indentation bug that folded
    every instance into the last one's data.)"""
    by_instance: Dict[str, Dict[str, InstanceOutcome]] = {}
    for o in outcomes:
        by_instance.setdefault(o.instance_id, {})[o.arm] = o

    tax_cases: Dict[str, List[dict]] = {}
    for instance_id, arms in by_instance.items():
        control = arms.get(ablation.CONTROL_ARM)
        if not control or not (control.method_scores or control.file_scores):
            continue
        control_hit = (control.method_scores or control.file_scores or {}).get("top5", 0)
        if not control_hit:
            continue
        for arm_name, o in arms.items():
            if arm_name == ablation.CONTROL_ARM:
                continue
            hit = (o.method_scores or o.file_scores or {}).get("top5", 0)
            if not hit:
                tax_cases.setdefault(arm_name, []).append({
                    "instance_id": instance_id,
                    "taxonomy_tags": o.taxonomy_tags,
                    "token_saving_vs_control": control.total_tokens - o.total_tokens,
                    "provisional": o.provisional,
                })
    return tax_cases


def render_text_report(report: dict) -> str:
    """Compact console view — the same numbers as the JSON, laid out so the
    accuracy/token trade-off is readable at a glance."""
    lines: List[str] = []
    lines.append(f"basis: {report['basis']} — {report['basis_note']}")
    for reason, arms in report.get("provisional_reasons", {}).items():
        lines.append(f"  - {reason}  [arms: {', '.join(arms)}]")
    header = (f"{'arm':<22}{'n':>4}{'top1':>8}{'top3':>8}{'top5':>8}{'MAP':>8}"
              f"{'MRR':>8}{'tokens':>10}{'saved':>10}")
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    deltas = report["vs_control"]["deltas"]
    for arm, summary in report["summary_by_arm"].items():
        m = summary["accuracy"]["method_level"] or summary["accuracy"]["file_level"] or {}
        tok = summary["tokens"].get("mean_total_tokens", 0)
        red = deltas.get(arm, {}).get("token_reduction_pct")
        red_s = f"{red:+.1f}%" if red is not None else "control"
        lines.append(
            f"{arm:<22}{summary['n_instances']:>4}{m.get('top1', 0):>8.3f}"
            f"{m.get('top3', 0):>8.3f}{m.get('top5', 0):>8.3f}{m.get('map', 0):>8.3f}"
            f"{m.get('mrr', 0):>8.3f}{tok:>10.0f}{red_s:>10}"
        )
    lines.append("")
    lines.append("element contributions — 'tokens' is the mean per-instance delta "
                 "(negative = the element saved tokens); the % is the matching "
                 "reduction (positive = saved):")
    for element, pairs in report["element_contributions"].items():
        for p in pairs:
            lines.append(
                f"  {element:<10} {p['with']:<22} vs {p['without']:<22} "
                f"tokens {p.get('mean_token_delta', 0):+.0f} "
                f"({p.get('token_reduction_pct', 0) or 0:+.1f}%)  "
                f"top1 {p.get('method_level_top1_delta', 0):+.3f}  "
                f"MRR {p.get('method_level_mrr_delta', 0):+.3f}"
            )
    tax = report["compression_tax_cases"]
    if tax:
        lines.append("")
        lines.append("compression tax cases (control hit top-5, arm missed):")
        for arm, cases in tax.items():
            tags = sorted({t for c in cases for t in c["taxonomy_tags"]})
            lines.append(f"  {arm:<22} {len(cases)} case(s)  tags={tags or ['(none)']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/wp1_results.json")
    ap.add_argument("--out", default="results/compression_tax_report.json")
    ap.add_argument("--json", action="store_true", help="print the full JSON report too")
    args = ap.parse_args()

    report = analyze(Path(args.results))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(render_text_report(report))
    if args.json:
        print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")

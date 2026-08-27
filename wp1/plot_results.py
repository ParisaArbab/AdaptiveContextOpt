"""
plot_results.py — figures for the evaluation framework.

Reads the analyzer's report (and, for distribution plots, the raw results)
and writes one figure per question the study asks:

  1. accuracy_by_arm.png       Top-1/3/5 + MRR per arm, method and file level
  2. tokens_by_arm.png         mean tokens per arm, labelled with % saved vs control
  3. tradeoff.png              THE headline figure — tokens (x) vs accuracy (y),
                               with the Pareto frontier drawn, so "cheaper but
                               worse" and "better but dearer" are visually separable
  4. tokens_by_stage.png       stacked per-stage breakdown: where tokens actually go
  5. element_contributions.png each element's marginal token saving vs accuracy cost
  6. compression_funnel.png    raw capture -> post-compression -> agent input
  7. taxonomy.png              T1-T5 tags and compression-tax cases per arm
  8. per_instance_heatmap.png  which instances each arm got right (Top-5)

Arms keep the same colour across every figure, and the control arm is drawn
in a fixed neutral colour wherever it appears, so figures can be read side by
side without re-checking the legend.

Usage:
    python plot_results.py --report results/compression_tax_report.json \
        --results results/wp1_results.json --out-dir results/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")           # headless: this runs on servers and in CI
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting: pip install matplotlib\n"
        f"(import failed: {e})"
    )

CONTROL_COLOR = "#7a7a7a"
PALETTE = ["#3b7dd8", "#e07b39", "#4fa363", "#c0504d", "#8064a2", "#2c8f9e", "#b8860b"]
GRID = dict(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)


def _fig(ax_title: str, figsize=(10, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(ax_title, fontsize=12, weight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def _save(fig, out_dir: Path, name: str, written: List[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)


class ArmStyle:
    """Stable colour per arm across every figure."""

    def __init__(self, arms: List[str], control: str):
        self.control = control
        self.colors: Dict[str, str] = {}
        i = 0
        for arm in arms:
            if arm == control:
                self.colors[arm] = CONTROL_COLOR
            else:
                self.colors[arm] = PALETTE[i % len(PALETTE)]
                i += 1

    def __call__(self, arm: str) -> str:
        return self.colors.get(arm, "#999999")


def _acc(summary: dict, level: str = "method_level") -> dict:
    acc = summary.get("accuracy", {})
    return acc.get(level) or acc.get("file_level") or {}


def _ordered_arms(report: dict) -> List[str]:
    """Control last, so it reads as the baseline the others are measured against."""
    control = report["vs_control"]["control_arm"]
    arms = [a for a in report["summary_by_arm"] if a != control]
    if control in report["summary_by_arm"]:
        arms.append(control)
    return arms


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_accuracy(report, arms, style, out_dir, written):
    for level, label in (("method_level", "method"), ("file_level", "file")):
        present = [a for a in arms if _acc(report["summary_by_arm"][a], level)]
        if not present:
            continue
        keys = ["top1", "top3", "top5", "mrr", "map"]
        fig, ax = _fig(f"Localization accuracy by pipeline arm ({label} level)")
        width = 0.8 / len(present)
        xs = range(len(keys))
        for i, arm in enumerate(present):
            vals = [_acc(report["summary_by_arm"][arm], level).get(k, 0.0) for k in keys]
            ax.bar([x + i * width for x in xs], vals, width, label=arm,
                   color=style(arm), edgecolor="white", linewidth=0.6)
        ax.set_xticks([x + 0.4 - width / 2 for x in xs])
        ax.set_xticklabels(["Top-1", "Top-3", "Top-5", "MRR", "MAP"])
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.grid(**GRID)
        ax.legend(fontsize=8, ncol=2, frameon=False)
        _save(fig, out_dir, f"accuracy_by_arm_{label}.png", written)


def plot_tokens(report, arms, style, out_dir, written):
    deltas = report["vs_control"]["deltas"]
    control = report["vs_control"]["control_arm"]
    vals = [report["summary_by_arm"][a]["tokens"].get("mean_total_tokens", 0) for a in arms]
    fig, ax = _fig("Mean tokens consumed per instance, by pipeline arm")
    bars = ax.bar(arms, vals, color=[style(a) for a in arms],
                  edgecolor="white", linewidth=0.6)
    top = max(vals) if vals else 1
    for arm, bar, val in zip(arms, bars, vals):
        red = deltas.get(arm, {}).get("token_reduction_pct")
        tag = "control" if arm == control else (f"{red:+.1f}%" if red is not None else "")
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.02,
                f"{val:,.0f}\n{tag}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("tokens")
    ax.set_ylim(0, top * 1.22)
    ax.grid(**GRID)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.text(0.995, -0.28, "% = token reduction vs control (positive = saved)",
            transform=ax.transAxes, ha="right", fontsize=7, color="#555")
    _save(fig, out_dir, "tokens_by_arm.png", written)


def plot_tradeoff(report, arms, style, out_dir, written, metric="top1"):
    """The headline figure: is the token saving worth the accuracy?"""
    pts = []
    for arm in arms:
        summary = report["summary_by_arm"][arm]
        pts.append((arm,
                    summary["tokens"].get("mean_total_tokens", 0),
                    _acc(summary).get(metric, 0.0)))
    if not pts:
        return
    fig, ax = _fig(f"Token / accuracy trade-off (accuracy = {metric.upper()})",
                   figsize=(9, 6))

    # Pareto frontier: nothing is both cheaper AND more accurate.
    frontier = sorted(
        [p for p in pts
         if not any(q[1] < p[1] and q[2] > p[2] for q in pts)],
        key=lambda p: p[1],
    )
    if len(frontier) > 1:
        ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
                color="#bbb", linestyle="--", linewidth=1.2, zorder=1,
                label="Pareto frontier")

    # Arms that behave similarly land on top of each other, which is exactly
    # when the labels matter most — stagger any label whose point is close to
    # one already placed, in normalized axis space so the threshold means the
    # same thing whatever the token scale.
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    x_span = (max(xs) - min(xs)) or 1.0
    y_span = (max(ys) - min(ys)) or 1.0
    offsets = [(0, 14), (0, -22), (0, 30), (0, -38)]
    placed: List[tuple] = []
    for arm, tokens, acc in sorted(pts, key=lambda p: (p[1], p[2])):
        norm = (tokens / x_span, acc / y_span)
        collisions = sum(1 for q in placed
                         if abs(q[0] - norm[0]) < 0.06 and abs(q[1] - norm[1]) < 0.12)
        placed.append(norm)
        ax.scatter(tokens, acc, s=170, color=style(arm), zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.annotate(arm, (tokens, acc), textcoords="offset points",
                    xytext=offsets[collisions % len(offsets)], ha="center", fontsize=8)
    ax.set_xlabel("mean tokens per instance  (left = cheaper)")
    ax.set_ylabel(f"{metric.upper()}  (up = more accurate)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.margins(0.16)
    if len(frontier) > 1:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.text(0.0, -0.13, "upper-left is strictly better: fewer tokens, higher accuracy",
            transform=ax.transAxes, fontsize=7, color="#555")
    _save(fig, out_dir, f"tradeoff_{metric}.png", written)


def plot_tokens_by_stage(report, arms, out_dir, written):
    stages: List[str] = []
    for arm in arms:
        for stage in report["summary_by_arm"][arm]["tokens"].get("mean_tokens_by_stage", {}):
            if stage not in stages:
                stages.append(stage)
    if not stages:
        return  # heuristic backend: no LLM calls, nothing to break down
    fig, ax = _fig("Where the tokens go: per-stage breakdown by arm")
    bottoms = [0.0] * len(arms)
    for i, stage in enumerate(stages):
        vals = [report["summary_by_arm"][a]["tokens"]
                .get("mean_tokens_by_stage", {}).get(stage, 0) for a in arms]
        ax.bar(arms, vals, bottom=bottoms, label=stage.replace("_", " "),
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("mean tokens per instance")
    ax.grid(**GRID)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out_dir, "tokens_by_stage.png", written)


def plot_element_contributions(report, out_dir, written):
    contributions = report.get("element_contributions", {})
    rows = []
    for element, pairs in contributions.items():
        for p in pairs:
            rows.append((f"{element}\n({p['with']} vs {p['without']})",
                         p.get("token_reduction_pct") or 0.0,
                         p.get("method_level_top1_delta",
                               p.get("file_level_top1_delta", 0.0)) or 0.0))
    if not rows:
        return
    labels = [r[0] for r in rows]
    savings = [r[1] for r in rows]
    acc_deltas = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Marginal contribution of each pipeline element",
                 fontsize=12, weight="bold")
    ax1.bar(labels, savings, color=["#4fa363" if v >= 0 else "#c0504d" for v in savings],
            edgecolor="white", linewidth=0.6)
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.set_ylabel("token reduction %\n(positive = saved)")
    ax1.grid(**GRID)
    ax2.bar(labels, acc_deltas,
            color=["#3b7dd8" if v >= 0 else "#c0504d" for v in acc_deltas],
            edgecolor="white", linewidth=0.6)
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.set_ylabel("Top-1 delta\n(positive = helped)")
    ax2.grid(**GRID)
    plt.setp(ax2.get_xticklabels(), rotation=18, ha="right", fontsize=7)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir, "element_contributions.png", written)


def plot_compression_funnel(report, arms, style, out_dir, written):
    keys = ["mean_raw_capture_tokens", "mean_agent_input_tokens"]
    labels = ["raw capture", "agent input (post compression + feedback)"]
    fig, ax = _fig("Context reduction: raw capture vs what the agent actually reads")
    width = 0.8 / len(keys)
    xs = range(len(arms))
    for i, (key, label) in enumerate(zip(keys, labels)):
        vals = [report["summary_by_arm"][a]["tokens"].get(key, 0) for a in arms]
        ax.bar([x + i * width for x in xs], vals, width, label=label,
               color=PALETTE[i], edgecolor="white", linewidth=0.6)
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean tokens per instance")
    ax.grid(**GRID)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out_dir, "compression_funnel.png", written)


def plot_taxonomy(report, arms, style, out_dir, written):
    tax = report.get("compression_tax_cases", {})
    tag_counts: Dict[str, Dict[str, int]] = {}
    for arm in arms:
        counts = report["summary_by_arm"][arm].get("taxonomy_tag_counts", {})
        for tag, n in counts.items():
            tag_counts.setdefault(tag, {})[arm] = n
    if not tag_counts and not tax:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Why compression cost accuracy", fontsize=12, weight="bold")

    ax = axes[0]
    if tag_counts:
        tags = sorted(tag_counts)
        width = 0.8 / max(1, len(arms))
        for i, arm in enumerate(arms):
            vals = [tag_counts.get(t, {}).get(arm, 0) for t in tags]
            ax.bar([x + i * width for x in range(len(tags))], vals, width,
                   label=arm, color=style(arm), edgecolor="white", linewidth=0.6)
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(tags))])
        ax.set_xticklabels([t.split("_", 1)[0] for t in tags])
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("taxonomy tags fired (T1-T5)", fontsize=10)
    ax.set_ylabel("instances")
    ax.grid(**GRID)

    ax = axes[1]
    if tax:
        names = list(tax)
        ax.bar(names, [len(tax[a]) for a in names],
               color=[style(a) for a in names], edgecolor="white", linewidth=0.6)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    ax.set_title("compression tax cases\n(control hit Top-5, arm missed)", fontsize=10)
    ax.set_ylabel("instances")
    ax.grid(**GRID)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "taxonomy.png", written)


def plot_per_instance(results_path: Optional[Path], arms, out_dir, written):
    """Which instances each arm localized — makes it obvious whether arms
    fail on the SAME instances (a hard-instance effect) or different ones
    (a real compression effect)."""
    if not results_path or not results_path.exists():
        return
    payload = json.loads(results_path.read_text())
    outcomes = payload["outcomes"] if isinstance(payload, dict) else payload
    grid: Dict[str, Dict[str, float]] = {}
    for o in outcomes:
        scores = o.get("method_scores") or o.get("file_scores") or {}
        grid.setdefault(o["instance_id"], {})[o["arm"]] = scores.get("top5", 0.0)
    instances = sorted(grid)
    if not instances:
        return
    matrix = [[grid.get(i, {}).get(a, float("nan")) for i in instances] for a in arms]

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(instances) + 4), 0.5 * len(arms) + 2.5))
    ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=8)
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(instances, rotation=90, fontsize=6)
    ax.set_title("Per-instance Top-5 hit by arm (green = localized)",
                 fontsize=12, weight="bold", pad=12)
    ax.legend(handles=[Patch(color="#1a9850", label="hit"),
                       Patch(color="#d73027", label="miss")],
              fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.005, 1.0))
    _save(fig, out_dir, "per_instance_heatmap.png", written)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="results/compression_tax_report.json")
    ap.add_argument("--results", default="results/wp1_results.json",
                    help="raw outcomes, for the per-instance heatmap")
    ap.add_argument("--out-dir", default="results/plots")
    ap.add_argument("--metric", default="top1", choices=["top1", "top3", "top5", "mrr", "map"],
                    help="accuracy axis for the trade-off figure")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    if not report.get("summary_by_arm"):
        raise SystemExit(f"{args.report} has no arm summaries — did the benchmark produce outcomes?")

    arms = _ordered_arms(report)
    style = ArmStyle(arms, report["vs_control"]["control_arm"])
    out_dir = Path(args.out_dir)
    written: List[Path] = []

    plot_accuracy(report, arms, style, out_dir, written)
    plot_tokens(report, arms, style, out_dir, written)
    plot_tradeoff(report, arms, style, out_dir, written, metric=args.metric)
    plot_tokens_by_stage(report, arms, out_dir, written)
    plot_element_contributions(report, out_dir, written)
    plot_compression_funnel(report, arms, style, out_dir, written)
    plot_taxonomy(report, arms, style, out_dir, written)
    plot_per_instance(Path(args.results), arms, out_dir, written)

    if report.get("basis") == "provisional":
        (out_dir / "PROVISIONAL.txt").write_text(
            "These figures are PROVISIONAL.\n\n"
            + report.get("basis_note", "") + "\n\n"
            + "\n".join(f"- {r}  [arms: {', '.join(a)}]"
                        for r, a in report.get("provisional_reasons", {}).items())
            + "\n"
        )
        written.append(out_dir / "PROVISIONAL.txt")

    print(f"wrote {len(written)} file(s) -> {out_dir}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()

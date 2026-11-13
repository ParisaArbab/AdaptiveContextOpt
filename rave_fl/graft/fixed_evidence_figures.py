"""Reproducible figures for the synthetic fixed T0/T1 report."""
from collections import defaultdict

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from rave_fl.graft.fixed_evidence_reporting import METRICS

COLORS = ("#0072B2", "#D55E00")
ARMS = ("T0", "T1")
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.labelcolor": "#243147", "text.color": "#243147",
    "grid.color": "#CBD5E1", "grid.alpha": 0.65, "svg.hashsalt": "graft-phase4-v1",
}


def figure(title, summary, panels=1, height=4.6, footnote=None):
    fig = Figure(figsize=(8.6, height), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, panels, squeeze=False)[0]
    fig.subplots_adjust(left=.13, right=.95, top=.76, bottom=.24, wspace=.36)
    fig.text(.13, .94, title, fontsize=17, weight="bold")
    fig.text(.13, .88, summary["label"], fontsize=9, weight="bold", color="#8A3600")
    context = (f"Saved fixed-evidence run · {summary['model']} / {summary['provider']} · "
               f"Packing allowance {summary['input_allowance']:,} units · Output cap {summary['output_cap_tokens']} tokens")
    fig.text(.13, .062, context, fontsize=8)
    if footnote:
        fig.text(.13, .103, footnote, fontsize=8)
    return fig, axes


def empty(ax, message):
    ax.set_axis_off()
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#475569")


def accuracy(summary, cohort):
    fig, (ax,) = figure("Method-level accuracy", summary,
                        footnote=f"Joint-complete labeled pairs, n = {cohort['n']}. Known pool misses count as zero.")
    if not cohort["n"]:
        empty(ax, "No joint-complete labeled pairs")
        return fig
    positions = np.arange(len(METRICS))
    for offset, name, color in zip((-.18, .18), ARMS, COLORS, strict=True):
        values = [cohort["arms"][name][k] for k in METRICS]
        bars = ax.bar(positions + offset, values, width=.33, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(positions, METRICS)
    ax.set_ylim(0, 1.17)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    return fig


def token_cost(summary):
    fig, (ax,) = figure("Verification token cost", summary,
                        footnote="All settled requests, including invalid replies.")
    for name, color in zip(ARMS, COLORS, strict=True):
        row = summary["operational"][name]
        inp, out = row["provider_reported_input_tokens"], row["provider_reported_output_tokens"]
        ax.bar(name, inp, width=.5, color=color, label=f"{name} input")
        ax.bar(name, out, width=.5, bottom=inp, color=color, hatch="////",
               edgecolor="white", label=f"{name} output")
        ax.annotate(f"{inp + out:,}", (name, inp + out), xytext=(0, 5),
                    textcoords="offset points", ha="center")
    ax.set_ylim(0, max(1, max(row["logical_tokens"] for row in summary["operational"].values()) * 1.25))
    ax.set_ylabel("Simulated provider tokens" if summary["synthetic"] else "Provider-reported tokens")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=8)
    return fig


def pareto(summary, cohort):
    fig, (ax,) = figure("Accuracy–cost Pareto comparison", summary,
                        footnote=f"Joint-complete labeled pairs, n = {cohort['n']}. Cost is provider input plus output.")
    if not cohort["n"]:
        empty(ax, "No matched points available")
        return fig
    points = [(cohort["arms"][name]["logical_tokens"], cohort["arms"][name]["MRR"]) for name in ARMS]
    for name, color, (cost, quality) in zip(ARMS, COLORS, points, strict=True):
        dominated = any(x <= cost and y >= quality and (x < cost or y > quality) for x, y in points)
        ax.scatter(cost, quality, s=106, marker="X" if dominated else "o",
                   color=color, edgecolor="white", linewidth=.7, label=f"{name} · {'dominated' if dominated else 'non-dominated'}")
    ax.margins(x=.24)
    ax.set_ylim(-.06, 1.15)
    ax.set_xlabel("Mean logical tokens per defect · lower is better")
    ax.set_ylabel("MRR · higher is better")
    ax.grid()
    ax.legend(frameon=False, loc="upper right")
    return fig


def paired_differences(summary, pairs):
    fig, axes = figure("Per-defect T1 − T0 differences", summary, panels=2,
                       height=max(4.9, 3.1 + .31 * len(pairs)),
                       footnote="Open circles indicate measured zero. A dash marks unavailable data.")
    fig.subplots_adjust(left=.23, wspace=.47)
    for ax, key, label in zip(axes, ("delta_logical_tokens", "delta_MRR"),
                              ("Δ logical tokens", "Δ MRR"), strict=True):
        if not pairs:
            empty(ax, "No scheduled pairs")
            continue
        values = [row[key] for row in pairs]
        largest = max((abs(value) for value in values if value is not None), default=0)
        limit = max(largest * 1.2, 0.1 if key == "delta_MRR" else 1)
        for i, value in enumerate(values):
            if value is None:
                ax.text(0, i, "—", color="#64748B", ha="center", va="center")
            else:
                color = COLORS[value >= 0]
                ax.hlines(i, 0, value, color=color, linewidth=2)
                ax.scatter(value, i, color=color if value else "white", edgecolor=color, zorder=3)
        ax.set_yticks(range(len(pairs)), [row["instance_id"] for row in pairs] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color="#64748B", linewidth=.8)
        ax.set_xlabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=key == "delta_logical_tokens"))
        ax.grid(axis="x")
    return fig


def evidence_selection(summary, evidence):
    fig, (overlap, size) = figure("Evidence selection diagnostics", summary, panels=2,
                                footnote="Planned target pairs with both selections. Packing units use the recorded counter.")
    groups = defaultdict(list)
    for row in evidence:
        if row["jaccard"] is not None and row["delta_input_units"] is not None:
            groups[row["counter"]].append(row)
    counts = [row["jaccard"] for group in groups.values() for row in group]
    if not counts:
        empty(overlap, "No paired selections")
        empty(size, "No paired selections")
        return fig
    overlap.hist(counts, bins=np.linspace(0, 1, 11), color=COLORS[0], edgecolor="white")
    overlap.set_xlim(0, 1)
    overlap.set_xlabel("Jaccard overlap")
    overlap.set_ylabel("Planned targets")
    overlap.yaxis.set_major_locator(MaxNLocator(integer=True))
    overlap.grid(axis="y")
    if len(groups) > 1:
        empty(size, "Multiple counters\nSee the evidence table by counter")
    else:
        counter, group = next(iter(groups.items()))
        values = [row["delta_input_units"] for row in group]
        size.hist(values, bins=min(10, max(1, len(set(values)))), color=COLORS[1], edgecolor="white")
        size.axvline(0, color="#64748B", linewidth=.8)
        size.set_xlabel("T1 − T0 input units")
        size.set_title("UTF-8 byte proxy" if counter.startswith("utf8_") else "Local template tokens")
        size.yaxis.set_major_locator(MaxNLocator(integer=True))
        size.grid(axis="y")
    return fig


def render_figures(output, summary, pairs, evidence):
    output.mkdir()
    cohort = summary["cohorts"]["matched_complete_labeled"]
    with mpl.rc_context(STYLE):
        figures = {
            "accuracy": accuracy(summary, cohort),
            "token_cost": token_cost(summary),
            "pareto": pareto(summary, cohort),
            "paired_differences": paired_differences(summary, pairs),
            "evidence_selection": evidence_selection(summary, evidence),
        }
        for name, fig in figures.items():
            fig.savefig(output / f"{name}.png", dpi=300, metadata={"Software": "GRAFT fixed-evidence report"})
            fig.savefig(output / f"{name}.svg", metadata={"Date": None, "Creator": "GRAFT fixed-evidence report"})
            fig.clear()

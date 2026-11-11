"""AITECH autonomous research *runner*, CPU baseline and paper discovery.

Research Compass's scheduled agent chooses and analyzes new hypotheses.
This runner executes an honest, deterministic, offline evidence-ablation sweep
over an archived real SymPy trace. It neither calls a paid LLM nor claims
SWE-bench localization accuracy. Fresh paper metadata is optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .evidence import select_evidence
from .sympy_trace_pilot import DEFAULT_TRACE, DEPENDENCIES, extract

BUDGET_PCTS = (50, 65, 75, 85, 100)
QUERY = '("fault localization" OR "software debugging") AND ("retrieval" OR "graph")'
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def discover_papers(max_papers: int = 3, timeout: int = 8) -> dict:
    """Retrieve *metadata only*, with clear failure status and a bounded call."""
    params = urllib.parse.urlencode({
        "search_query": 'all:"fault localization" AND (all:"retrieval" OR all:"graph")',
        "start": 0, "max_results": max_papers,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    url = "https://export.arxiv.org/api/query?" + params
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AITECH-Research-Agent/0.1 (academic metadata)"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            xml = response.read(150_000)
        root = ET.fromstring(xml)
        papers = []
        for entry in root.findall("a:entry", ARXIV_NS):
            def value(tag: str) -> str:
                node = entry.find("a:" + tag, ARXIV_NS)
                return " ".join(node.text.split()) if node is not None and node.text else ""
            papers.append({"title": value("title"),
                           "url": value("id"), "published": value("published")})
        return {"status": "retrieved", "query_url": url, "papers": papers,
                "interpretation": "Metadata only, papers were not read or verified."}
    except (OSError, ET.ParseError, ValueError) as error:
        return {"status": "unavailable", "query_url": url, "papers": [],
                "error_type": type(error).__name__,
                "interpretation": "Offline experiment remains valid; literature was not retrieved."}


def run_ablation(trace: str) -> dict:
    """Run matched-budget selectors on seven archived *real-source* lines."""
    evidence = extract(trace)
    raw = sum(item.tokens for item in evidence)
    rows = []
    for percent in BUDGET_PCTS:
        budget = raw * percent // 100
        for name, preserve in (
            ("relevance_only", False), ("dependency_aware", True),
        ):
            result = select_evidence(evidence, DEPENDENCIES, budget,
                                     preserve_dependencies=preserve)
            rows.append({
                "budget_pct": percent, "budget_proxy": budget,
                "method": name, "tokens_proxy": result.tokens,
                "selected_ids": list(result.ids),
                "evidence_coverage": len(result.ids) / len(evidence),
                "assertion_retained": "test_assertion" in result.ids,
                "known_dangling": len(result.dangling),
                "dangling_ids": list(result.dangling),
            })
    return {
        "scope": "Archived SymPy bug, seven hand-selected real-source lines",
        "instance": "sympy__sympy-20590",
        "measured": "Deterministic evidence selection with regex proxy tokens",
        "not_measured": [
            "LLM fault localization accuracy", "full repository call-graph accuracy",
            "inference tokens", "end-to-end debugging latency",
        ],
        "raw_tokens_proxy": raw,
        "source_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
        "gold_used_for_selection": False,
        "score_source": "Fixed manual relevance weights, not trained ML",
        "dependencies": "Manually verified source-level links, not complete causal graph",
        "rows": rows,
    }


def choose_next_action(rows: list[dict]) -> str:
    """Evidence-driven next step, not automatic claims of improved reasoning."""
    for r in rows:
        if r["method"] == "dependency_aware" and r["known_dangling"] == 0 and not r["assertion_retained"]:
            return ("Test a risk-aware anchor-preservation policy: zero dangling "
                    "does not imply retention of the failure assertion. Compare "
                    "root-evidence recall and dependency closure on independent bugs.")
    return ("Build AST-derived support edges on separate bug instances, "
            "then test downstream localization under identical model token budgets.")


def render_report(result: dict, papers: dict) -> str:
    rows = result["rows"]
    report = [
        "# AITECH autonomous CPU evidence-ablation report",
        "",
        f"Bug: `{result['instance']}`", "",
        "**Scope:** one seven-line real-source microcase, not independent SWE-bench evaluation.",
        "**Method:** fixed manual relevance scores and a regex-token proxy.",
        "**No LLM was run.** Gold was not used to select evidence.", "",
        "| Budget | Selector | Tokens | Evidence items kept | Assertion kept | Known dangling |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['budget_pct']}% | {r['method']} | {r['tokens_proxy']} | "
            f"{len(r['selected_ids'])} | {'Yes' if r['assertion_retained'] else 'No'} | "
            f"{r['known_dangling']} |"
        )
    report += [
        "", "## Interpretation",
        "Removing all important evidence can produce zero dangling links. "
        "Report assertion retention and evidence coverage together with known violations.",
        "The evidence lines and dependency graph were curated in advance, "
        "so this experiment tests selector mechanics rather than automatic debugging.",
        "", "## Next hypothesis", choose_next_action(rows), "",
        "## Research discovery",
        f"Metadata fetch status: **{papers['status']}**. "
        "A retrieved paper title is only a candidate to read, not verified evidence.",
    ]
    for p in papers["papers"]:
        report.append(f"- {p['title']} ({p['published']}): {p['url']}")
    report += ["", "## Reproducibility",
               f"Archived trace SHA-256: `{result['source_sha256']}`",
               "Reproduce with `python -m aitech.research_agent --no-search --output-dir /tmp/aitech-agent`.",
               "", "## Limitations",
               "Single previously inspected bug; no held-out data; only known manually supplied links; "
               "no downstream LLM, graph construction, actual tokenizer, or patch validation.",
               ""]
    return "\n".join(report)


def run_agent(trace_path: Path, output_dir: Path, *, search: bool = True) -> dict:
    trace = trace_path.read_text(encoding="utf-8")
    result = run_ablation(trace)
    papers = discover_papers() if search else {
        "status": "disabled", "papers": [], "interpretation": "Offline mode"}
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    record = {"executed_at_utc": now, "evidence": result, "discovery": papers,
              "next_action": choose_next_action(result["rows"])}
    (output_dir / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(result, papers), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aitech/agent_latest"))
    parser.add_argument("--no-search", action="store_true",
                        help="Skip optional arXiv metadata search, useful for offline CI")
    args = parser.parse_args()
    record = run_agent(args.trace, args.output_dir, search=not args.no_search)
    print(json.dumps({
        "status": "completed", "rows": len(record["evidence"]["rows"]),
        "literature": record["discovery"]["status"],
        "next_action": record["next_action"],
        "report": str(args.output_dir / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

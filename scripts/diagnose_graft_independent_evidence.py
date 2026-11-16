"""Diagnose local FlexFL evidence availability for a frozen Defects4J manifest.

Does not download, generate, or substitute failure evidence. Does not invoke
Qwen, Defects4J, or any ground-truth reader.

Example:
  python -m scripts.diagnose_graft_independent_evidence \
      --manifest configs/rave_independent_smoke.json \
      --output results/graft_independent_evidence_diagnosis.json
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from wp1.flexfl_data import flexfl_root

CATEGORIES = ("bug_reports", "trigger_tests")


def possible_paths(repo: Path, bug: str, category: str) -> list[Path]:
    base = flexfl_root(repo) / "data" / "input" / category
    return [
        base / "Defects4J" / f"{bug}.txt",
        base / f"{bug}.txt",
        base / "Defects4J" / bug,
        base / bug,
    ]


def _source_record(path: Path) -> dict:
    present = path.is_file()
    content = path.read_bytes() if present else b""
    return {
        "path": str(path),
        "exists": present,
        "nonempty": bool(content.strip()),
        "size_bytes": len(content),
        "sha256": sha256(content).hexdigest() if present else None,
    }


def inspect(manifest: Path) -> dict:
    manifest = manifest.resolve()
    value = json.loads(manifest.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("Expected frozen Defects4J manifest with 'cases'")
    base = Path.cwd() if value.get("path_base") == "cwd" or "instances" not in value else manifest.parent
    bugs = []
    for row in value["cases"]:
        bug = row.get("bug")
        if not isinstance(bug, str) or not bug:
            raise ValueError("Missing bug ID")
        repo = (base / row.get("flexfl_repo", "")).resolve()
        sources = {}
        for kind in CATEGORIES:
            candidates = possible_paths(repo, bug, kind)
            sources[kind] = {
                "candidates": [_source_record(p) for p in candidates],
                "first_existing": next((str(p) for p in candidates if p.is_file()), None),
            }
        evidence_file = row.get("evidence_file", "")
        explicit = _source_record((base / evidence_file).resolve()) if evidence_file else None
        # Adapter falls back when the selected source's content is empty,
        # and historically reads only the *first existing* FlexFL file.
        def chosen_source(kind):
            entries = sources[kind]["candidates"]
            return next((r for r in entries if r["exists"]), None)

        chosen = [chosen_source(k) for k in CATEGORIES]
        flexfl_ok = any(v and v["nonempty"] for v in chosen)
        ready_evidence = bool(
            (explicit and explicit["nonempty"])
            or (row.get("evidence", {}).get("text") if isinstance(row.get("evidence"), dict) else False)
            or row.get("issue")
            or flexfl_ok
        )
        checkout = (base / row.get("checkout", "")).resolve() if row.get("checkout") else None
        bugs.append({
            "bug": bug,
            "flexfl_repo": str(repo),
            "flexfl_root_exists": flexfl_root(repo).is_dir(),
            "checkout": str(checkout) if checkout else None,
            "checkout_exists": checkout.is_dir() if checkout else False,
            "sources": sources,
            "explicit_evidence_file": explicit,
            "evidence_available_to_current_adapter": ready_evidence,
            "diagnosis": (
                "ready_to_load_evidence" if ready_evidence
                else "missing_or_empty_flexfl_bug_report_and_trigger_test"
            ),
        })
    return {
        "schema": "graft-evidence-availability-v1",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "bug_count": len(bugs),
        "all_have_evidence": all(x["evidence_available_to_current_adapter"] for x in bugs),
        "new_qwen_calls": 0,
        "external_tests_performed": False,
        "uses_gold_labels": False,
        "bugs": bugs,
        "note": (
            "This checks evidence files only. Passing this step does not prove "
            "checkout revisions, FlexFL candidates, or model readiness."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/rave_independent_smoke.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = inspect(args.manifest)
    for item in report["bugs"]:
        print(f"{item['bug']:8} evidence={item['diagnosis']} checkout_exists={item['checkout_exists']}")
        for kind in CATEGORIES:
            entries = item["sources"][kind]["candidates"]
            selected = next((x for x in entries if x["exists"]), None)
            print(f"  {kind}: {selected['path'] if selected else 'NO FILE'}"
                  + (f" ({selected['size_bytes']} bytes)" if selected else ""))
        if item["explicit_evidence_file"]:
            print("  explicit:", item["explicit_evidence_file"]["path"])
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("Saved:", args.output)
    if not report["all_have_evidence"]:
        print("STOP: Missing input evidence. Do not start model inference or invent logs.")
    else:
        print("Evidence files exist. Run strict GRAFT preflight next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

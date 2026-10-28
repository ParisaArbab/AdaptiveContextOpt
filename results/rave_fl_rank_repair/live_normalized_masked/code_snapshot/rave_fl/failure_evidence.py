"""Build concise, observed-only failure evidence from Defects4J failing_tests.

Keeps test names, assertion details and a few project stack frames while
dropping repetitive JUnit/Ant infrastructure frames. Never reads gold labels.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def compress_failures(raw: str, project_prefix: str = "org.joda.time", max_frames: int = 3) -> str:
    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative")
    sections = re.split(r"(?m)^---\s+", raw)[1:]
    if not sections:
        raise ValueError("No Defects4J failure sections found")
    results = []
    for section in sections:
        lines = section.splitlines()
        if not lines or not lines[0].strip():
            raise ValueError("Empty test name in Defects4J failure section")
        test_name = lines[0].strip()
        # Match the initial Time-25 extractor exactly; do not infer expected
        # outputs, method names or gold faults from source patches.
        errors = [line.strip() for line in lines if "ComparisonFailure:" in line]
        if not errors:
            # Do not silently claim a valid summary for unsupported failures.
            raise ValueError(f"No ComparisonFailure found for {test_name}")
        frames = [line.strip() for line in lines
                  if line.strip().startswith(f"at {project_prefix}.")]
        results.append(
            f"TEST: {test_name}\n"
            f"FAILURE: {errors[0]}\n"
            "PROJECT FRAMES:\n" + "\n".join(frames[:max_frames])
        )
    return "\n\n".join(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failing-tests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-prefix", default="org.joda.time")
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    result = compress_failures(args.failing_tests.read_text(encoding="utf-8"),
                               args.project_prefix, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    print(f"Parsed {result.count('TEST: ')} failing tests; "
          f"{len(result)} characters; saved {args.output}")


if __name__ == "__main__":
    main()

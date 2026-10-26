"""Create tiny synthetic, committed Java/Python sources; never fetch a benchmark."""
import argparse
import json
from pathlib import Path
import subprocess

from rave_fl.eval.schema import CompatibilityError


def create(directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise CompatibilityError("Fixture directory exists; choose a new path")
    repo = directory / "checkout"
    repo.mkdir(parents=True)
    (repo / "example.py").write_text(
        "def calculate(value):\n    if value < 0:\n        raise ValueError(value)\n    return value + 1\n\ndef helper(value):\n    return value * 2\n"
    )
    (repo / "Example.java").write_text(
        "class Example {\n    int calculate(int value) {\n        if (value < 0) throw new IllegalArgumentException();\n        return value + 1;\n    }\n    int helper(int value) { return value * 2; }\n}\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Synthetic Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Immutable synthetic sources",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    rows = []
    for language, path, function in [
        ("python", "example.py", "example.py::calculate"),
        ("java", "Example.java", "Example.java::Example.calculate(int value)"),
    ]:
        rows.append(
            {
                "instance_id": "synthetic-" + language,
                "repository": "synthetic/" + language,
                "language": language,
                "revision": revision,
                "checkout": "checkout",
                "issue": "calculate rejects negative input; investigate the behavior",
                "evidence": {
                    "kind": "observed_failure",
                    "text": "Synthetic failing test: calculate(-1) raises unexpectedly",
                    "source": "fixture",
                },
                "gold": {
                    "files": [path],
                    "functions": [function],
                    "semantics": "synthetic fixture labels; not a measured benchmark",
                },
            }
        )
    target = directory / "instances.json"
    target.write_text(json.dumps({"version": "synthetic-v1", "instances": rows}, indent=2) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    print(create(parser.parse_args().output_dir))

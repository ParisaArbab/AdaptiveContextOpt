"""Run with: python -m rave_fl.demo"""
import json
from .core import Action, Candidate, Evidence, RaveFL


def mock_verifier(sketches, action):
    # Demonstration only: replace with a real LLM verification adapter.
    return [
        Evidence(s.function_id, 0.6 if "parse" in s.function_id else 0.2,
                 "Illustrative synthetic verdict", 2, 0.4, action)
        for s in sketches
    ]


def main():
    candidates = [
        Candidate("parse_config", "def parse_config(path):", "return config[path]", .9, .8, ("read_file",), 3),
        Candidate("render_ui", "def render_ui(data):", "return str(data)", .3, .2, (), 1),
        Candidate("read_file", "def read_file(path):", "return open(path).read()", .5, .4, (), 2),
    ]
    result = RaveFL(candidates, token_budget=250, verifier=mock_verifier).execute()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

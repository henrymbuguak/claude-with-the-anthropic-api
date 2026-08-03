"""Compare eval results between two system prompt versions on the same case set.

Usage:
    uv run eval/compare.py prompts/scientist_v1.txt prompts/scientist_v2.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import DEFAULT_CASES_PATH, print_summary, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two system prompts on the same eval cases.")
    parser.add_argument("prompt_a", type=Path)
    parser.add_argument("prompt_b", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()

    for prompt_path in (args.prompt_a, args.prompt_b):
        results = run(prompt_path, args.cases)
        print_summary(prompt_path, results)


if __name__ == "__main__":
    main()

"""Validate one guide-generation request and export its plan metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_planned_guide(plan_path: Path, guide_id: int) -> dict:
    """Return one offline planned guide or raise a clear validation error."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    guide = next(
        (entry for entry in plan.get("guides", []) if entry.get("id") == guide_id),
        None,
    )
    if guide is None:
        raise ValueError(f"guide {guide_id} does not exist")
    if guide.get("status") != "planned":
        raise ValueError(f"guide {guide_id} is not planned")
    if guide.get("tier") != "offline":
        raise ValueError(f"guide {guide_id} is not offline")
    output = Path(guide["output"])
    if output.parent != Path("docs/guides"):
        raise ValueError("guide output must be under docs/guides")
    if output.stem.startswith("guide-"):
        raise ValueError("guide output must use a descriptive filename")
    if output.exists():
        raise ValueError(f"guide output already exists: {output}")
    return guide


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("guides/plan.json"))
    parser.add_argument("--guide-id", type=int, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        guide = select_planned_guide(args.plan, args.guide_id)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    metadata = {
        "branch_prefix": f"guide/{guide['id']}-",
        "guide_id": str(guide["id"]),
        "output": guide["output"],
        "title": guide["title"],
    }
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output_file:
            for key, value in metadata.items():
                output_file.write(f"{key}={value}\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

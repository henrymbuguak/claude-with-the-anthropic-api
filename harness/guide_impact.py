"""Report published guides whose declared implementation files changed."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class GuideImpact:
    """One published guide whose declared implementation changed."""

    guide_id: int
    output: str
    title: str
    changed_paths: tuple[str, ...]


def _normalise(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def impacted_guides(plan_path: Path, changed_paths: list[str]) -> tuple[GuideImpact, ...]:
    """Return published guides affected by changed declared primary-code files."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    changed = {_normalise(path) for path in changed_paths if path.strip()}
    impacts: list[GuideImpact] = []

    for guide in plan.get("guides", []):
        if guide.get("status") != "published":
            continue
        primary_code = {_normalise(path) for path in guide.get("primary_code", [])}
        matched = tuple(sorted(changed & primary_code))
        if matched:
            impacts.append(
                GuideImpact(
                    guide_id=guide["id"],
                    output=guide["output"],
                    title=guide["title"],
                    changed_paths=matched,
                )
            )
    return tuple(impacts)


def render_summary(impacts: tuple[GuideImpact, ...]) -> str:
    """Render an advisory Markdown summary suitable for GitHub Actions."""
    if not impacts:
        return "## Documentation impact\n\nNo published guides declare changed primary code.\n"

    rows = [
        "## Documentation impact",
        "",
        "Review these published guides because their declared primary code changed.",
        "This advisory does not block the change.",
        "",
        "| Guide | Changed primary code |",
        "| --- | --- |",
    ]
    for impact in impacts:
        paths = "<br>".join(f"`{path}`" for path in impact.changed_paths)
        rows.append(f"| [{impact.title}]({impact.output}) | {paths} |")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("changed_paths", nargs="*", help="Repository-relative changed paths")
    parser.add_argument("--plan", type=Path, default=Path("guides/plan.json"))
    parser.add_argument("--summary", type=Path, help="Write Markdown to this path")
    args = parser.parse_args()

    summary = render_summary(impacted_guides(args.plan, args.changed_paths))
    if args.summary is None:
        print(summary, end="")
    else:
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
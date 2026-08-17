"""Verify curriculum prerequisites, guide links, and MkDocs learning order."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_NAV_GUIDE = re.compile(r"^\s{6}-\s+.+:\s+(?P<path>\S+\.md)\s*$")
_REDIRECT = re.compile(r"^\s{8}(?P<source>\S+\.md):\s+(?P<target>\S+\.md)\s*$")
_AVAILABLE_STATUSES = {"draft", "published"}


@dataclass(frozen=True)
class FlowIssue:
    """One curriculum or article-flow problem."""

    location: str
    message: str


def _section(text: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group("body") if match else None


def _local_targets(text: str, source: Path, root: Path) -> list[Path]:
    targets: list[Path] = []
    for match in _LINK.finditer(text):
        raw = match.group("target").strip()
        if not raw or raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_text = raw.split("#", 1)[0].split("?", 1)[0]
        if not path_text:
            continue
        target = (
            root / path_text.lstrip("/")
            if path_text.startswith("/")
            else source.parent / path_text
        )
        targets.append(target.resolve())
    return targets


def _guide_nav_paths(mkdocs_text: str) -> list[str]:
    lines = mkdocs_text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line == "  - Guides:")
    except StopIteration:
        return []
    paths: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("  - "):
            break
        match = _NAV_GUIDE.match(line)
        if match:
            paths.append(match.group("path"))
    return paths


def _redirect_maps(mkdocs_text: str) -> dict[str, str]:
    return {
        match.group("source"): match.group("target")
        for line in mkdocs_text.splitlines()
        if (match := _REDIRECT.match(line))
    }


def verify_guide_flow(
    root: Path,
    *,
    plan_path: Path = Path("guides/plan.json"),
    mkdocs_path: Path = Path("mkdocs.yml"),
) -> tuple[FlowIssue, ...]:
    """Return all prerequisite, link, section, and navigation-order issues."""
    root = root.resolve()
    issues: list[FlowIssue] = []
    plan = json.loads((root / plan_path).read_text(encoding="utf-8"))
    guides = plan.get("guides", [])
    guide_by_id = {guide.get("id"): guide for guide in guides}
    positions = {guide.get("id"): index for index, guide in enumerate(guides)}

    for guide in guides:
        guide_id = guide.get("id")
        location = f"{plan_path}:guide-{guide_id}"
        for prerequisite in guide.get("prerequisites", []):
            if prerequisite not in guide_by_id:
                issues.append(FlowIssue(location, f"unknown prerequisite: {prerequisite}"))
            elif positions[prerequisite] >= positions[guide_id]:
                issues.append(
                    FlowIssue(location, f"prerequisite {prerequisite} must appear earlier")
                )

    available = [
        guide
        for guide in guides
        if guide.get("status") in _AVAILABLE_STATUSES
        and (root / guide.get("output", "")).is_file()
    ]
    expected_nav = [Path(guide["output"]).relative_to("docs").as_posix() for guide in available]
    mkdocs_text = (root / mkdocs_path).read_text(encoding="utf-8")
    actual_nav = _guide_nav_paths(mkdocs_text)
    if actual_nav != expected_nav:
        issues.append(
            FlowIssue(
                str(mkdocs_path),
                f"guide navigation order is {actual_nav}, expected {expected_nav}",
            )
        )

    declared_redirects = {
        redirect: Path(guide["output"]).relative_to("docs").as_posix()
        for guide in guides
        for redirect in guide.get("redirects", [])
    }
    actual_redirects = _redirect_maps(mkdocs_text)
    if actual_redirects != declared_redirects:
        issues.append(
            FlowIssue(
                str(mkdocs_path),
                f"redirect maps are {actual_redirects}, expected {declared_redirects}",
            )
        )
    for redirect in declared_redirects:
        if (root / "docs" / redirect).exists():
            issues.append(
                FlowIssue(str(plan_path), f"redirect source still exists: {redirect}")
            )

    for available_index, guide in enumerate(available):
        output = Path(guide["output"])
        path = root / output
        text = path.read_text(encoding="utf-8")
        location = output.as_posix()
        if output.stem.startswith("guide-"):
            issues.append(FlowIssue(location, "guide must use a descriptive filename"))
        expected_title = f"# {guide['title']}"
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if first_line != expected_title:
            issues.append(
                FlowIssue(location, f"H1 is {first_line!r}, expected {expected_title!r}")
            )

        before = _section(text, "Before you begin")
        next_steps = _section(text, "Next steps")
        if before is None:
            issues.append(FlowIssue(location, "missing Before you begin section"))
        if next_steps is None:
            issues.append(FlowIssue(location, "missing Next steps section"))

        for target in _local_targets(text, path, root):
            if not target.exists():
                issues.append(
                    FlowIssue(location, f"broken local link: {target.as_posix()}")
                )

        before_targets = set(_local_targets(before or "", path, root))
        for prerequisite_id in guide.get("prerequisites", []):
            prerequisite = guide_by_id.get(prerequisite_id)
            if prerequisite is None:
                continue
            prerequisite_path = (root / prerequisite["output"]).resolve()
            if not prerequisite_path.is_file():
                issues.append(
                    FlowIssue(
                        location,
                        f"prerequisite guide {prerequisite_id} is not available",
                    )
                )
            elif prerequisite_path not in before_targets:
                issues.append(
                    FlowIssue(
                        location,
                        f"Before you begin must link prerequisite guide {prerequisite_id}",
                    )
                )
        if available_index + 1 < len(available):
            next_guide = available[available_index + 1]
            next_path = (root / next_guide["output"]).resolve()
            next_targets = set(_local_targets(next_steps or "", path, root))
            if next_path not in next_targets:
                issues.append(
                    FlowIssue(
                        location,
                        f"Next steps must link next guide {next_guide['id']}",
                    )
                )
    return tuple(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    issues = verify_guide_flow(args.root)
    for issue in issues:
        print(f"FAIL {issue.location}: {issue.message}")
    if issues:
        print(f"Guide flow verification failed: {len(issues)} issue(s)")
        return 1
    print("Guide flow verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

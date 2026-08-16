"""Verify the curriculum, measured ledger, and facts displayed in guides."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from harness.measure_facts import measure_facts, render_facts

_FACT = re.compile(
    r"<!--\s*fact\s+(?P<key>[a-zA-Z0-9_.-]+)\s*-->"
    r"(?P<value>.*?)"
    r"<!--\s*/fact\s*-->",
    re.DOTALL,
)
_REQUIRED_GUIDE_FIELDS = {
    "build_from_empty",
    "concepts",
    "facts",
    "id",
    "output",
    "prerequisites",
    "primary_code",
    "status",
    "tests",
    "tier",
    "title",
    "type",
}


@dataclass(frozen=True)
class FactIssue:
    """One curriculum, ledger, or displayed-fact problem."""

    location: str
    message: str


def flatten_facts(value, prefix: str = "") -> dict[str, object]:
    """Flatten nested dictionaries into dotted fact paths."""
    flattened: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten_facts(child, path))
    else:
        flattened[prefix] = value
    return flattened


def display_value(value: object) -> str:
    """Render a scalar fact exactly as guide prose should display it."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError("only scalar facts can be displayed inline")


def verify_repository(
    root: Path,
    *,
    plan_path: Path,
    ledger_path: Path,
) -> tuple[FactIssue, ...]:
    """Return all curriculum, measured-ledger, and guide-reference issues."""
    root = root.resolve()
    issues: list[FactIssue] = []
    plan = json.loads((root / plan_path).read_text(encoding="utf-8"))
    ledger_file = root / ledger_path
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    measured = measure_facts(root)
    if render_facts(ledger) != render_facts(measured):
        issues.append(FactIssue(str(ledger_path), "committed ledger differs from measurement"))

    flattened = flatten_facts(ledger)
    guides = plan.get("guides")
    if not isinstance(guides, list):
        return tuple(issues + [FactIssue(str(plan_path), "guides must be a list")])

    guide_ids = [guide.get("id") for guide in guides if isinstance(guide, dict)]
    if len(guide_ids) != len(set(guide_ids)):
        issues.append(FactIssue(str(plan_path), "guide ids must be unique"))
    known_ids = set(guide_ids)

    for guide in guides:
        if not isinstance(guide, dict):
            issues.append(FactIssue(str(plan_path), "each guide must be an object"))
            continue
        guide_id = guide.get("id", "unknown")
        location = f"{plan_path}:guide-{guide_id}"
        missing_fields = sorted(_REQUIRED_GUIDE_FIELDS - guide.keys())
        if missing_fields:
            issues.append(
                FactIssue(location, f"missing fields: {', '.join(missing_fields)}")
            )
            continue
        if guide["tier"] not in {"offline", "keyed"}:
            issues.append(FactIssue(location, "tier must be offline or keyed"))
        if guide["type"] not in {"tutorial", "how-to", "explanation", "reference"}:
            issues.append(FactIssue(location, "unsupported documentation type"))
        if guide["status"] not in {"planned", "draft", "published"}:
            issues.append(FactIssue(location, "unsupported guide status"))
        if len(guide["concepts"]) > plan.get("series", {}).get("concept_budget", 3):
            issues.append(FactIssue(location, "guide exceeds the concept budget"))
        unknown_prerequisites = set(guide["prerequisites"]) - known_ids
        if unknown_prerequisites:
            issues.append(
                FactIssue(location, f"unknown prerequisites: {sorted(unknown_prerequisites)}")
            )
        unknown_facts = set(guide["facts"]) - flattened.keys()
        if unknown_facts:
            issues.append(FactIssue(location, f"unknown facts: {sorted(unknown_facts)}"))

        guide_path = root / guide["output"]
        if not guide_path.exists():
            if guide["status"] != "planned":
                issues.append(FactIssue(location, f"missing output: {guide['output']}"))
            continue

        text = guide_path.read_text(encoding="utf-8")
        references = list(_FACT.finditer(text))
        used_facts = {reference.group("key") for reference in references}
        declared_facts = set(guide["facts"])
        if used_facts - declared_facts:
            issues.append(
                FactIssue(
                    guide["output"],
                    f"undeclared facts: {sorted(used_facts - declared_facts)}",
                )
            )
        if declared_facts - used_facts:
            issues.append(
                FactIssue(
                    guide["output"],
                    f"declared facts not displayed: {sorted(declared_facts - used_facts)}",
                )
            )
        for reference in references:
            key = reference.group("key")
            if key not in flattened:
                issues.append(FactIssue(guide["output"], f"unknown fact reference: {key}"))
                continue
            try:
                expected = display_value(flattened[key])
            except TypeError as exc:
                issues.append(FactIssue(guide["output"], f"{key}: {exc}"))
                continue
            actual = reference.group("value").strip()
            if actual != expected:
                issues.append(
                    FactIssue(
                        guide["output"],
                        f"{key} displays {actual!r}, expected {expected!r}",
                    )
                )
    return tuple(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, default=Path("guides/plan.json"))
    parser.add_argument("--ledger", type=Path, default=Path("docs/facts.json"))
    args = parser.parse_args()

    issues = verify_repository(
        args.root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    for issue in issues:
        print(f"FAIL {issue.location}: {issue.message}")
    if issues:
        print(f"Fact verification failed: {len(issues)} issue(s)")
        return 1
    print("Fact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

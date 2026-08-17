"""Validate the constrained output of one generated guide branch."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.verify_facts import verify_repository
from harness.verify_guide_flow import verify_guide_flow
from harness.verify_guides import lint_guide

_PLAN_PATH = Path("guides/plan.json")
_LEDGER_PATH = Path("docs/facts.json")


@dataclass(frozen=True)
class GenerationIssue:
    """One generated-branch policy violation."""

    message: str


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        shell=False,
        check=False,
    )


def _guide_by_id(plan: dict, guide_id: int) -> dict | None:
    return next(
        (guide for guide in plan.get("guides", []) if guide.get("id") == guide_id),
        None,
    )


def validate_generation(
    root: Path,
    *,
    guide_id: int,
    base_revision: str,
) -> tuple[GenerationIssue, ...]:
    """Return policy violations for a generated guide candidate."""
    root = root.resolve()
    issues: list[GenerationIssue] = []
    base_plan_process = _git(root, "show", f"{base_revision}:{_PLAN_PATH.as_posix()}")
    if base_plan_process.returncode != 0:
        return (GenerationIssue("could not read the base curriculum"),)

    try:
        base_plan = json.loads(base_plan_process.stdout)
        current_plan = json.loads((root / _PLAN_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (GenerationIssue(f"could not parse the curriculum: {exc}"),)

    base_guide = _guide_by_id(base_plan, guide_id)
    current_guide = _guide_by_id(current_plan, guide_id)
    if base_guide is None:
        return (GenerationIssue(f"guide {guide_id} does not exist in the base plan"),)
    if base_guide.get("status") != "planned":
        issues.append(GenerationIssue(f"guide {guide_id} is not planned"))
    if current_guide is None:
        return tuple(issues + [GenerationIssue(f"guide {guide_id} was removed")])

    expected_guide = {**base_guide, "status": "draft"}
    if current_guide != expected_guide:
        issues.append(
            GenerationIssue("the selected plan entry may only change status to draft")
        )
    base_other = [guide for guide in base_plan["guides"] if guide.get("id") != guide_id]
    current_other = [
        guide for guide in current_plan.get("guides", []) if guide.get("id") != guide_id
    ]
    if current_other != base_other:
        issues.append(GenerationIssue("other curriculum entries changed"))
    if current_plan.get("series") != base_plan.get("series"):
        issues.append(GenerationIssue("series metadata changed"))

    output = Path(base_guide["output"])
    if output.parent != Path("docs/guides"):
        issues.append(GenerationIssue("guide output must be under docs/guides"))
    if output.stem.startswith("guide-"):
        issues.append(GenerationIssue("guide output must use a descriptive filename"))
    if not (root / output).is_file():
        issues.append(GenerationIssue(f"generated output is missing: {output.as_posix()}"))

    changed_process = _git(root, "diff", "--name-only", f"{base_revision}...HEAD")
    if changed_process.returncode != 0:
        issues.append(GenerationIssue("could not inspect generated changes"))
    else:
        changed = {Path(line) for line in changed_process.stdout.splitlines() if line}
        allowed = {output, Path("mkdocs.yml"), _PLAN_PATH, _LEDGER_PATH}
        unexpected = sorted(path.as_posix() for path in changed - allowed)
        if unexpected:
            issues.append(
                GenerationIssue(f"generated branch changed forbidden paths: {unexpected}")
            )
        required = {output, Path("mkdocs.yml"), _PLAN_PATH}
        missing = sorted(path.as_posix() for path in required - changed)
        if missing:
            issues.append(
                GenerationIssue(f"generated branch did not change required paths: {missing}")
            )

    navigation = (root / "mkdocs.yml").read_text(encoding="utf-8")
    docs_relative = output.relative_to("docs").as_posix()
    if docs_relative not in navigation:
        issues.append(GenerationIssue("generated guide is missing from MkDocs navigation"))

    if (root / output).exists():
        coverage = lint_guide(root / output)
        issues.extend(
            GenerationIssue(f"guide line {issue.line}: {issue.message}")
            for issue in coverage.issues
        )
    fact_issues = verify_repository(
        root,
        plan_path=_PLAN_PATH,
        ledger_path=_LEDGER_PATH,
    )
    issues.extend(
        GenerationIssue(f"{issue.location}: {issue.message}") for issue in fact_issues
    )
    flow_issues = verify_guide_flow(root)
    issues.extend(
        GenerationIssue(f"{issue.location}: {issue.message}") for issue in flow_issues
    )
    return tuple(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--guide-id", type=int, required=True)
    parser.add_argument("--base", required=True, help="Base revision to compare")
    args = parser.parse_args()

    issues = validate_generation(
        args.root,
        guide_id=args.guide_id,
        base_revision=args.base,
    )
    for issue in issues:
        print(f"FAIL {issue.message}")
    if issues:
        print(f"Generation policy failed: {len(issues)} issue(s)")
        return 1
    print(f"Generation policy passed for guide {args.guide_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lint guide verification annotations without executing their commands."""

from __future__ import annotations

import argparse
import glob
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

_ANNOTATION = re.compile(r"^\s*<!--\s*verify\s+(cmd|expect|manual)\b(.*?)-->\s*$")
_FENCE = re.compile(r"^\s*```([A-Za-z0-9_-]*)\s*$")
_SHELL_LANGUAGES = {"bash", "console", "powershell", "pwsh", "sh", "shell"}
_TIERS = {"offline", "keyed"}
_MATCH_MODES = {"contains", "exact", "ordered"}
_ORDERED_ITEM = re.compile(r"^\d+\.\s")
_UNDER_INDENTED_CONTINUATION = re.compile(r"^ {1,3}\S")


@dataclass(frozen=True)
class Issue:
    """One guide annotation problem."""

    line: int
    message: str


@dataclass(frozen=True)
class Coverage:
    """Verification coverage for one guide."""

    commands: int
    verified: int
    manual: int
    issues: tuple[Issue, ...]

    @property
    def unaccounted(self) -> int:
        return sum("has no verify annotation" in issue.message for issue in self.issues)


def _parse_attributes(raw: str, line: int) -> tuple[dict[str, str], list[Issue]]:
    attributes: dict[str, str] = {}
    issues: list[Issue] = []
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as exc:
        return {}, [Issue(line, f"invalid annotation quoting: {exc}")]

    for token in tokens:
        if "=" not in token:
            issues.append(Issue(line, f"attribute must use name=value: {token!r}"))
            continue
        name, value = token.split("=", 1)
        if name in attributes:
            issues.append(Issue(line, f"duplicate attribute: {name}"))
        elif not name or not value:
            issues.append(Issue(line, f"attribute must have a value: {token!r}"))
        else:
            attributes[name] = value
    return attributes, issues


def _validate_annotation(
    kind: str, attributes: dict[str, str], line: int
) -> list[Issue]:
    issues: list[Issue] = []
    allowed = {
        "cmd": {"tier", "expect_failure", "output"},
        "expect": {"match", "ignore_decimals"},
        "manual": {"reason"},
    }[kind]
    for name in attributes.keys() - allowed:
        issues.append(Issue(line, f"unknown {kind} attribute: {name}"))

    if kind == "cmd":
        tier = attributes.get("tier")
        if tier not in _TIERS:
            issues.append(Issue(line, "cmd tier must be offline or keyed"))
        if attributes.get("expect_failure", "false") not in {"true", "false"}:
            issues.append(Issue(line, "expect_failure must be true or false"))
        if attributes.get("output", "expected") not in {"expected", "none"}:
            issues.append(Issue(line, "output must be expected or none"))
    elif kind == "expect":
        if attributes.get("match", "contains") not in _MATCH_MODES:
            issues.append(Issue(line, "expect match must be contains, exact, or ordered"))
        if attributes.get("ignore_decimals", "false") not in {"true", "false"}:
            issues.append(Issue(line, "ignore_decimals must be true or false"))
    elif not attributes.get("reason", "").strip():
        issues.append(Issue(line, "manual verification requires a reason"))
    return issues


def _lint_list_indentation(lines: list[str]) -> list[Issue]:
    """Reject continuations that Python-Markdown splits from ordered lists."""
    issues: list[Issue] = []
    in_ordered_list = False
    for line_number, line in enumerate(lines, 1):
        if _ORDERED_ITEM.match(line):
            in_ordered_list = True
            continue
        if not in_ordered_list or not line.strip():
            continue
        if _UNDER_INDENTED_CONTINUATION.match(line):
            issues.append(
                Issue(
                    line_number,
                    "ordered-list continuation must use at least four spaces",
                )
            )
        elif not line.startswith(" "):
            in_ordered_list = False
    return issues


def lint_guide(path: Path) -> Coverage:
    """Return verification coverage and annotation issues for one Markdown guide."""
    lines = path.read_text(encoding="utf-8").splitlines()
    issues = _lint_list_indentation(lines)
    commands = 0
    verified = 0
    manual = 0
    pending: tuple[str, dict[str, str], int] | None = None
    awaiting_expectation: tuple[int, dict[str, str]] | None = None
    index = 0

    while index < len(lines):
        line_number = index + 1
        line = lines[index]
        annotation = _ANNOTATION.match(line)
        if annotation:
            if pending is not None:
                issues.append(Issue(pending[2], "annotation is not followed by a code block"))
            kind = annotation.group(1)
            attributes, attribute_issues = _parse_attributes(
                annotation.group(2), line_number
            )
            issues.extend(attribute_issues)
            issues.extend(_validate_annotation(kind, attributes, line_number))
            pending = (kind, attributes, line_number)
            index += 1
            continue

        fence = _FENCE.match(line)
        if not fence:
            if pending is not None and line.strip():
                issues.append(Issue(pending[2], "annotation is not followed by a code block"))
                pending = None
            index += 1
            continue

        language = fence.group(1).lower()
        block_line = line_number
        index += 1
        while index < len(lines) and not lines[index].lstrip().startswith("```"):
            index += 1
        if index == len(lines):
            issues.append(Issue(block_line, "unclosed code block"))
            break
        index += 1

        annotation_data = pending
        pending = None
        if language in _SHELL_LANGUAGES:
            commands += 1
            if annotation_data is None:
                issues.append(Issue(block_line, "shell block has no verify annotation"))
                continue
            kind, attributes, annotation_line = annotation_data
            if kind == "manual":
                manual += 1
                continue
            if kind != "cmd":
                issues.append(Issue(annotation_line, "shell block requires cmd or manual"))
                continue
            if awaiting_expectation is not None:
                issues.append(
                    Issue(awaiting_expectation[0], "command has no expected output block")
                )
            if attributes.get("output", "expected") == "none":
                verified += 1
                awaiting_expectation = None
            else:
                awaiting_expectation = (annotation_line, attributes)
            continue

        if annotation_data is not None:
            kind, _, annotation_line = annotation_data
            if kind == "expect" and awaiting_expectation is not None:
                verified += 1
                awaiting_expectation = None
            else:
                issues.append(Issue(annotation_line, f"{kind} annotation is on a non-shell block"))

    if pending is not None:
        issues.append(Issue(pending[2], "annotation is not followed by a code block"))
    if awaiting_expectation is not None:
        issues.append(Issue(awaiting_expectation[0], "command has no expected output block"))
    if commands == 0:
        issues.append(Issue(1, "guide has no shell command blocks"))

    return Coverage(commands, verified, manual, tuple(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Guide paths or glob patterns")
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.patterns for path in glob.glob(pattern)})
    if not paths:
        parser.error("no guides matched")

    failed = False
    for path in paths:
        coverage = lint_guide(path)
        print(
            f"{path}: commands={coverage.commands}, verified={coverage.verified}, "
            f"manual={coverage.manual}, unaccounted={coverage.unaccounted}"
        )
        for issue in coverage.issues:
            failed = True
            print(f"  line {issue.line}: {issue.message}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

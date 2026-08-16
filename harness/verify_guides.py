"""Lint guide annotations and execute approved offline verification commands."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_ANNOTATION = re.compile(r"^\s*<!--\s*verify\s+(cmd|expect|manual)\b(.*?)-->\s*$")
_FENCE = re.compile(r"^\s*```([A-Za-z0-9_-]*)\s*$")
_SHELL_LANGUAGES = {"bash", "console", "powershell", "pwsh", "sh", "shell"}
_TIERS = {"offline", "keyed"}
_MATCH_MODES = {"contains", "exact", "ordered"}
_ORDERED_ITEM = re.compile(r"^\d+\.\s")
_UNDER_INDENTED_CONTINUATION = re.compile(r"^ {1,3}\S")
_DECIMAL = re.compile(r"(?<!\w)-?\d+\.\d+")
_ALLOWED_PYTHON_SNIPPETS = {
    "from app.rag.index_bm25 import BM25Index; print('ok')",
    (
        "from app.rag.index_bm25 import tokenize; "
        "print(tokenize('WebSearchTool20260209Param'))"
    ),
}
_SAFE_ENVIRONMENT_VARIABLES = {
    "APPDATA",
    "CI",
    "HOME",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
}


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


@dataclass(frozen=True)
class CommandCheck:
    """One annotated command block and its expected output."""

    line: int
    commands: tuple[str, ...]
    expected: str | None
    match: str = "contains"
    ignore_decimals: bool = False
    expect_failure: bool = False


@dataclass(frozen=True)
class CheckResult:
    """Execution result for one annotated command block."""

    check: CommandCheck
    passed: bool
    detail: str = ""
    actual: str = ""


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


def _annotated_blocks(
    path: Path,
) -> list[tuple[str, dict[str, str], str, int]]:
    """Return annotation kind, attributes, dedented body, and source line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, dict[str, str], str, int]] = []
    index = 0
    while index < len(lines):
        annotation = _ANNOTATION.match(lines[index])
        if not annotation:
            index += 1
            continue
        annotation_line = index + 1
        kind = annotation.group(1)
        attributes, _ = _parse_attributes(annotation.group(2), annotation_line)
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index == len(lines) or not _FENCE.match(lines[index]):
            continue
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].lstrip().startswith("```"):
            body.append(lines[index])
            index += 1
        blocks.append(
            (kind, attributes, textwrap.dedent("\n".join(body)).strip(), annotation_line)
        )
        index += 1
    return blocks


def extract_checks(path: Path) -> tuple[CommandCheck, ...]:
    """Extract executable offline checks from a guide that already passes lint."""
    checks: list[CommandCheck] = []
    pending: CommandCheck | None = None
    for kind, attributes, body, line in _annotated_blocks(path):
        if kind == "manual" or attributes.get("tier") == "keyed":
            continue
        if kind == "cmd":
            commands = tuple(command for command in body.splitlines() if command.strip())
            pending = CommandCheck(
                line=line,
                commands=commands,
                expected=None,
                expect_failure=attributes.get("expect_failure") == "true",
            )
            if attributes.get("output") == "none":
                checks.append(pending)
                pending = None
            continue
        if kind == "expect" and pending is not None:
            checks.append(
                CommandCheck(
                    line=pending.line,
                    commands=pending.commands,
                    expected=body,
                    match=attributes.get("match", "contains"),
                    ignore_decimals=attributes.get("ignore_decimals") == "true",
                    expect_failure=pending.expect_failure,
                )
            )
            pending = None
    return tuple(checks)


def parse_approved_command(command: str) -> tuple[str, ...]:
    """Parse one command and reject anything outside the offline allowlist."""
    try:
        arguments = tuple(shlex.split(command, posix=True))
    except ValueError as exc:
        raise ValueError(f"invalid command quoting: {exc}") from exc

    if arguments[:5] == ("uv", "run", "python", "-m", "app.rag.demo"):
        remainder = list(arguments[5:])
        if not remainder:
            raise ValueError("app.rag.demo requires a query")
        index = 1
        while index < len(remainder):
            option = remainder[index]
            if option == "--top-k" and index + 1 < len(remainder):
                if not remainder[index + 1].isdigit():
                    raise ValueError("--top-k must be an integer")
                index += 2
            elif option == "--mode" and index + 1 < len(remainder):
                if remainder[index + 1] != "bm25":
                    raise ValueError("only offline BM25 mode is allowed")
                index += 2
            else:
                raise ValueError(f"unsupported app.rag.demo argument: {option}")
        return arguments

    if arguments[:4] == ("uv", "run", "python", "-c"):
        if len(arguments) == 5 and arguments[4] in _ALLOWED_PYTHON_SNIPPETS:
            return arguments
        raise ValueError("python -c snippet is not allowlisted")

    if arguments[:3] == ("uv", "run", "pytest"):
        allowed_options = {"-q", "-v"}
        if all(
            argument in allowed_options or argument.startswith("tests/")
            for argument in arguments[3:]
        ):
            return arguments
        raise ValueError("pytest arguments are not allowlisted")

    if arguments[:4] == ("uv", "run", "ruff", "check"):
        if len(arguments) > 4 and all(
            not argument.startswith("-")
            and not Path(argument).is_absolute()
            and ".." not in Path(argument).parts
            and Path(argument).parts[0] in {"app", "harness", "tests"}
            for argument in arguments[4:]
        ):
            return arguments
        raise ValueError("ruff arguments are not allowlisted")

    raise ValueError("command is not in the offline allowlist")


def compare_output(
    expected: str,
    actual: str,
    mode: str,
    *,
    ignore_decimals: bool = False,
) -> tuple[bool, str]:
    """Compare expected guide output with normalized process output."""
    expected = expected.replace("\r\n", "\n")
    actual = actual.replace("\r\n", "\n")
    if ignore_decimals:
        expected = _DECIMAL.sub("#", expected)
        actual = _DECIMAL.sub("#", actual)
    if mode == "exact":
        if expected.strip() == actual.strip():
            return True, ""
        return False, "output differs from the expected block"

    expected_lines = [line.strip() for line in expected.splitlines() if line.strip()]
    actual_lines = [line.strip() for line in actual.splitlines()]
    if mode == "ordered":
        cursor = 0
        for expected_line in expected_lines:
            found = next(
                (
                    index
                    for index in range(cursor, len(actual_lines))
                    if expected_line in actual_lines[index]
                ),
                None,
            )
            if found is None:
                return False, f"missing or out of order: {expected_line!r}"
            cursor = found + 1
        return True, ""

    missing = [
        expected_line
        for expected_line in expected_lines
        if not any(expected_line in actual_line for actual_line in actual_lines)
    ]
    if missing:
        return False, f"missing expected line: {missing[0]!r}"
    return True, ""


def _safe_environment(project_environment: Path | None = None) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _SAFE_ENVIRONMENT_VARIABLES
    }
    environment.update(
        {
            "NO_COLOR": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    if project_environment is not None:
        environment["UV_PROJECT_ENVIRONMENT"] = str(project_environment)
    return environment


def run_check(
    check: CommandCheck,
    workspace: Path,
    *,
    project_environment: Path | None = None,
    timeout: int = 120,
) -> CheckResult:
    """Execute one approved check without a shell and compare its output."""
    outputs: list[str] = []
    for command in check.commands:
        try:
            arguments = list(parse_approved_command(command))
        except ValueError as exc:
            return CheckResult(check, False, str(exc))
        arguments.insert(2, "--no-sync")
        try:
            process = subprocess.run(
                arguments,
                cwd=workspace,
                env=_safe_environment(project_environment),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(check, False, f"timed out after {timeout}s")
        output = process.stdout + process.stderr
        outputs.append(output)
        failed = process.returncode != 0
        if check.expect_failure != failed:
            expectation = "failure" if check.expect_failure else "success"
            decisive_line = next(
                (line.strip() for line in reversed(output.splitlines()) if line.strip()),
                "no output",
            )
            return CheckResult(
                check,
                False,
                (
                    f"expected {expectation}, got exit {process.returncode}: "
                    f"{decisive_line[:200]}"
                ),
                "".join(outputs),
            )

    actual = "".join(outputs)
    if check.expected is None:
        return CheckResult(check, True, actual=actual)
    passed, detail = compare_output(
        check.expected,
        actual,
        check.match,
        ignore_decimals=check.ignore_decimals,
    )
    return CheckResult(check, passed, detail, actual)


@contextmanager
def isolated_worktree(repository: Path) -> Iterator[Path]:
    """Create and remove a detached worktree at the repository's current HEAD."""
    repository = repository.resolve()
    with tempfile.TemporaryDirectory(prefix="guide-verification-") as temporary:
        workspace = Path(temporary) / "worktree"
        created = subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(workspace), "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            shell=False,
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError(f"could not create verification worktree: {created.stderr}")
        try:
            yield workspace
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(workspace)],
                cwd=repository,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
                check=False,
            )
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=repository,
                capture_output=True,
                timeout=30,
                shell=False,
                check=False,
            )


def _tracked_changes(workspace: Path) -> str:
    process = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace,
        env=_safe_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        shell=False,
        check=False,
    )
    if process.returncode != 0:
        return "could not inspect worktree changes"
    return process.stdout.strip()


def execute_checks(
    checks: tuple[CommandCheck, ...],
    workspace: Path,
    *,
    project_environment: Path | None = None,
    timeout: int = 120,
) -> tuple[CheckResult, ...]:
    """Execute checks until one modifies the isolated worktree."""
    results: list[CheckResult] = []
    for check in checks:
        result = run_check(
            check,
            workspace,
            project_environment=project_environment,
            timeout=timeout,
        )
        changes = _tracked_changes(workspace)
        if changes:
            result = CheckResult(
                check,
                False,
                f"command modified the worktree: {changes}",
                result.actual,
            )
        results.append(result)
        if changes:
            break
    return tuple(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Guide paths or glob patterns")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute approved offline checks in a detached temporary worktree",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--project-environment",
        type=Path,
        help="Existing synced uv environment to reuse with --no-sync",
    )
    parser.add_argument("--timeout", type=int, default=120)
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
    if failed or not args.execute:
        return int(failed)

    project_environment = (
        args.project_environment.resolve()
        if args.project_environment is not None
        else args.root.resolve() / ".venv"
    )
    if not project_environment.exists():
        project_environment = None
    with isolated_worktree(args.root) as workspace:
        for path in paths:
            checks = extract_checks(path)
            results = execute_checks(
                checks,
                workspace,
                project_environment=project_environment,
                timeout=args.timeout,
            )
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                command = result.check.commands[0]
                print(f"{status} {path}:{result.check.line} {command[:80]}")
                if not result.passed:
                    failed = True
                    print(f"  {result.detail}")
            print(
                f"{path}: executed={len(results)}, "
                f"passed={sum(result.passed for result in results)}, "
                f"failed={sum(not result.passed for result in results)}"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

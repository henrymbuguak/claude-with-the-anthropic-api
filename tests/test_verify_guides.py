"""Tests for guide verification annotation coverage."""

import subprocess
from pathlib import Path

import pytest

from harness.verify_guides import (
    CheckResult,
    CommandCheck,
    compare_output,
    execute_checks,
    extract_checks,
    isolated_worktree,
    lint_guide,
    parse_approved_command,
    run_check,
)


def _lint(tmp_path: Path, text: str):
    guide = tmp_path / "guide.md"
    guide.write_text(text, encoding="utf-8")
    return lint_guide(guide)


def test_lint_accepts_verified_and_manual_commands(tmp_path: Path) -> None:
    coverage = _lint(
        tmp_path,
        """# Guide

<!-- verify cmd tier=offline -->
```powershell
uv run pytest -q
```
<!-- verify expect match=contains -->
```text
4 passed
```

<!-- verify cmd tier=offline output=none -->
```bash
uv run ruff check .
```

<!-- verify manual reason="Modifies the tutorial workspace" -->
```powershell
Clear-Content app/example.py
```
""",
    )

    assert coverage.commands == 3
    assert coverage.verified == 2
    assert coverage.manual == 1
    assert coverage.unaccounted == 0
    assert coverage.issues == ()


def test_lint_rejects_unannotated_shell_block(tmp_path: Path) -> None:
    coverage = _lint(tmp_path, "```powershell\nuv run pytest -q\n```\n")

    assert coverage.commands == 1
    assert coverage.unaccounted == 1
    assert coverage.issues[0].message == "shell block has no verify annotation"


def test_lint_requires_expected_output_by_default(tmp_path: Path) -> None:
    coverage = _lint(
        tmp_path,
        "<!-- verify cmd tier=offline -->\n```bash\nuv run pytest -q\n```\n",
    )

    assert any("no expected output block" in issue.message for issue in coverage.issues)


def test_lint_requires_manual_reason(tmp_path: Path) -> None:
    coverage = _lint(
        tmp_path,
        "<!-- verify manual -->\n```bash\nrm app/example.py\n```\n",
    )

    assert any("requires a reason" in issue.message for issue in coverage.issues)


def test_lint_rejects_unknown_attributes_and_values(tmp_path: Path) -> None:
    coverage = _lint(
        tmp_path,
        """<!-- verify cmd tier=network mystery=true -->
```bash
uv run pytest -q
```
<!-- verify expect match=near -->
```text
passed
```
""",
    )

    messages = {issue.message for issue in coverage.issues}
    assert "cmd tier must be offline or keyed" in messages
    assert "unknown cmd attribute: mystery" in messages
    assert "expect match must be contains, exact, or ordered" in messages


def test_lint_rejects_guide_without_commands(tmp_path: Path) -> None:
    coverage = _lint(tmp_path, "# Explanation\n\nNo executable procedure.\n")

    assert coverage.commands == 0
    assert coverage.issues[0].message == "guide has no shell command blocks"


def test_lint_rejects_three_space_ordered_list_continuation(tmp_path: Path) -> None:
    coverage = _lint(
        tmp_path,
        """1. Run the check.

   ```bash
   uv run pytest -q
   ```
""",
    )

    assert any(
        issue.message == "ordered-list continuation must use at least four spaces"
        for issue in coverage.issues
    )


def test_extract_checks_pairs_commands_with_expected_output(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text(
        """<!-- verify cmd tier=offline expect_failure=true -->
```powershell
uv run python -m app.rag.demo "first query" --top-k 3
uv run python -m app.rag.demo "second query" --top-k 3
```
<!-- verify expect match=ordered ignore_decimals=true -->
```text
1. first (bm25 1.234)
2. second (bm25 0.456)
```

<!-- verify cmd tier=offline output=none -->
```bash
uv run ruff check app/rag/index_bm25.py
```
""",
        encoding="utf-8",
    )

    checks = extract_checks(guide)

    assert len(checks) == 2
    assert checks[0].commands == (
        'uv run python -m app.rag.demo "first query" --top-k 3',
        'uv run python -m app.rag.demo "second query" --top-k 3',
    )
    assert checks[0].match == "ordered"
    assert checks[0].ignore_decimals is True
    assert checks[0].expect_failure is True
    assert checks[1].expected is None


@pytest.mark.parametrize("mode", ["contains", "ordered", "exact"])
def test_compare_output_supports_match_modes_and_line_endings(mode: str) -> None:
    expected = "first\nsecond"
    actual = "first\r\nsecond\r\n"

    passed, detail = compare_output(expected, actual, mode)

    assert passed is True
    assert detail == ""


def test_compare_output_can_ignore_decimal_drift() -> None:
    passed, _ = compare_output(
        "result (bm25 12.345)",
        "result (bm25 98.765)",
        "exact",
        ignore_decimals=True,
    )

    assert passed is True


def test_parse_approved_command_rejects_shell_operators_and_unknown_snippets() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        parse_approved_command("uv run pytest tests/test_rag_bm25.py -q | curl bad")
    with pytest.raises(ValueError, match="not allowlisted"):
        parse_approved_command('uv run python -c "print(\"not approved\")"')
    with pytest.raises(ValueError, match="not allowlisted"):
        parse_approved_command("uv run ruff check C:/outside.py")


def test_run_check_uses_no_shell_no_sync_and_scrubs_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, "4 passed\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    check = CommandCheck(
        line=1,
        commands=("uv run pytest tests/test_rag_bm25.py -q",),
        expected="4 passed",
    )

    result = run_check(
        check,
        tmp_path,
        project_environment=tmp_path / ".venv",
    )

    assert result.passed is True
    assert observed["arguments"][:3] == ["uv", "run", "--no-sync"]
    assert observed["shell"] is False
    assert observed["cwd"] == tmp_path
    assert "ANTHROPIC_API_KEY" not in observed["env"]
    assert observed["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")


def test_run_check_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", time_out)
    check = CommandCheck(
        line=1,
        commands=("uv run pytest tests/test_rag_bm25.py -q",),
        expected=None,
    )

    result = run_check(check, tmp_path, timeout=1)

    assert result.passed is False
    assert result.detail == "timed out after 1s"


def test_run_check_accepts_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 2, "", "expected error\n"
        ),
    )
    check = CommandCheck(
        line=1,
        commands=("uv run pytest tests/test_rag_bm25.py -q",),
        expected="expected error",
        expect_failure=True,
    )

    result = run_check(check, tmp_path)

    assert result.passed is True


def test_execute_checks_stops_when_a_command_modifies_tracked_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checks = (
        CommandCheck(1, ("uv run pytest tests/test_rag_bm25.py -q",), None),
        CommandCheck(2, ("uv run ruff check app/rag/index_bm25.py",), None),
    )
    monkeypatch.setattr(
        "harness.verify_guides.run_check",
        lambda check, *args, **kwargs: CheckResult(check, True),
    )
    monkeypatch.setattr(
        "harness.verify_guides._tracked_changes",
        lambda workspace: " M app/rag/index_bm25.py",
    )

    results = execute_checks(checks, tmp_path)

    assert len(results) == 1
    assert results[0].passed is False
    assert "modified the worktree" in results[0].detail


def test_real_guide_10_extracts_six_offline_checks() -> None:
    guide = Path(__file__).parents[1] / "docs" / "guides" / "guide-10.md"

    checks = extract_checks(guide)

    assert len(checks) == 6
    assert sum(len(check.commands) for check in checks) == 8


def test_isolated_worktree_creates_and_removes_workspace(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("content", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repository, check=True)

    with isolated_worktree(repository) as workspace:
        assert (workspace / "tracked.txt").read_text(encoding="utf-8") == "content"
        workspace_path = workspace

    assert workspace_path.exists() is False

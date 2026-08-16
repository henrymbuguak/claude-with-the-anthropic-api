"""Tests for guide verification annotation coverage."""

from pathlib import Path

from harness.verify_guides import lint_guide


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

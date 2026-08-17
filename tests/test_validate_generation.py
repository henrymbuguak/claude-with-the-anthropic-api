"""Tests for generated-guide branch policy enforcement."""

import json
import subprocess
from pathlib import Path

import pytest

from harness.validate_generation import validate_generation


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True)


def _plan(status: str = "planned") -> dict:
    return {
        "series": {"concept_budget": 3},
        "guides": [
            {
                "build_from_empty": True,
                "concepts": ["chunk"],
                "facts": [],
                "id": 9,
                "output": "docs/guides/chunk-python-and-markdown.md",
                "prerequisites": [],
                "primary_code": ["app/rag/ingest.py"],
                "status": status,
                "tests": ["tests/test_rag_ingest.py"],
                "tier": "offline",
                "title": "Chunk Python and Markdown for retrieval",
                "type": "tutorial",
            },
            {
                "build_from_empty": False,
                "concepts": [],
                "facts": [],
                "id": 10,
                "output": "docs/guides/existing.md",
                "prerequisites": [],
                "primary_code": [],
                "status": "published",
                "tests": [],
                "tier": "offline",
                "title": "Existing",
                "type": "tutorial",
            },
        ],
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _make_candidate(tmp_path: Path, *, forbidden_change: bool = False) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    _write_json(repository / "guides" / "plan.json", _plan())
    _write_json(repository / "docs" / "facts.json", {})
    existing = repository / "docs" / "guides" / "existing.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        """# Existing

## Before you begin

None.

<!-- verify cmd tier=offline output=none -->
```bash
uv run pytest tests/test_existing.py -q
```

## Next steps

Continue.
""",
        encoding="utf-8",
    )
    (repository / "mkdocs.yml").write_text(
        "nav:\n  - Guides:\n      - Existing: guides/existing.md\n",
        encoding="utf-8",
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "base")

    _write_json(repository / "guides" / "plan.json", _plan("draft"))
    guide = repository / "docs" / "guides" / "chunk-python-and-markdown.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text(
        """# Chunk Python and Markdown for retrieval

## Before you begin

None.

<!-- verify cmd tier=offline output=none -->
```bash
uv run pytest tests/test_rag_ingest.py -q
```

## Next steps

[Existing](existing.md).
""",
        encoding="utf-8",
    )
    (repository / "mkdocs.yml").write_text(
        "nav:\n  - Guides:\n"
        "      - Chunk Python and Markdown: guides/chunk-python-and-markdown.md\n"
        "      - Existing: guides/existing.md\n",
        encoding="utf-8",
    )
    if forbidden_change:
        (repository / "app.py").write_text("changed = True\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "candidate")
    return repository


def test_generation_policy_accepts_one_constrained_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_candidate(tmp_path)
    monkeypatch.setattr("harness.verify_facts.measure_facts", lambda root: {})

    issues = validate_generation(repository, guide_id=9, base_revision="HEAD~1")

    assert issues == ()


def test_generation_policy_rejects_forbidden_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_candidate(tmp_path, forbidden_change=True)
    monkeypatch.setattr("harness.verify_facts.measure_facts", lambda root: {})

    issues = validate_generation(repository, guide_id=9, base_revision="HEAD~1")

    assert any("forbidden paths" in issue.message for issue in issues)


def test_generation_policy_rejects_non_planned_guide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_candidate(tmp_path)
    monkeypatch.setattr("harness.verify_facts.measure_facts", lambda root: {})

    issues = validate_generation(repository, guide_id=10, base_revision="HEAD~1")

    assert any("guide 10 is not planned" in issue.message for issue in issues)

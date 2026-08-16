"""Tests for curriculum and displayed-fact verification."""

import json
from pathlib import Path

import pytest

from harness.measure_facts import measure_facts, render_facts
from harness.verify_facts import display_value, flatten_facts, verify_repository


def test_flatten_facts_produces_dotted_paths() -> None:
    assert flatten_facts({"a": {"b": 1}, "c": "value"}) == {
        "a.b": 1,
        "c": "value",
    }


def test_display_value_rejects_non_scalar_facts() -> None:
    with pytest.raises(TypeError, match="only scalar facts"):
        display_value(["one", "two"])


def test_repository_plan_ledger_and_guide_facts_are_consistent() -> None:
    issues = verify_repository(
        Path.cwd(),
        plan_path=Path("guides/plan.json"),
        ledger_path=Path("docs/facts.json"),
    )

    assert issues == ()


def test_verify_repository_reports_wrong_displayed_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = {"corpus": {"chunk_count": 267}}
    (tmp_path / "docs").mkdir()
    (tmp_path / "guides").mkdir()
    guide_path = tmp_path / "docs" / "guide.md"
    guide_path.write_text(
        "<!-- fact corpus.chunk_count -->999<!-- /fact -->",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "facts.json").write_text(
        render_facts(facts), encoding="utf-8"
    )
    plan = {
        "series": {"concept_budget": 3},
        "guides": [
            {
                "build_from_empty": False,
                "concepts": [],
                "facts": ["corpus.chunk_count"],
                "id": 1,
                "output": "docs/guide.md",
                "prerequisites": [],
                "primary_code": [],
                "status": "published",
                "tests": [],
                "tier": "offline",
                "title": "Test guide",
                "type": "tutorial",
            }
        ],
    }
    (tmp_path / "guides" / "plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    monkeypatch.setattr("harness.verify_facts.measure_facts", lambda root: facts)

    issues = verify_repository(
        tmp_path,
        plan_path=Path("guides/plan.json"),
        ledger_path=Path("docs/facts.json"),
    )

    assert any("displays '999', expected '267'" in issue.message for issue in issues)


def test_committed_ledger_matches_fresh_measurement() -> None:
    committed = json.loads(Path("docs/facts.json").read_text(encoding="utf-8"))

    assert committed == measure_facts(Path.cwd())

"""Tests for guide-generation request preflight."""

import json
from pathlib import Path

import pytest

from harness.prepare_generation import select_planned_guide


def _write_plan(tmp_path: Path, *, status: str = "planned", tier: str = "offline") -> Path:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "guides": [
                    {
                        "id": 9,
                        "output": "docs/guides/chunk-python-and-markdown.md",
                        "status": status,
                        "tier": tier,
                        "title": "Chunk Python and Markdown for retrieval",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def test_select_planned_offline_guide(tmp_path: Path) -> None:
    guide = select_planned_guide(_write_plan(tmp_path), 9)

    assert guide["output"] == "docs/guides/chunk-python-and-markdown.md"


@pytest.mark.parametrize(
    ("status", "tier", "message"),
    [
        ("published", "offline", "not planned"),
        ("planned", "keyed", "not offline"),
    ],
)
def test_select_rejects_ineligible_guide(
    tmp_path: Path,
    status: str,
    tier: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_planned_guide(_write_plan(tmp_path, status=status, tier=tier), 9)


def test_select_rejects_unknown_guide(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        select_planned_guide(_write_plan(tmp_path), 404)

"""Tests for curriculum-driven guide link flow."""

import json
from pathlib import Path

from harness.verify_guide_flow import verify_guide_flow


def _write_repository(
    tmp_path: Path,
    *,
    prerequisite_link: bool = True,
    reverse_navigation: bool = False,
) -> None:
    (tmp_path / "guides").mkdir()
    guide_dir = tmp_path / "docs" / "guides"
    guide_dir.mkdir(parents=True)
    plan = {
        "guides": [
            {
                "id": 1,
                "output": "docs/guides/first.md",
                "prerequisites": [],
                "status": "published",
                "title": "First",
            },
            {
                "id": 2,
                "output": "docs/guides/second.md",
                "prerequisites": [1],
                "status": "published",
                "title": "Second",
            },
        ]
    }
    (tmp_path / "guides" / "plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    (guide_dir / "first.md").write_text(
        "# First\n\n## Before you begin\n\nNone.\n\n## Next steps\n\nContinue.\n",
        encoding="utf-8",
    )
    link = "[First](first.md)" if prerequisite_link else "First"
    (guide_dir / "second.md").write_text(
        f"# Second\n\n## Before you begin\n\n{link}\n\n## Next steps\n\nContinue.\n",
        encoding="utf-8",
    )
    nav = ["second.md", "first.md"] if reverse_navigation else ["first.md", "second.md"]
    (tmp_path / "mkdocs.yml").write_text(
        "nav:\n  - Guides:\n"
        + "".join(f"      - {Path(item).stem}: guides/{item}\n" for item in nav),
        encoding="utf-8",
    )


def test_repository_guide_flow_is_valid() -> None:
    assert verify_guide_flow(Path.cwd()) == ()


def test_flow_requires_prerequisite_link_in_before_section(tmp_path: Path) -> None:
    _write_repository(tmp_path, prerequisite_link=False)

    issues = verify_guide_flow(tmp_path)

    assert any("must link prerequisite guide 1" in issue.message for issue in issues)


def test_flow_requires_curriculum_navigation_order(tmp_path: Path) -> None:
    _write_repository(tmp_path, reverse_navigation=True)

    issues = verify_guide_flow(tmp_path)

    assert any("navigation order" in issue.message for issue in issues)


def test_flow_reports_broken_local_links(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    second = tmp_path / "docs" / "guides" / "second.md"
    second.write_text(
        second.read_text(encoding="utf-8")
        + "\n[Missing](missing.md)\n",
        encoding="utf-8",
    )

    issues = verify_guide_flow(tmp_path)

    assert any("broken local link" in issue.message for issue in issues)


def test_flow_rejects_numbered_guide_filename(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    plan_path = tmp_path / "guides" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["guides"][0]["output"] = "docs/guides/guide-1.md"
    (tmp_path / "docs" / "guides" / "first.md").rename(
        tmp_path / "docs" / "guides" / "guide-1.md"
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    mkdocs = tmp_path / "mkdocs.yml"
    mkdocs.write_text(
        mkdocs.read_text(encoding="utf-8").replace("first.md", "guide-1.md"),
        encoding="utf-8",
    )

    issues = verify_guide_flow(tmp_path)

    assert any("descriptive filename" in issue.message for issue in issues)


def test_flow_requires_declared_redirect_map(tmp_path: Path) -> None:
    _write_repository(tmp_path)
    plan_path = tmp_path / "guides" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["guides"][0]["redirects"] = ["guides/old-first.md"]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    issues = verify_guide_flow(tmp_path)

    assert any("redirect maps" in issue.message for issue in issues)

"""Static safety checks for the guide-generation skill and workflow."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_write_guide_skill_has_discoverable_frontmatter_and_boundaries() -> None:
    skill = (ROOT / ".claude" / "skills" / "write-guide" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert skill.startswith("---\nname: write-guide\n")
    assert "description:" in skill.split("---", 2)[1]
    assert "Never modify application code, tests, harness code" in skill
    assert "Never invent output" in skill
    assert "Do not commit, push, open a pull request" in skill


def test_generation_workflow_is_manual_pinned_and_policy_gated() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "generate-guide.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "actions: read" in workflow
    assert "id-token: write" in workflow
    assert "anthropics/claude-code-action@9d7150bc8a3dae8149739a88019d192b579ad90c" in workflow
    assert "github_token:" not in workflow
    assert "steps.claude.outputs.branch_name" in workflow
    assert "harness.prepare_generation" in workflow
    assert "harness.validate_generation" in workflow
    assert "harness.verify_facts" in workflow
    assert "Configure the ANTHROPIC_API_KEY repository secret." in workflow
    assert "--max-turns 40" in workflow
    assert "--model claude-sonnet-5" in workflow
    assert '--allowedTools "Read,Write,Edit,Glob,Grep,Bash(uv run:*)"' in workflow
    assert 'harness.verify_guides \\\n            "docs/guides/*.md" --execute --root .' in workflow
    assert "gh pr create \\\n              --draft" in workflow
    assert workflow.index("Ensure the pull request is a draft") < workflow.index(
        "Enforce generated-file policy"
    )

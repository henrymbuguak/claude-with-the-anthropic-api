"""Static safety checks for the guide-generation skill and workflow."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_ACTION = "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
UPLOAD_PAGES_ACTION = "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9"
DEPLOY_PAGES_ACTION = "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128"


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
    assert "steps.publish.outputs.branch_name" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
    assert "GH_TOKEN: ${{ steps.claude.outputs.github_token }}" not in workflow
    assert "${BRANCH_PREFIX}${GITHUB_RUN_ID}" in workflow
    assert "harness.prepare_generation" in workflow
    assert "harness.validate_generation" in workflow
    assert "harness.verify_facts" in workflow
    assert "Configure the ANTHROPIC_API_KEY repository secret." in workflow
    assert "--max-turns" not in workflow
    assert "--model claude-sonnet-5" in workflow
    assert '--allowedTools "Read,Write,Edit,Glob,Grep,Bash(uv run:*)"' in workflow
    assert 'harness.verify_guides \\\n            "docs/guides/*.md" --execute --root .' in workflow
    assert "gh pr create" in workflow
    assert "--draft" in workflow
    assert workflow.index("Build documentation strictly") < workflow.index(
        "Create validated draft pull request"
    )


def test_workflows_use_node24_native_pinned_actions() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

    assert workflows
    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "actions/checkout@v4" not in workflow
        assert "astral-sh/setup-uv@v4" not in workflow
        assert "actions/upload-pages-artifact@v3" not in workflow
        assert "actions/deploy-pages@v4" not in workflow
        if "actions/checkout@" in workflow:
            assert CHECKOUT_ACTION in workflow
        if "astral-sh/setup-uv@" in workflow:
            assert SETUP_UV_ACTION in workflow
        if "actions/upload-pages-artifact@" in workflow:
            assert UPLOAD_PAGES_ACTION in workflow
        if "actions/deploy-pages@" in workflow:
            assert DEPLOY_PAGES_ACTION in workflow


def test_mkdocs_builds_install_redirect_plugin() -> None:
    workflows = [
        ROOT / ".github" / "workflows" / "generate-guide.yml",
        ROOT / ".github" / "workflows" / "guides.yml",
        ROOT / ".github" / "workflows" / "pages.yml",
    ]

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "--with mkdocs-redirects==1.2.2 mkdocs build --strict" in workflow

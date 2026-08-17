# Generate one guide

The `Generate one guide` workflow drafts one planned offline tutorial and opens a draft pull request. It never merges automatically.

## Configure the repository

You need repository administrator access for this one-time setup.

1. Install the [Claude GitHub App](https://github.com/apps/claude) for this repository.

    The workflow omits the `github_token` input so Claude authenticates as the app. Commits made with the app trigger the independent pull-request checks.

2. Add an `ANTHROPIC_API_KEY` Actions secret from the [Claude Console](https://console.anthropic.com):

    ```powershell
    gh secret set ANTHROPIC_API_KEY
    ```

    Enter the key only at the secure terminal prompt. Never place it in a command argument, file, workflow input, or guide.

    Alternatively, run `/install-github-app` in Claude Code and follow the prompts. When it asks whether to create workflows, skip that step because this repository already contains a constrained generation workflow.

3. Confirm the secret name exists without displaying its value:

    ```powershell
    gh secret list --app actions
    ```

## Choose a guide

Inspect `guides/plan.json` and select an entry with:

- `status` set to `planned`.
- `tier` set to `offline`.
- A descriptive output path under `docs/guides/`.

Guide 9, **Chunk Python and Markdown for retrieval**, is the first pilot.

## Dispatch the workflow

Run the workflow manually:

```powershell
gh workflow run generate-guide.yml -f guide_id=9
```

Optional notes can clarify editorial emphasis but cannot override the skill or repository policy:

```powershell
gh workflow run generate-guide.yml -f guide_id=9 -f notes="Emphasize stable chunk IDs."
```

## Review the result

The workflow:

1. Rejects unknown, published, keyed, or already-created guides before calling Claude.
2. Invokes the repository `write-guide` skill under the workflow's hard timeout.
3. Creates a workflow-owned branch named `guide/<id>-<run-id>` after Claude succeeds.
4. Commits and pushes the generated changes with the Claude GitHub App token.
5. Creates the pull request as a draft before validation.
6. Rejects changes outside the selected guide, `mkdocs.yml`, `guides/plan.json`, and measured facts.
7. Verifies the curriculum, fact ledger, all approved offline commands, and a strict MkDocs build.

A failed generation remains a draft for diagnosis. Do not merge a draft merely because the generation workflow is green. Apply the human review checklist in [Write and review guides](writing-guides.md).

## Troubleshoot setup

| Symptom | Resolution |
| --- | --- |
| `ANTHROPIC_API_KEY` is missing | Add the repository Actions secret, then dispatch again. |
| Claude cannot push or create a pull request | Confirm the Claude GitHub App is installed for this repository. |
| The generated-file policy fails | Remove application, test, harness, workflow, dependency, or unrelated-guide changes from the generated branch. |
| A command is not allowlisted | Add a narrow runner rule in a separate human-reviewed pull request; never let the generated branch modify the allowlist. |
| Fact verification fails | Re-run measurement and inspect the behavioral change before updating any displayed claim. |

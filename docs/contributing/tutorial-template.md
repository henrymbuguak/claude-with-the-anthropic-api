# Tutorial template

Copy this structure when starting a numbered guide. Remove sections that genuinely do not apply, but keep the learning sequence intact.

````text
# Build a specific outcome

## What you'll build

State what the reader produces and how long the tutorial takes.

## Before you begin

List software, configuration, API keys, and prerequisite guides.

## See it work

<!-- verify cmd tier=offline -->

```powershell
uv run python -m package.demo
```

<!-- verify expect match=contains -->

```text
Stable expected output
```

## How it works

Explain no more than the concepts required for the procedure.

## Build the feature

Add numbered steps. Indent every paragraph, annotation, and code block that
continues a numbered item with four spaces. The guide linter rejects three-space
continuations because MkDocs renders them as separate lists.

## Verify your work

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check path/to/file.py tests/test_feature.py
```

## Break it on purpose

Demonstrate one measured limitation that motivates the next guide.

## Troubleshooting

| Symptom                | Resolution        | Source    |
| ---------------------- | ----------------- | --------- |
| Exact observed message | Corrective action | Observed  |
| Predicted failure mode | Diagnostic action | Predicted |

## Next steps

Link the next guide and primary external documentation.

## Tested against

| Item                | Value            |
| ------------------- | ---------------- |
| Python              | Measured version |
| Repository revision | Commit SHA       |
| Verified            | YYYY-MM-DD       |
````

Run the annotation linter before requesting review:

```powershell
python -m harness.verify_guides "docs/guides/*.md" --execute --root .
```

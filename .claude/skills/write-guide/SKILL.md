---
name: write-guide
description: "Use when: drafting one planned tutorial from guides/plan.json into docs/guides with measured facts and executable verification annotations."
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(uv run:*)
---

# Write one planned guide

Write exactly one guide identified by the guide ID in the request. The repository's independent checks, not your judgment, decide whether the draft is valid.

## Read before writing

1. Read only the selected entry in `guides/plan.json` plus its prerequisite entries.
2. Confirm the selected entry has `status: "planned"` and `tier: "offline"`.
3. Read `docs/contributing/writing-guides.md` and `docs/contributing/tutorial-template.md`.
4. Read `docs/facts.json`, every file in `primary_code`, and every file in `tests` from the selected entry.
5. Read prerequisite guide pages only when needed to avoid reteaching their concepts.

Stop without editing if the guide is not planned, is keyed, or its declared output already exists.

## Allowed changes

Change only:

- The selected entry's declared `output` file.
- `mkdocs.yml`, adding one navigation entry for that output.
- The selected entry's `status`, from `planned` to `draft`.
- `docs/facts.json` only by running `uv run python -m harness.measure_facts --root . --write docs/facts.json`; never edit measured values manually.

Never modify application code, tests, harness code, workflow files, dependencies, lockfiles, existing guides, or any other curriculum entry.

## Authoring rules

- Follow the Google developer documentation style rules in the repository writing guide.
- Teach only the concepts declared in the selected entry; the default budget is three.
- Use the selected entry's descriptive output path. Never create a numbered `guide-N.md` filename.
- Use four spaces for every paragraph, annotation, and code block that continues a numbered item.
- Run every deterministic command before showing its output. Never invent output, versions, rankings, metrics, scores, errors, or API behavior.
- Wrap every shell block with `verify cmd` or `verify manual`. A manual annotation requires a specific reason.
- Use only offline commands. Do not access Anthropic, Voyage, or any other paid API.
- Wrap every displayed scalar from `docs/facts.json` in a `fact` annotation and declare it in the selected plan entry.
- Do not add a fact merely because it is interesting. Add only facts necessary to the lesson.
- Keep destructive build-from-empty steps manual. Never clear or overwrite source files in the checked-out repository.

## Verification sequence

Run these checks after writing:

```bash
uv run python -m harness.verify_facts --root .
uv run python -m harness.verify_guides "docs/guides/*.md"
uv run pytest -q
uv run ruff check .
```

The executable guide gate may reject a new command because it is not allowlisted. Do not weaken, bypass, or edit the allowlist. Leave the draft for a human to add a narrowly reviewed command rule.

Do not commit, push, open a pull request, or alter the generated branch. The workflow owns Git operations and changed-file enforcement.

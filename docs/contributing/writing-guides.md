# Write and review guides

This project uses one editorial standard and one information-architecture model:

- Follow the [Google developer documentation style guide](https://developers.google.com/style/) for wording, grammar, procedures, code examples, links, and accessibility.
- Use [Diátaxis](https://diataxis.fr/) to decide whether a page is a tutorial, how-to guide, explanation, or reference.
- Consult the [Microsoft Writing Style Guide](https://learn.microsoft.com/style-guide/welcome/) only for inclusive-language or accessibility questions that Google does not answer.

When guidance conflicts, Google style controls the wording. Do not blend sentence-level conventions from multiple style guides.

## Know the page type

| Type        | Reader need                       | Example in this project        |
| ----------- | --------------------------------- | ------------------------------ |
| Tutorial    | Learn through a guided experience | Implement BM25 from scratch    |
| How-to      | Complete a specific task          | Enable hybrid retrieval        |
| Explanation | Understand a design or concept    | Retrieval trust boundaries     |
| Reference   | Look up exact facts               | Environment variable reference |

The numbered guide series consists of tutorials. Keep reference tables and broad architecture explanations on separate pages, then link to them when needed.

## Write for this reader

Assume the reader:

- Writes Python and can use a command line.
- Has not built an application with a language model.
- Wants to understand the implementation, not only copy it.
- Uses Windows, macOS, or Linux.

Define a new AI term when it first appears. Introduce no more than three new concepts in one guide unless the curriculum records a deliberate exception.

## Follow the tutorial structure

Use these sections in order when they apply:

1. **What you'll build** states the observable outcome and estimated time.
2. **Before you begin** lists software, keys, and prerequisite guides.
3. **See it work** gives the reader a concrete result before the explanation.
4. **How it works** explains only the concepts needed for the procedure.
5. **Build the feature** presents ordered, testable steps.
6. **Verify your work** proves the result with focused checks.
7. **Break it on purpose** demonstrates one meaningful limitation.
8. **Troubleshooting** separates observed failures from predicted ones.
9. **Next steps** links the next guide and primary external documentation.
10. **Tested against** records versions and the verification date.

Use task-oriented titles such as “Implement Okapi BM25 from scratch,” not topic titles such as “About BM25.”

## Make procedures usable

- Address the reader as “you.”
- Use active voice and present tense.
- Begin each numbered step with an imperative verb.
- Put the file or location before the action.
- Keep one action in each numbered step.
- Explain the expected result immediately after the action.
- Avoid “easy,” “simply,” “just,” “obviously,” and directional phrases such as “below.”
- Use repository-relative paths in prose and output.
- Show PowerShell and Bash when commands differ.
- Keep secrets out of commands, output, screenshots, and fixtures.

## Account for every command

Every shell block in `docs/guides/` requires one annotation immediately before it.

Use a machine-checkable command when it is safe and deterministic:

````markdown
<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_bm25.py -q
```

<!-- verify expect match=contains -->

```text
4 passed
```
````

Set `output=none` only when success is represented entirely by the exit code:

````markdown
<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check app/rag/index_bm25.py
```
````

Mark destructive or inherently visual steps for human verification:

````markdown
<!-- verify manual reason="Modifies the temporary tutorial workspace" -->

```powershell
Clear-Content app/rag/index_bm25.py
```
````

A manual annotation is an explicit review obligation, not a passing automated check. Never run destructive tutorial steps in the repository checkout; future scenario verification will use a temporary workspace.

## Use verification tiers

- `offline` commands require no API key, network call, or paid service.
- `keyed` commands call Anthropic or Voyage and run only through an approved environment.

Do not require exact generated prose from a model. Verify deterministic structure, source IDs, configuration, or sanitized metadata instead.

## Review a guide

Confirm that:

- The learning outcome is observable.
- Every stated number or ranking was measured.
- Every shell block is verified or has a specific manual reason.
- The guide introduces only its declared concepts.
- Commands work from the repository root.
- Expected output excludes absolute paths, timing, and temporary names.
- Internal links resolve and the page appears in `mkdocs.yml`.
- `python -m harness.verify_guides "docs/guides/*.md"` passes.
- `mkdocs build --strict` passes in CI.

Machine verification establishes factual consistency. A human reviewer still decides whether the guide teaches clearly.

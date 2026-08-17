# Test model output with deterministic rules

## What you'll build

In this tutorial, you build the deterministic response checks used by this
repository's prompt-evaluation workflow. You detect banned hype terms, repeated
exclamation marks, and responses that exceed a fixed character limit, then
measure where those simple rules produce false positives and false negatives.

**Time:** About 25 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.
- [Benchmark retrieval quality](benchmark-retrieval-quality.md).

You do not need an API key. This tutorial covers only `run_rule_checks()` and
does not call the adjacent LLM-as-judge evaluator.

## See it work

Run the four focused rule-check tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_evaluators.py::test_run_rule_checks_passes_clean_response tests/test_evaluators.py::test_run_rule_checks_flags_hype_words tests/test_evaluators.py::test_run_rule_checks_flags_excessive_exclamation_marks tests/test_evaluators.py::test_run_rule_checks_flags_overly_long_response -v
```

<!-- verify expect match=contains -->

```text
test_run_rule_checks_passes_clean_response PASSED
test_run_rule_checks_flags_hype_words PASSED
test_run_rule_checks_flags_excessive_exclamation_marks PASSED
test_run_rule_checks_flags_overly_long_response PASSED
4 passed
```

The tests cover one clean response and one failure for each rule. They run
without network access and return the same result for the same text on every
run.

## How deterministic checks work

A **deterministic check** applies explicit logic instead of asking another model
for a judgment. The result depends only on the input text and the checked-in
rules. These checks are fast, inexpensive, reproducible, and easy to diagnose.

A deterministic rule does not understand intent. A **false positive** occurs
when a rule rejects acceptable text, such as a neutral quotation containing a
banned term. A **false negative** occurs when unacceptable text avoids the exact
patterns the rule knows, such as hype phrased with an unlisted expression.

The goal is not to encode every quality judgment as a string match. Use rules
for narrow requirements whose tradeoffs you can explain and test.

## Define the rule result

1. In `eval/evaluators.py`, define the banned terms and response limit:

    ```python
    BANNED_HYPE_WORDS = [
        "revolutionary",
        "game-changing",
        "unbelievable",
        "insane",
        "mind-blowing",
        "shocking",
        "you won't believe",
        "miracle",
    ]

    MAX_RESPONSE_CHARS = 4000
    ```

2. Add the result data class:

    ```python
    @dataclass
    class RuleCheckResult:
        passed: bool
        details: list[str]
    ```

    `passed` supports a simple gate. `details` preserves every reason for a
    failure so the evaluation report can explain what to change.

## Detect banned hype terms

1. Start `run_rule_checks()` with an empty detail list and a lowercase copy:

    ```python
    def run_rule_checks(response_text: str) -> RuleCheckResult:
        """Run deterministic checks that require no model call."""
        details: list[str] = []
        lower = response_text.lower()
    ```

2. Collect every banned term found in the response:

    ```python
        hype_hits = [word for word in BANNED_HYPE_WORDS if word in lower]
        if hype_hits:
            details.append(f"Contains hype words: {', '.join(hype_hits)}")
    ```

    Lowercasing makes the check case-insensitive. Collecting every match gives a
    writer more useful feedback than stopping at the first term.

## Check punctuation and length

1. Count exclamation marks and reject more than one:

    ```python
        exclamations = response_text.count("!")
        if exclamations > 1:
            details.append(
                f"Overuses exclamation marks ({exclamations} found)"
            )
    ```

    The threshold allows one exclamation mark while detecting repeated emphasis.

2. Compare response length with the fixed character limit:

    ```python
        if len(response_text) > MAX_RESPONSE_CHARS:
            details.append(
                f"Response too long ({len(response_text)} chars)"
            )
    ```

    A character limit is deterministic and easy to enforce, though it does not
    measure whether a shorter response is relevant or complete.

3. Return one result after every rule runs:

    ```python
        return RuleCheckResult(passed=not details, details=details)
    ```

    An empty detail list means every rule passed. Running all rules lets one
    response report multiple independent problems.

## Verify your work

Run the four focused tests without verbose output:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_evaluators.py::test_run_rule_checks_passes_clean_response tests/test_evaluators.py::test_run_rule_checks_flags_hype_words tests/test_evaluators.py::test_run_rule_checks_flags_excessive_exclamation_marks tests/test_evaluators.py::test_run_rule_checks_flags_overly_long_response -q
```

<!-- verify expect match=contains -->

```text
4 passed
```

Check the evaluator and tests with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check eval/evaluators.py tests/test_evaluators.py
```

## Break it on purpose

Run two responses that expose the limits of substring rules:

<!-- verify manual reason="Illustrative heuristic experiment uses an arbitrary Python snippet that is intentionally outside the command allowlist" -->

```powershell
uv run python -c "from eval.evaluators import run_rule_checks; print(run_rule_checks('The paper calls the prior method revolutionary.')); print(run_rule_checks('This result changes everything forever.'))"
```

The command prints:

```text
RuleCheckResult(passed=False, details=['Contains hype words: revolutionary'])
RuleCheckResult(passed=True, details=[])
```

The first response is a false positive: it reports another source's wording but
still contains the banned substring. The second response is a false negative:
its claim is promotional, but none of its words appears in the banned list.

Treat these cases as policy decisions. You can refine a rule when its expected
precision matters, but subjective qualities such as tone and evidence require a
separate evaluator rather than an ever-growing word list.

## Troubleshooting

| Symptom | Resolution | Source |
| --- | --- | --- |
| A clean quotation fails the hype check | Decide whether quotations need preprocessing or an explicit exception. | Observed |
| Promotional text passes | Add a narrowly justified term and a regression test, or use a subjective evaluator. | Observed |
| A response with one exclamation mark fails | Confirm the condition uses `> 1`, not `>= 1`. | Predicted |
| A 4,000-character response fails | Confirm the condition uses `> MAX_RESPONSE_CHARS`; exactly 4,000 characters should pass. | Predicted |
| Only one problem appears in `details` | Run every rule before constructing the result instead of returning early. | Predicted |

## Next steps

A planned LLM-as-judge tutorial evaluates subjective criteria such as active
voice, calm tone, and evidence grounding. Keep deterministic rules for explicit
constraints, and compare both evaluator types before adopting a prompt revision.

For guidance on evaluating generated text, read the
[Google Cloud generative AI evaluation overview](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview).

## Tested against

| Item | Value |
| --- | --- |
| Python | 3.13.14; the repository requires 3.12 or later |
| pytest | 9.1.1 |
| Verified | 2026-08-17 |

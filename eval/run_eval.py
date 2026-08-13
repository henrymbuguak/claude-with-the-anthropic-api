"""Prompt evaluation harness.

Runs a fixed set of test questions through a candidate system prompt, scores
each response with rule-based checks and an LLM-as-judge grader, and prints /
saves a summary report.

Usage:
    uv run eval/run_eval.py prompts/scientist_v1.txt
    uv run eval/run_eval.py prompts/scientist_v1.txt --cases eval/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from anthropic import Anthropic, omit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from eval.evaluators import (
    JudgeParseError,
    run_llm_judge,
    run_rule_checks,
)

DEFAULT_CASES_PATH = Path(__file__).parent / "cases.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class CaseResult:
    id: str
    input: str
    response: str
    thinking: str | None
    rule_checks_passed: bool
    rule_check_details: list[str]
    judge_active_voice: int | None
    judge_calm_tone: int | None
    judge_evidence_grounded: int | None
    judge_average: float | None
    judge_justification: str | None
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(prompt_path: Path, cases_path: Path) -> list[CaseResult]:
    settings = Settings.from_env()
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    client = Anthropic(api_key=settings.api_key, max_retries=3)

    # Extended thinking is configured once in Settings and shared by the CLI
    # (app/claude_client.py) and this eval harness, so a prompt is always
    # evaluated under the same thinking behavior it runs with in the chatbot.
    thinking = (
        {
            "type": "enabled",
            "budget_tokens": settings.thinking_budget_tokens,
            "display": "summarized",
        }
        if settings.thinking_enabled
        else omit
    )
    temperature = omit if settings.thinking_enabled else settings.temperature

    # All cases share this exact system prompt, so once prompt caching is
    # enabled the first case writes the cache and later cases in the same
    # run can read from it.
    system = (
        [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if settings.prompt_cache_enabled
        else system_prompt
    )

    results: list[CaseResult] = []
    for case in load_cases(cases_path):
        message = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": case["input"]}],
            thinking=thinking,
        )
        response_text = "".join(
            block.text for block in message.content if block.type == "text")
        thinking_text = "\n\n".join(
            block.thinking
            for block in message.content
            if block.type == "thinking" and block.thinking
        ) or None

        rule_result = run_rule_checks(response_text)

        try:
            judge = run_llm_judge(client, settings.model,
                                  case["input"], response_text)
            judge_fields = {
                "judge_active_voice": judge.active_voice,
                "judge_calm_tone": judge.calm_tone,
                "judge_evidence_grounded": judge.evidence_grounded,
                "judge_average": judge.average,
                "judge_justification": judge.justification,
            }
        except JudgeParseError as exc:
            judge_fields = {
                "judge_active_voice": None,
                "judge_calm_tone": None,
                "judge_evidence_grounded": None,
                "judge_average": None,
                "judge_justification": f"Judge error: {exc}",
            }

        results.append(
            CaseResult(
                id=case["id"],
                input=case["input"],
                response=response_text,
                thinking=thinking_text,
                rule_checks_passed=rule_result.passed,
                rule_check_details=rule_result.details,
                cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
                **judge_fields,
            )
        )

    return results


def print_summary(prompt_path: Path, results: list[CaseResult]) -> None:
    print(f"\nPrompt: {prompt_path}")
    if any(r.thinking for r in results):
        print("Extended thinking: enabled (reasoning saved in the JSON report)")
    total_cache_read = sum(r.cache_read_input_tokens for r in results)
    if total_cache_read:
        print(f"Prompt cache: {total_cache_read} tokens read from cache")
    print(f"{'ID':<16} {'Rules':<7} {'Active':<7} {'Calm':<6} {'Evid.':<6} {'Avg':<6}")
    for r in results:
        rules = "PASS" if r.rule_checks_passed else "FAIL"
        av = r.judge_active_voice if r.judge_active_voice is not None else "-"
        ct = r.judge_calm_tone if r.judge_calm_tone is not None else "-"
        eg = r.judge_evidence_grounded if r.judge_evidence_grounded is not None else "-"
        avg = f"{r.judge_average:.1f}" if r.judge_average is not None else "-"
        print(f"{r.id:<16} {rules:<7} {av!s:<7} {ct!s:<6} {eg!s:<6} {avg:<6}")

    scored = [r for r in results if r.judge_average is not None]
    if scored:
        overall_avg = sum(r.judge_average for r in scored) / len(scored)
        print(f"\nOverall judge average: {overall_avg:.2f}/5")

    rule_pass_rate = sum(
        r.rule_checks_passed for r in results) / len(results) * 100
    print(f"Rule check pass rate: {rule_pass_rate:.0f}%")

    for r in results:
        if not r.rule_checks_passed:
            print(f"  [{r.id}] rule failures: {r.rule_check_details}")


def save_report(prompt_path: Path, results: list[CaseResult]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"{prompt_path.stem}_{timestamp}.json"
    payload = {
        "prompt_file": str(prompt_path),
        "timestamp": timestamp,
        "results": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a system prompt against a fixed test set.")
    parser.add_argument("prompt_file", type=Path,
                        help="Path to a system prompt text file")
    parser.add_argument(
        "--cases", type=Path, default=DEFAULT_CASES_PATH, help="Path to eval cases (JSONL)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write a JSON report to eval/results/")
    args = parser.parse_args()

    results = run(args.prompt_file, args.cases)
    print_summary(args.prompt_file, results)

    if not args.no_save:
        report_path = save_report(args.prompt_file, results)
        print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()

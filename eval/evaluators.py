"""Evaluators for scoring Claude's responses against the system prompt's rubric.

Two complementary evaluator types:
- Rule-based checks: cheap, deterministic heuristics (banned words, length,
  punctuation) that don't require another model call.
- LLM-as-judge: asks Claude itself to score subjective criteria (tone, voice,
  evidence-grounding) that regex can't reliably detect.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anthropic import Anthropic

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


@dataclass
class RuleCheckResult:
    passed: bool
    details: list[str]


def run_rule_checks(response_text: str) -> RuleCheckResult:
    """Cheap, deterministic checks that don't require another model call."""
    details: list[str] = []
    lower = response_text.lower()

    hype_hits = [word for word in BANNED_HYPE_WORDS if word in lower]
    if hype_hits:
        details.append(f"Contains hype words: {', '.join(hype_hits)}")

    exclamations = response_text.count("!")
    if exclamations > 1:
        details.append(f"Overuses exclamation marks ({exclamations} found)")

    if len(response_text) > MAX_RESPONSE_CHARS:
        details.append(f"Response too long ({len(response_text)} chars)")

    return RuleCheckResult(passed=not details, details=details)


JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator. You will be shown a user question and an AI \
assistant's response. Score the response from 1 (poor) to 5 (excellent) on \
each of the following criteria, then give a one-sentence justification.

Criteria:
- active_voice: Does the response consistently use active voice, avoiding \
passive constructions except where the actor is genuinely unknown?
- calm_tone: Is the tone calm, measured, and free of hype or alarmism?
- evidence_grounded: Are claims grounded in evidence/reasoning rather than \
speculation, with uncertainty acknowledged where appropriate?

Respond with ONLY a JSON object in this exact shape, no other text:
{"active_voice": <1-5>, "calm_tone": <1-5>, "evidence_grounded": <1-5>, "justification": "<one sentence>"}
"""


class JudgeParseError(RuntimeError):
    """Raised when the judge model's output isn't valid JSON in the expected shape."""


@dataclass
class JudgeScore:
    active_voice: int
    calm_tone: int
    evidence_grounded: int
    justification: str

    @property
    def average(self) -> float:
        return (self.active_voice + self.calm_tone + self.evidence_grounded) / 3


def run_llm_judge(client: Anthropic, model: str, question: str, response_text: str) -> JudgeScore:
    """Ask Claude to grade another response against the rubric (LLM-as-judge)."""
    judge_input = f"User question:\n{question}\n\nAssistant response:\n{response_text}"

    message = client.messages.create(
        model=model,
        max_tokens=300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": judge_input}],
    )
    raw = "".join(
        block.text for block in message.content if block.type == "text").strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise JudgeParseError(f"Judge did not return JSON: {raw!r}")

    try:
        data = json.loads(match.group(0))
        return JudgeScore(
            active_voice=int(data["active_voice"]),
            calm_tone=int(data["calm_tone"]),
            evidence_grounded=int(data["evidence_grounded"]),
            justification=str(data["justification"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise JudgeParseError(
            f"Could not parse judge output: {raw!r}") from exc

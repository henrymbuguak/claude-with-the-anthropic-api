"""Tests for eval.evaluators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval.evaluators import (
    JudgeParseError,
    JudgeScore,
    run_llm_judge,
    run_rule_checks,
)


def test_run_rule_checks_passes_clean_response() -> None:
    result = run_rule_checks(
        "Water boils at 100 degrees Celsius at sea level.")

    assert result.passed is True
    assert result.details == []


def test_run_rule_checks_flags_hype_words() -> None:
    result = run_rule_checks(
        "This is a revolutionary, mind-blowing discovery.")

    assert result.passed is False
    assert any("hype words" in detail for detail in result.details)


def test_run_rule_checks_flags_excessive_exclamation_marks() -> None:
    result = run_rule_checks("Wow! This is great! Amazing!")

    assert result.passed is False
    assert any("exclamation" in detail for detail in result.details)


def test_run_rule_checks_flags_overly_long_response() -> None:
    result = run_rule_checks("a" * 4001)

    assert result.passed is False
    assert any("too long" in detail for detail in result.details)


def _make_client(raw_response_text: str) -> SimpleNamespace:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=raw_response_text)])
    messages = SimpleNamespace(create=lambda **kwargs: message)
    return SimpleNamespace(messages=messages)


def test_run_llm_judge_parses_valid_json_response() -> None:
    raw = (
        '{"active_voice": 5, "calm_tone": 4, "evidence_grounded": 3, '
        '"justification": "Solid answer."}'
    )
    client = _make_client(raw)

    score = run_llm_judge(client, "claude-test",
                          "Why is the sky blue?", "Rayleigh scattering.")

    assert score == JudgeScore(
        active_voice=5, calm_tone=4, evidence_grounded=3, justification="Solid answer."
    )
    assert score.average == pytest.approx(4.0)


def test_run_llm_judge_extracts_json_surrounded_by_extra_text() -> None:
    raw = 'Here is my score: {"active_voice": 1, "calm_tone": 1, "evidence_grounded": 1, "justification": "Poor."} Thanks!'
    client = _make_client(raw)

    score = run_llm_judge(client, "claude-test", "Q", "A")

    assert score.active_voice == 1
    assert score.justification == "Poor."


def test_run_llm_judge_raises_on_non_json_response() -> None:
    client = _make_client("I refuse to answer in JSON.")

    with pytest.raises(JudgeParseError):
        run_llm_judge(client, "claude-test", "Q", "A")


def test_run_llm_judge_raises_on_missing_fields() -> None:
    client = _make_client('{"active_voice": 5}')

    with pytest.raises(JudgeParseError):
        run_llm_judge(client, "claude-test", "Q", "A")

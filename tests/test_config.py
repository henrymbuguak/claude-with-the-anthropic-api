"""Tests for app.config.Settings."""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from a known-empty environment for the settings we care about."""
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_MAX_TOKENS",
        "ANTHROPIC_TEMPERATURE",
        "ANTHROPIC_SYSTEM_PROMPT_FILE",
        "ANTHROPIC_SYSTEM_PROMPT",
        "LOG_LEVEL",
        "CONVERSATION_HISTORY_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_from_env_raises_when_api_key_missing() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        Settings.from_env()


def test_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.model == "claude-sonnet-4-6"
    assert settings.max_tokens == 500
    assert settings.temperature == 1.0
    assert settings.system_prompt is None
    assert settings.log_level == "INFO"
    assert settings.history_file == "conversation_history.json"


def test_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "1024")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.5")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("CONVERSATION_HISTORY_FILE", "history.json")

    settings = Settings.from_env()

    assert settings.model == "claude-test-model"
    assert settings.max_tokens == 1024
    assert settings.temperature == 0.5
    assert settings.log_level == "DEBUG"
    assert settings.history_file == "history.json"


def test_from_env_rejects_non_integer_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "not-a-number")

    with pytest.raises(ConfigError, match="ANTHROPIC_MAX_TOKENS"):
        Settings.from_env()


def test_from_env_rejects_non_numeric_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "not-a-number")

    with pytest.raises(ConfigError, match="ANTHROPIC_TEMPERATURE"):
        Settings.from_env()


def test_system_prompt_from_inline_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_SYSTEM_PROMPT", "Be concise.")

    settings = Settings.from_env()

    assert settings.system_prompt == "Be concise."


def test_system_prompt_file_takes_precedence_over_inline(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a helpful scientist.\n", encoding="utf-8")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_SYSTEM_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("ANTHROPIC_SYSTEM_PROMPT", "This should be ignored.")

    settings = Settings.from_env()

    assert settings.system_prompt == "You are a helpful scientist."


def test_system_prompt_file_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_SYSTEM_PROMPT_FILE",
                       str(tmp_path / "missing.txt"))

    with pytest.raises(ConfigError, match="missing file"):
        Settings.from_env()

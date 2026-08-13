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
        "ANTHROPIC_WEB_SEARCH_ENABLED",
        "ANTHROPIC_WEB_SEARCH_MAX_USES",
        "ANTHROPIC_WEB_SEARCH_ALLOWED_DOMAINS",
        "ANTHROPIC_THINKING_ENABLED",
        "ANTHROPIC_THINKING_BUDGET_TOKENS",
        "ANTHROPIC_PROMPT_CACHE_ENABLED",
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
    assert settings.web_search_enabled is False
    assert settings.web_search_max_uses == 3
    assert settings.web_search_allowed_domains is None
    assert settings.thinking_enabled is False
    assert settings.thinking_budget_tokens == 10000
    assert settings.prompt_cache_enabled is False


def test_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "1024")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.5")
    monkeypatch.setenv("ANTHROPIC_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_WEB_SEARCH_MAX_USES", "5")
    monkeypatch.setenv(
        "ANTHROPIC_WEB_SEARCH_ALLOWED_DOMAINS", " WHO.INT, cdc.gov ")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("CONVERSATION_HISTORY_FILE", "history.json")

    settings = Settings.from_env()

    assert settings.model == "claude-test-model"
    assert settings.max_tokens == 1024
    assert settings.temperature == 0.5
    assert settings.web_search_enabled is True
    assert settings.web_search_max_uses == 5
    assert settings.web_search_allowed_domains == ["who.int", "cdc.gov"]
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


@pytest.mark.parametrize("value", ["yes", "1", "enabled"])
def test_from_env_rejects_invalid_web_search_boolean(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WEB_SEARCH_ENABLED", value)

    with pytest.raises(ConfigError, match="ANTHROPIC_WEB_SEARCH_ENABLED"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_from_env_rejects_invalid_web_search_max_uses(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WEB_SEARCH_MAX_USES", value)

    with pytest.raises(ConfigError, match="ANTHROPIC_WEB_SEARCH_MAX_USES"):
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


def test_from_env_enables_thinking_with_valid_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "16000")
    monkeypatch.setenv("ANTHROPIC_THINKING_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "10000")

    settings = Settings.from_env()

    assert settings.thinking_enabled is True
    assert settings.thinking_budget_tokens == 10000


@pytest.mark.parametrize("value", ["yes", "1", "enabled"])
def test_from_env_rejects_invalid_thinking_boolean(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_THINKING_ENABLED", value)

    with pytest.raises(ConfigError, match="ANTHROPIC_THINKING_ENABLED"):
        Settings.from_env()


def test_from_env_rejects_non_integer_thinking_budget_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "not-a-number")

    with pytest.raises(ConfigError, match="ANTHROPIC_THINKING_BUDGET_TOKENS"):
        Settings.from_env()


def test_from_env_rejects_thinking_budget_below_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "16000")
    monkeypatch.setenv("ANTHROPIC_THINKING_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "100")

    with pytest.raises(ConfigError, match="at least 1024"):
        Settings.from_env()


def test_from_env_rejects_thinking_budget_not_less_than_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "5000")
    monkeypatch.setenv("ANTHROPIC_THINKING_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "5000")

    with pytest.raises(ConfigError, match="ANTHROPIC_MAX_TOKENS must be greater than"):
        Settings.from_env()


def test_from_env_rejects_custom_temperature_with_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "16000")
    monkeypatch.setenv("ANTHROPIC_THINKING_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "10000")
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.5")

    with pytest.raises(ConfigError, match="ANTHROPIC_TEMPERATURE cannot be customized"):
        Settings.from_env()


def test_from_env_enables_prompt_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.prompt_cache_enabled is True


@pytest.mark.parametrize("value", ["yes", "1", "enabled"])
def test_from_env_rejects_invalid_prompt_cache_boolean(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE_ENABLED", value)

    with pytest.raises(ConfigError, match="ANTHROPIC_PROMPT_CACHE_ENABLED"):
        Settings.from_env()

"""Tests for the native Anthropic web-search definition."""

from app.config import Settings
from app.tools.web_search import build_web_search_tool


def make_settings(**overrides) -> Settings:
    defaults = {
        "api_key": "test-key",
        "model": "claude-test-model",
        "max_tokens": 100,
        "temperature": 1.0,
        "system_prompt": None,
        "log_level": "INFO",
        "history_file": "history.json",
        "web_search_enabled": True,
        "web_search_max_uses": 3,
        "web_search_allowed_domains": None,
        "thinking_enabled": False,
        "thinking_budget_tokens": 10000,
        "prompt_cache_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_build_web_search_tool_uses_configured_limit() -> None:
    tool = build_web_search_tool(make_settings(web_search_max_uses=4))

    assert tool == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 4,
    }


def test_build_web_search_tool_includes_allowed_domains() -> None:
    tool = build_web_search_tool(
        make_settings(web_search_allowed_domains=["who.int", "cdc.gov"])
    )

    assert tool["allowed_domains"] == ["who.int", "cdc.gov"]

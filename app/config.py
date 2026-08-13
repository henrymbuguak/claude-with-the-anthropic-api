"""Application configuration loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


MIN_THINKING_BUDGET_TOKENS = 1024


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the chat CLI, sourced entirely from the environment."""

    api_key: str
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str | None
    log_level: str
    history_file: str
    web_search_enabled: bool
    web_search_max_uses: int
    web_search_allowed_domains: list[str] | None
    thinking_enabled: bool
    thinking_budget_tokens: int
    prompt_cache_enabled: bool

    @classmethod
    def from_env(cls) -> Settings:
        """Build validated settings from environment variables, failing fast on errors."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        max_tokens_raw = os.getenv("ANTHROPIC_MAX_TOKENS", "500")
        temperature_raw = os.getenv("ANTHROPIC_TEMPERATURE", "1.0")
        web_search_enabled_raw = os.getenv(
            "ANTHROPIC_WEB_SEARCH_ENABLED", "false").strip().lower()
        web_search_max_uses_raw = os.getenv(
            "ANTHROPIC_WEB_SEARCH_MAX_USES", "3")

        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ConfigError(
                f"ANTHROPIC_MAX_TOKENS must be an integer, got {max_tokens_raw!r}"
            ) from exc

        try:
            temperature = float(temperature_raw)
        except ValueError as exc:
            raise ConfigError(
                f"ANTHROPIC_TEMPERATURE must be a number, got {temperature_raw!r}"
            ) from exc

        if web_search_enabled_raw not in {"true", "false"}:
            raise ConfigError(
                "ANTHROPIC_WEB_SEARCH_ENABLED must be 'true' or 'false', "
                f"got {web_search_enabled_raw!r}"
            )

        try:
            web_search_max_uses = int(web_search_max_uses_raw)
        except ValueError as exc:
            raise ConfigError(
                "ANTHROPIC_WEB_SEARCH_MAX_USES must be an integer, "
                f"got {web_search_max_uses_raw!r}"
            ) from exc
        if web_search_max_uses < 1:
            raise ConfigError(
                "ANTHROPIC_WEB_SEARCH_MAX_USES must be greater than zero"
            )

        allowed_domains = [
            domain.strip().lower()
            for domain in os.getenv(
                "ANTHROPIC_WEB_SEARCH_ALLOWED_DOMAINS", "").split(",")
            if domain.strip()
        ]

        thinking_enabled_raw = os.getenv(
            "ANTHROPIC_THINKING_ENABLED", "false").strip().lower()
        if thinking_enabled_raw not in {"true", "false"}:
            raise ConfigError(
                "ANTHROPIC_THINKING_ENABLED must be 'true' or 'false', "
                f"got {thinking_enabled_raw!r}"
            )
        thinking_enabled = thinking_enabled_raw == "true"

        thinking_budget_tokens_raw = os.getenv(
            "ANTHROPIC_THINKING_BUDGET_TOKENS", "10000")
        try:
            thinking_budget_tokens = int(thinking_budget_tokens_raw)
        except ValueError as exc:
            raise ConfigError(
                "ANTHROPIC_THINKING_BUDGET_TOKENS must be an integer, "
                f"got {thinking_budget_tokens_raw!r}"
            ) from exc

        if thinking_enabled:
            if thinking_budget_tokens < MIN_THINKING_BUDGET_TOKENS:
                raise ConfigError(
                    "ANTHROPIC_THINKING_BUDGET_TOKENS must be at least "
                    f"{MIN_THINKING_BUDGET_TOKENS} when extended thinking is "
                    f"enabled, got {thinking_budget_tokens}"
                )
            if thinking_budget_tokens >= max_tokens:
                raise ConfigError(
                    "ANTHROPIC_MAX_TOKENS must be greater than "
                    "ANTHROPIC_THINKING_BUDGET_TOKENS so there is room left "
                    f"for the final response (max_tokens={max_tokens}, "
                    f"thinking_budget_tokens={thinking_budget_tokens})"
                )
            if temperature != 1.0:
                raise ConfigError(
                    "ANTHROPIC_TEMPERATURE cannot be customized while "
                    "extended thinking is enabled - Claude requires its "
                    "default sampling temperature for thinking requests. "
                    "Remove ANTHROPIC_TEMPERATURE or set "
                    "ANTHROPIC_THINKING_ENABLED=false."
                )

        prompt_cache_enabled_raw = os.getenv(
            "ANTHROPIC_PROMPT_CACHE_ENABLED", "false").strip().lower()
        if prompt_cache_enabled_raw not in {"true", "false"}:
            raise ConfigError(
                "ANTHROPIC_PROMPT_CACHE_ENABLED must be 'true' or 'false', "
                f"got {prompt_cache_enabled_raw!r}"
            )
        prompt_cache_enabled = prompt_cache_enabled_raw == "true"

        system_prompt = cls._load_system_prompt()

        return cls(
            api_key=api_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            history_file=os.getenv(
                "CONVERSATION_HISTORY_FILE", "conversation_history.json"),
            web_search_enabled=web_search_enabled_raw == "true",
            web_search_max_uses=web_search_max_uses,
            web_search_allowed_domains=allowed_domains or None,
            thinking_enabled=thinking_enabled,
            thinking_budget_tokens=thinking_budget_tokens,
            prompt_cache_enabled=prompt_cache_enabled,
        )

    @staticmethod
    def _load_system_prompt() -> str | None:
        """Prefer a versioned prompt file (ANTHROPIC_SYSTEM_PROMPT_FILE) so prompts
        can be tracked, diffed, and run through the eval harness. Falls back to the
        inline ANTHROPIC_SYSTEM_PROMPT env var for simple setups."""
        prompt_file = os.getenv("ANTHROPIC_SYSTEM_PROMPT_FILE", "").strip()
        if prompt_file:
            path = Path(prompt_file)
            if not path.exists():
                raise ConfigError(
                    f"ANTHROPIC_SYSTEM_PROMPT_FILE points to a missing file: {path}"
                )
            return path.read_text(encoding="utf-8").strip()

        return os.getenv("ANTHROPIC_SYSTEM_PROMPT") or None

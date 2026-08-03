"""Application configuration loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


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

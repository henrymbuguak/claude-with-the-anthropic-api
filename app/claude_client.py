"""Thin wrapper around the Anthropic Messages API with retries and friendly errors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    omit,
)

from app.config import Settings


class ChatClientError(RuntimeError):
    """Raised when the Anthropic API call fails."""


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int


class ClaudeChatClient:
    """Wraps the Anthropic SDK client with config-driven defaults and error handling."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Anthropic(api_key=settings.api_key, max_retries=3)

    def stream_message(
        self,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        """Send the conversation to Claude, streaming text chunks to `on_chunk` as they arrive."""
        try:
            with self._client.messages.stream(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
                system=self._settings.system_prompt or omit,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    if on_chunk is not None:
                        on_chunk(text)
                final_message = stream.get_final_message()
        except RateLimitError as exc:
            raise ChatClientError(
                "Rate limit reached. Wait a moment and try again.") from exc
        except APIConnectionError as exc:
            raise ChatClientError(
                "Could not reach the Anthropic API. Check your network connection."
            ) from exc
        except APIStatusError as exc:
            raise ChatClientError(
                f"Anthropic API returned an error ({exc.status_code}): {exc.message}"
            ) from exc

        return ChatResponse(
            text="".join(
                block.text for block in final_message.content if block.type == "text"),
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
        )

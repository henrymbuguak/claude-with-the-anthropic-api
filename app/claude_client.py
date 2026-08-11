"""Thin wrapper around the Anthropic Messages API with retries and friendly errors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    omit,
)

from app.config import Settings
from app.tools.web_search import build_web_search_tool


class ChatClientError(RuntimeError):
    """Raised when the Anthropic API call fails."""


@dataclass
class Citation:
    url: str
    title: str | None
    cited_text: str | None


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    citations: list[Citation] = field(default_factory=list)
    web_search_requests: int = 0


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
        tools = (
            [build_web_search_tool(self._settings)]
            if self._settings.web_search_enabled
            else omit
        )
        try:
            with self._client.messages.stream(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
                system=self._settings.system_prompt or omit,
                messages=messages,
                tools=tools,
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

        citations: list[Citation] = []
        seen_urls: set[str] = set()
        for block in final_message.content:
            if block.type != "text":
                continue
            for citation in block.citations or []:
                if citation.type != "web_search_result_location":
                    continue
                if citation.url in seen_urls:
                    continue
                seen_urls.add(citation.url)
                citations.append(
                    Citation(
                        url=citation.url,
                        title=citation.title,
                        cited_text=citation.cited_text,
                    )
                )

        server_tool_use = final_message.usage.server_tool_use
        return ChatResponse(
            text="".join(
                block.text for block in final_message.content if block.type == "text"),
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            citations=citations,
            web_search_requests=(
                server_tool_use.web_search_requests if server_tool_use else 0
            ),
        )

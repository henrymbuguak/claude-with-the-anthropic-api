"""Tests for app.claude_client.ClaudeChatClient."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Self

import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, RateLimitError, omit

from app.claude_client import ChatClientError, Citation, ClaudeChatClient
from app.config import Settings


def make_settings(**overrides) -> Settings:
    defaults = {
        "api_key": "test-key",
        "model": "claude-test-model",
        "max_tokens": 100,
        "temperature": 1.0,
        "system_prompt": None,
        "log_level": "INFO",
        "history_file": "history.json",
        "web_search_enabled": False,
        "web_search_max_uses": 3,
        "web_search_allowed_domains": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_final_message(
    text: str,
    input_tokens: int = 10,
    output_tokens: int = 20,
    citations=None,
    web_search_requests: int = 0,
):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text, citations=citations)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            server_tool_use=(
                SimpleNamespace(web_search_requests=web_search_requests)
                if web_search_requests
                else None
            ),
        ),
    )


class FakeStream:
    """Mimics the context manager returned by `client.messages.stream(...)`."""

    def __init__(self, chunks: list[str], final_message) -> None:
        self._chunks = chunks
        self._final_message = final_message

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return self._final_message


class FakeMessages:
    def __init__(self, stream_result=None, stream_error: Exception | None = None) -> None:
        self._stream_result = stream_result
        self._stream_error = stream_error
        self.last_kwargs: dict | None = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        if self._stream_error is not None:
            raise self._stream_error
        return self._stream_result


def build_client(settings: Settings, messages: FakeMessages, monkeypatch: pytest.MonkeyPatch) -> ClaudeChatClient:
    fake_anthropic_client = SimpleNamespace(messages=messages)
    monkeypatch.setattr(
        "app.claude_client.Anthropic", lambda *args, **kwargs: fake_anthropic_client
    )
    return ClaudeChatClient(settings)


def test_stream_message_returns_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    final_message = make_final_message(
        "Hello, world!", input_tokens=5, output_tokens=7)
    messages = FakeMessages(stream_result=FakeStream(
        ["Hello, ", "world!"], final_message))
    client = build_client(make_settings(), messages, monkeypatch)

    chunks: list[str] = []
    response = client.stream_message(
        [{"role": "user", "content": "Hi"}], on_chunk=chunks.append)

    assert chunks == ["Hello, ", "world!"]
    assert response.text == "Hello, world!"
    assert response.input_tokens == 5
    assert response.output_tokens == 7
    assert response.citations == []
    assert response.web_search_requests == 0


def test_stream_message_omits_tools_when_web_search_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_message = make_final_message("ok")
    messages = FakeMessages(stream_result=FakeStream(["ok"], final_message))
    client = build_client(make_settings(
        web_search_enabled=False), messages, monkeypatch)

    client.stream_message([{"role": "user", "content": "Hi"}])

    assert messages.last_kwargs["tools"] is omit


def test_stream_message_enables_configured_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_message = make_final_message("ok")
    messages = FakeMessages(stream_result=FakeStream(["ok"], final_message))
    client = build_client(
        make_settings(
            web_search_enabled=True,
            web_search_max_uses=2,
            web_search_allowed_domains=["who.int"],
        ),
        messages,
        monkeypatch,
    )

    client.stream_message([{"role": "user", "content": "Latest guidance?"}])

    assert messages.last_kwargs["tools"] == [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 2,
            "allowed_domains": ["who.int"],
        }
    ]


def test_stream_message_returns_unique_web_citations_and_search_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citation = SimpleNamespace(
        type="web_search_result_location",
        url="https://example.com/report",
        title="Example report",
        cited_text="Relevant evidence",
    )
    final_message = make_final_message(
        "Grounded answer",
        citations=[citation, citation],
        web_search_requests=2,
    )
    messages = FakeMessages(
        stream_result=FakeStream(["Grounded answer"], final_message)
    )
    client = build_client(make_settings(
        web_search_enabled=True), messages, monkeypatch)

    response = client.stream_message(
        [{"role": "user", "content": "Research this"}])

    assert response.citations == [
        Citation(
            url="https://example.com/report",
            title="Example report",
            cited_text="Relevant evidence",
        )
    ]
    assert response.web_search_requests == 2


def test_stream_message_works_without_on_chunk_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    final_message = make_final_message("Answer")
    messages = FakeMessages(
        stream_result=FakeStream(["Answer"], final_message))
    client = build_client(make_settings(), messages, monkeypatch)

    response = client.stream_message([{"role": "user", "content": "Hi"}])

    assert response.text == "Answer"


def test_stream_message_omits_system_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    final_message = make_final_message("ok")
    messages = FakeMessages(stream_result=FakeStream(["ok"], final_message))
    client = build_client(make_settings(
        system_prompt=None), messages, monkeypatch)

    client.stream_message([{"role": "user", "content": "Hi"}])

    assert messages.last_kwargs["system"] is omit


def test_stream_message_passes_configured_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    final_message = make_final_message("ok")
    messages = FakeMessages(stream_result=FakeStream(["ok"], final_message))
    client = build_client(make_settings(
        system_prompt="Be terse."), messages, monkeypatch)

    client.stream_message([{"role": "user", "content": "Hi"}])

    assert messages.last_kwargs["system"] == "Be terse."


def test_stream_message_wraps_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    error = RateLimitError("slow down", response=response, body=None)
    messages = FakeMessages(stream_error=error)
    client = build_client(make_settings(), messages, monkeypatch)

    with pytest.raises(ChatClientError, match="Rate limit"):
        client.stream_message([{"role": "user", "content": "Hi"}])


def test_stream_message_wraps_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = APIConnectionError(request=request)
    messages = FakeMessages(stream_error=error)
    client = build_client(make_settings(), messages, monkeypatch)

    with pytest.raises(ChatClientError, match="Could not reach"):
        client.stream_message([{"role": "user", "content": "Hi"}])


def test_stream_message_wraps_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    error = APIStatusError("server exploded", response=response, body=None)
    messages = FakeMessages(stream_error=error)
    client = build_client(make_settings(), messages, monkeypatch)

    with pytest.raises(ChatClientError, match="500"):
        client.stream_message([{"role": "user", "content": "Hi"}])

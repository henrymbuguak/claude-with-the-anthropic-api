"""Tests for CLI response formatting and the RAG-augmented chat loop."""

from pathlib import Path

import pytest

from app.claude_client import ChatResponse, Citation
from app.conversation import Conversation
from app.rag.config import RagSettings
from app.rag.models import Chunk, SearchResult
from main import format_local_sources, format_sources, run_chat_loop


def test_format_sources_returns_empty_string_without_citations() -> None:
    assert format_sources([]) == ""


def test_format_sources_numbers_titles_and_falls_back_to_url() -> None:
    citations = [
        Citation(
            url="https://example.com/report",
            title="Example report",
            cited_text="Evidence",
        ),
        Citation(
            url="https://example.org/data",
            title=None,
            cited_text=None,
        ),
    ]

    assert format_sources(citations) == (
        "Sources:\n"
        "[1] Example report - https://example.com/report\n"
        "[2] https://example.org/data - https://example.org/data"
    )


def test_format_local_sources_returns_empty_string_without_results() -> None:
    assert format_local_sources([]) == ""


def test_format_local_sources_numbers_chunk_ids() -> None:
    results = [
        SearchResult(chunk=Chunk("one", "text", "one.py", "code"),
                     score=1.0, rank=1),
        SearchResult(chunk=Chunk("two", "text", "two.py", "code"),
                     score=0.9, rank=2),
    ]

    assert format_local_sources(results) == "Local sources:\n[1] one\n[2] two"


class FakeClient:
    def __init__(self, response: ChatResponse) -> None:
        self.response = response
        self.received_messages: list[dict] | None = None

    def stream_message(self, messages, on_chunk=None):
        self.received_messages = list(messages)
        if on_chunk:
            on_chunk(self.response.text)
        return self.response


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query, *, mode, top_k):
        self.queries.append(query)
        return self.results


def make_rag_settings(**overrides: object) -> RagSettings:
    defaults: dict[str, object] = {
        "voyage_api_key": None,
        "enabled": True,
        "chat_mode": "bm25",
        "chat_top_k": 5,
        "context_char_budget": 6000,
    }
    defaults.update(overrides)
    return RagSettings(**defaults)


def test_run_chat_loop_augments_api_call_but_persists_only_the_plain_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = iter(["How is history saved?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    response = ChatResponse(text="It is saved as JSON.",
                            input_tokens=10, output_tokens=5)
    client = FakeClient(response)
    conversation = Conversation()
    retriever = FakeRetriever(
        [
            SearchResult(
                chunk=Chunk(
                    "app/conversation.py#class-Conversation",
                    "class Conversation: ...",
                    "app/conversation.py",
                    "code",
                ),
                score=1.0,
                rank=1,
            )
        ]
    )
    rag_settings = make_rag_settings()
    history_path = tmp_path / "history.json"

    run_chat_loop(client, conversation, history_path, retriever, rag_settings)

    assert conversation.messages == [
        {"role": "user", "content": "How is history saved?"},
        {"role": "assistant", "content": "It is saved as JSON."},
    ]
    sent_content = client.received_messages[-1]["content"]
    assert sent_content != "How is history saved?"
    assert "How is history saved?" in sent_content
    assert "app/conversation.py#class-Conversation" in sent_content
    assert retriever.queries == ["How is history saved?"]


def test_run_chat_loop_without_retriever_sends_plain_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = iter(["Hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    response = ChatResponse(text="Hi there.", input_tokens=1, output_tokens=1)
    client = FakeClient(response)
    conversation = Conversation()
    history_path = tmp_path / "history.json"

    run_chat_loop(client, conversation, history_path)

    assert client.received_messages == [{"role": "user", "content": "Hello"}]
    assert conversation.messages[-1] == {
        "role": "assistant", "content": "Hi there."}

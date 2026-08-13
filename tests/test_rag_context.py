"""Tests for building temporary, citation-grounded chat context."""

from app.rag.context import build_augmented_message, build_context_block
from app.rag.models import Chunk, SearchResult


def make_result(chunk_id: str, text: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(chunk_id, text, chunk_id, "code"), score=1.0, rank=rank
    )


def test_build_context_block_includes_bracketed_source_ids() -> None:
    block = build_context_block(
        [make_result("app/config.py#function-from_env", "text one", 1)],
        char_budget=1000,
    )

    assert 'id="app/config.py#function-from_env"' in block
    assert "text one" in block


def test_build_context_block_always_includes_first_result() -> None:
    block = build_context_block(
        [make_result("one", "x" * 5000, 1)], char_budget=10)

    assert "x" * 5000 in block


def test_build_context_block_drops_results_exceeding_budget() -> None:
    block = build_context_block(
        [make_result("one", "short", 1), make_result("two", "y" * 5000, 2)],
        char_budget=50,
    )

    assert 'id="one"' in block
    assert 'id="two"' not in block


def test_build_augmented_message_includes_instructions_and_question() -> None:
    message = build_augmented_message(
        "How is history saved?",
        [make_result("app/conversation.py#class-Conversation", "saves json", 1)],
        char_budget=1000,
    )

    assert "insufficient information" in message
    assert "ignore" in message.lower()
    assert "How is history saved?" in message
    assert "app/conversation.py#class-Conversation" in message


def test_build_augmented_message_returns_plain_query_without_results() -> None:
    message = build_augmented_message("plain question", [], char_budget=1000)

    assert message == "plain question"

"""Tests for code-aware BM25 retrieval."""

import pytest

from app.rag.index_bm25 import BM25Index, tokenize
from app.rag.models import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, text, "sample.py", "code")


def test_tokenize_preserves_and_splits_code_identifiers() -> None:
    assert tokenize("stream_message webSearchRequests") == [
        "stream_message",
        "stream",
        "message",
        "websearchrequests",
        "web",
        "search",
        "requests",
    ]


def test_search_ranks_matching_identifier_first() -> None:
    index = BM25Index(
        [
            make_chunk(
                "client", "def stream_message(): handle Anthropic responses"),
            make_chunk("history", "def save(): persist conversation history"),
            make_chunk("config", "def from_env(): configure web search"),
        ]
    )

    results = index.search("Where is stream message handled?", top_k=2)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "client"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_search_returns_empty_results_for_no_match_or_empty_index() -> None:
    assert BM25Index([]).search("anything") == []
    assert BM25Index([make_chunk("one", "alpha")]).search("beta") == []


def test_search_rejects_invalid_top_k() -> None:
    index = BM25Index([make_chunk("one", "alpha")])

    with pytest.raises(ValueError, match="top_k"):
        index.search("alpha", top_k=0)

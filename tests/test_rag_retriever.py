"""Tests for hybrid retrieval orchestration."""

import pytest

from app.rag.index_vector import VectorIndex
from app.rag.models import Chunk
from app.rag.retriever import HybridRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.query_types: list[str] = []

    def embed_query(self, query: str, file_type: str) -> list[float]:
        self.query_types.append(file_type)
        return [1.0, 0.0] if file_type == "code" else [0.0, 1.0]


def make_chunk(chunk_id: str, text: str, file_type: str) -> Chunk:
    return Chunk(chunk_id, text, chunk_id, file_type)


def test_hybrid_retrieval_queries_separate_spaces_and_fuses_rankings() -> None:
    code = make_chunk("code", "save conversation", "code")
    prose = make_chunk("prose", "history persistence guide", "prose")
    embedder = FakeEmbedder()
    retriever = HybridRetriever(
        [code, prose],
        embedder,
        VectorIndex([code], [[1.0, 0.0]], model="code", dimension=2),
        VectorIndex([prose], [[0.0, 1.0]], model="prose", dimension=2),
    )

    results = retriever.search("save history", mode="hybrid", top_k=2)

    assert {result.chunk.chunk_id for result in results} == {"code", "prose"}
    assert embedder.query_types == ["code", "prose"]
    assert [result.rank for result in results] == [1, 2]


def test_bm25_mode_does_not_call_embedder() -> None:
    code = make_chunk("code", "save conversation", "code")
    embedder = FakeEmbedder()
    retriever = HybridRetriever([code], embedder, None, None)

    results = retriever.search("save", mode="bm25")

    assert results[0].chunk.chunk_id == "code"
    assert embedder.query_types == []


def test_bm25_mode_works_without_an_embedder() -> None:
    code = make_chunk("code", "save conversation", "code")
    retriever = HybridRetriever([code], None, None, None)

    results = retriever.search("save", mode="bm25")

    assert results[0].chunk.chunk_id == "code"


def test_vector_mode_requires_embedder_when_index_present() -> None:
    code = make_chunk("code", "save conversation", "code")
    retriever = HybridRetriever(
        [code], None, VectorIndex(
            [code], [[1.0, 0.0]], model="code", dimension=2), None
    )

    with pytest.raises(ValueError, match="embedder is required"):
        retriever.search("save", mode="vector")

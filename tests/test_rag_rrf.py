"""Tests for reciprocal rank fusion."""

import pytest

from app.rag.models import Chunk, SearchResult
from app.rag.rrf import reciprocal_rank_fusion


def result(chunk_id: str, rank: int) -> SearchResult:
    chunk = Chunk(chunk_id, chunk_id, "source.py", "code")
    return SearchResult(chunk=chunk, score=1.0 / rank, rank=rank)


def test_rrf_promotes_chunks_found_by_multiple_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        [
            [result("shared", 1), result("lexical", 2)],
            [result("semantic", 1), result("shared", 2)],
        ],
        k=60,
    )

    assert [item.chunk.chunk_id for item in fused] == [
        "shared",
        "semantic",
        "lexical",
    ]
    assert [item.rank for item in fused] == [1, 2, 3]


def test_rrf_deduplicates_within_one_ranking() -> None:
    fused = reciprocal_rank_fusion([[result("one", 1), result("one", 2)]])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_uses_chunk_id_for_deterministic_ties() -> None:
    fused = reciprocal_rank_fusion([[result("beta", 1)], [result("alpha", 1)]])

    assert [item.chunk.chunk_id for item in fused] == ["alpha", "beta"]


def test_rrf_validates_configuration() -> None:
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[]], weights=[])
    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_fusion([], k=0)

"""Reciprocal Rank Fusion for combining independent retrieval rankings."""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.models import Chunk, SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[SearchResult]:
    """Fuse rankings by chunk ID without comparing retriever-specific scores."""
    if k < 1:
        raise ValueError("k must be greater than zero")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights cannot be negative")

    chunks: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    for results, weight in zip(rankings, weights, strict=True):
        seen_in_ranking: set[str] = set()
        for position, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(
                chunk_id, 0.0) + weight / (k + position)
            best_ranks[chunk_id] = min(
                best_ranks.get(chunk_id, position), position)

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )
    return [
        SearchResult(chunk=chunks[chunk_id], score=scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]

"""Shared data models for ingestion and retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A source unit that can be indexed and cited."""

    chunk_id: str
    text: str
    source_path: str
    file_type: str
    symbol: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """A ranked chunk returned by a retriever or rank fusion."""

    chunk: Chunk
    score: float
    rank: int

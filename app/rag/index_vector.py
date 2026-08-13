"""Exact vector search and persistence for small local corpora."""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.rag.models import Chunk, SearchResult

_INDEX_SCHEMA_VERSION = 1


class VectorIndexError(RuntimeError):
    """Raised when a vector index is malformed or incompatible."""


class VectorIndex:
    """An exact inner-product index over L2-normalized vectors."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        model: str,
        dimension: int,
    ) -> None:
        if len(chunks) != len(vectors):
            raise VectorIndexError(
                "chunks and vectors must have the same length")
        self._chunks = list(chunks)
        self._vectors = [self._normalize(vector, dimension)
                         for vector in vectors]
        self.model = model
        self.dimension = dimension

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return chunks ordered by exact cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        normalized_query = self._normalize(query_vector, self.dimension)
        scored = [
            (sum(left * right for left,
             right in zip(normalized_query, vector, strict=True)), chunk)
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def save(self, path: Path) -> None:
        """Persist vectors and source metadata as deterministic JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "model": self.model,
            "dimension": self.dimension,
            "entries": [
                {
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source_path": chunk.source_path,
                        "file_type": chunk.file_type,
                        "symbol": chunk.symbol,
                    },
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors, strict=True)
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_model: str | None = None,
        expected_dimension: int | None = None,
    ) -> VectorIndex:
        """Load an index and reject stale model or dimension configurations."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["schema_version"] != _INDEX_SCHEMA_VERSION:
                raise VectorIndexError(
                    "Unsupported vector index schema version")
            model = payload["model"]
            dimension = payload["dimension"]
            if expected_model is not None and model != expected_model:
                raise VectorIndexError(
                    f"Vector index model mismatch: expected {expected_model}, got {model}"
                )
            if expected_dimension is not None and dimension != expected_dimension:
                raise VectorIndexError(
                    "Vector index dimension mismatch: "
                    f"expected {expected_dimension}, got {dimension}"
                )
            chunks = [Chunk(**entry["chunk"]) for entry in payload["entries"]]
            vectors = [entry["vector"] for entry in payload["entries"]]
        except VectorIndexError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise VectorIndexError(
                f"Could not load vector index: {exc}") from exc
        return cls(chunks, vectors, model=model, dimension=dimension)

    @staticmethod
    def _normalize(vector: list[float], dimension: int) -> list[float]:
        if len(vector) != dimension:
            raise VectorIndexError(
                f"Expected a {dimension}-dimensional vector, got {len(vector)}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise VectorIndexError(
                "Cannot index or search with a zero-length vector")
        return [float(value / norm) for value in vector]

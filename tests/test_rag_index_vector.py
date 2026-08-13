"""Tests for exact vector search and persistence."""

from pathlib import Path

import pytest

from app.rag.index_vector import VectorIndex, VectorIndexError
from app.rag.models import Chunk


def chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id, chunk_id, "source.py", "code")


def test_vector_search_ranks_cosine_similarity_and_breaks_ties() -> None:
    index = VectorIndex(
        [chunk("right"), chunk("up"), chunk("also-right")],
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
        model="test-model",
        dimension=2,
    )

    results = index.search([1.0, 0.0], top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "also-right",
        "right",
        "up",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[2].score == pytest.approx(0.0)


def test_vector_index_round_trips_and_validates_manifest(tmp_path: Path) -> None:
    path = tmp_path / "code-index.json"
    VectorIndex(
        [chunk("one")],
        [[3.0, 4.0]],
        model="voyage-code-3",
        dimension=2,
    ).save(path)

    loaded = VectorIndex.load(
        path, expected_model="voyage-code-3", expected_dimension=2
    )

    assert loaded.search([3.0, 4.0])[0].chunk.chunk_id == "one"
    with pytest.raises(VectorIndexError, match="model mismatch"):
        VectorIndex.load(path, expected_model="voyage-4")
    with pytest.raises(VectorIndexError, match="dimension mismatch"):
        VectorIndex.load(path, expected_dimension=3)


def test_vector_index_rejects_invalid_vectors() -> None:
    with pytest.raises(VectorIndexError, match="dimensional"):
        VectorIndex([chunk("one")], [[1.0]], model="model", dimension=2)
    with pytest.raises(VectorIndexError, match="zero-length"):
        VectorIndex([chunk("one")], [[0.0, 0.0]], model="model", dimension=2)

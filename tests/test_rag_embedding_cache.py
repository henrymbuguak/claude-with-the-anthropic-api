"""Tests for persistent document embedding caching."""

from pathlib import Path

from app.rag.embedding_cache import EmbeddingCache


def test_cache_key_changes_with_content_model_and_dimension() -> None:
    base = EmbeddingCache.key("text", model="model-a", dimension=2)

    assert base != EmbeddingCache.key("changed", model="model-a", dimension=2)
    assert base != EmbeddingCache.key("text", model="model-b", dimension=2)
    assert base != EmbeddingCache.key("text", model="model-a", dimension=3)


def test_cache_round_trips_vectors(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.json"
    cache = EmbeddingCache(path)
    cache.put("key", [0.6, 0.8])
    cache.save()

    loaded = EmbeddingCache(path)

    assert loaded.get("key") == [0.6, 0.8]
    assert loaded.get("missing") is None


def test_cache_ignores_corrupt_files(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.json"
    path.write_text("not json", encoding="utf-8")

    assert EmbeddingCache(path).get("key") is None

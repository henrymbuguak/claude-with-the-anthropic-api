"""Tests for cached, model-specific vector index construction."""

from pathlib import Path

from app.rag.build_index import build_indexes
from app.rag.config import RagSettings
from app.rag.embed_voyage import EmbeddingBatch
from app.rag.index_vector import VectorIndex


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls: list[tuple[list[str], str]] = []

    def model_for(self, file_type: str) -> str:
        return "code-model" if file_type == "code" else "prose-model"

    def embed_documents(self, texts: list[str], file_type: str) -> EmbeddingBatch:
        self.document_calls.append((texts, file_type))
        vector = [1.0, 0.0] if file_type == "code" else [0.0, 1.0]
        return EmbeddingBatch(
            vectors=[vector[:] for _ in texts],
            model=self.model_for(file_type),
            total_tokens=len(texts),
        )


def test_build_indexes_separates_models_and_reuses_cache(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def run():\n    return True\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Guide\nRun it.\n", encoding="utf-8")
    settings = RagSettings(
        voyage_api_key=None,
        code_embedding_model="code-model",
        prose_embedding_model="prose-model",
        embedding_dimension=2,
        index_dir=tmp_path / ".rag-index",
    )
    embedder = FakeEmbedder()

    first = build_indexes(tmp_path, settings, embedder)
    second = build_indexes(tmp_path, settings, embedder)

    assert first.embedded_chunks == 2
    assert first.cache_hits == 0
    assert second.embedded_chunks == 0
    assert second.cache_hits == 2
    assert len(embedder.document_calls) == 2
    assert VectorIndex.load(
        settings.index_dir / "code-index.json", expected_model="code-model"
    ).model == "code-model"
    assert VectorIndex.load(
        settings.index_dir / "prose-index.json", expected_model="prose-model"
    ).model == "prose-model"

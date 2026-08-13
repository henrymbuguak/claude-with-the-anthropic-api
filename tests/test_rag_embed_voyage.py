"""Tests for the Voyage embedding adapter without live API calls."""

from types import SimpleNamespace

import pytest

from app.rag.config import RagSettings
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder


class FakeVoyageClient:
    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict]] = []

    def embed(self, texts: list[str], **kwargs):
        self.calls.append((texts, kwargs))
        vectors = [[3.0, 4.0] + [0.0] * (self.dimension - 2) for _ in texts]
        return SimpleNamespace(embeddings=vectors, total_tokens=len(texts) * 3)


def settings(**overrides) -> RagSettings:
    defaults = {
        "voyage_api_key": None,
        "embedding_dimension": 2,
        "embedding_batch_size": 2,
    }
    defaults.update(overrides)
    return RagSettings(**defaults)


def test_embed_documents_selects_code_model_and_normalizes() -> None:
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(settings(), client=client)

    result = embedder.embed_documents(["first", "second"], "code")

    assert result.model == "voyage-code-3"
    assert result.total_tokens == 6
    assert result.vectors[0] == pytest.approx([0.6, 0.8])
    assert result.vectors[1] == pytest.approx([0.6, 0.8])
    assert client.calls[0][1] == {
        "model": "voyage-code-3",
        "input_type": "document",
        "truncation": False,
        "output_dimension": 2,
        "output_dtype": "float",
    }


def test_embed_query_selects_prose_model_and_query_input_type() -> None:
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(settings(), client=client)

    vector = embedder.embed_query("How is history saved?", "prose")

    assert vector == pytest.approx([0.6, 0.8])
    assert client.calls[0][1]["model"] == "voyage-4"
    assert client.calls[0][1]["input_type"] == "query"


def test_embed_documents_batches_requests_and_sums_tokens() -> None:
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(settings(embedding_batch_size=2), client=client)

    result = embedder.embed_documents(["one", "two", "three"], "code")

    assert [texts for texts, _ in client.calls] == [["one", "two"], ["three"]]
    assert result.total_tokens == 9


def test_embed_documents_handles_empty_input_without_api_call() -> None:
    client = FakeVoyageClient()
    embedder = VoyageEmbedder(settings(), client=client)

    result = embedder.embed_documents([], "code")

    assert result.vectors == []
    assert result.total_tokens == 0
    assert client.calls == []


def test_embedder_rejects_bad_dimensions_and_wraps_api_errors() -> None:
    bad_client = FakeVoyageClient(dimension=3)
    embedder = VoyageEmbedder(settings(), client=bad_client)

    with pytest.raises(EmbeddingError, match="unexpected embedding dimension"):
        embedder.embed_query("query", "code")

    class FailingClient:
        def embed(self, texts: list[str], **kwargs):
            raise RuntimeError("service unavailable")

    with pytest.raises(EmbeddingError, match="service unavailable"):
        VoyageEmbedder(settings(), client=FailingClient()
                       ).embed_query("query", "code")

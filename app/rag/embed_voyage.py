"""Voyage AI embedding adapter for code and prose retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from app.rag.config import RagSettings


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated or validated."""


class VoyageClient(Protocol):
    """The subset of voyageai.Client used by this project."""

    def embed(self, texts: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EmbeddingBatch:
    """Normalized embeddings and the API usage needed to produce them."""

    vectors: list[list[float]]
    model: str
    total_tokens: int


class VoyageEmbedder:
    """Select Voyage models by file type and prepare vectors for cosine search."""

    def __init__(
        self,
        settings: RagSettings,
        client: VoyageClient | None = None,
    ) -> None:
        self._settings = settings
        if client is None:
            if not settings.voyage_api_key:
                raise EmbeddingError(
                    "VOYAGE_API_KEY is required for Voyage embeddings")
            try:
                import voyageai
            except ImportError as exc:
                raise EmbeddingError(
                    "The voyageai package is not installed. Run: uv add voyageai"
                ) from exc
            client = voyageai.Client(
                api_key=settings.voyage_api_key, max_retries=3)
        self._client = client

    def embed_documents(self, texts: list[str], file_type: str) -> EmbeddingBatch:
        """Embed indexable source chunks with Voyage's document input type."""
        return self._embed(texts, file_type=file_type, input_type="document")

    def embed_query(self, query: str, file_type: str) -> list[float]:
        """Embed one search query in the selected model's retrieval space."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        batch = self._embed([query], file_type=file_type, input_type="query")
        return batch.vectors[0]

    def model_for(self, file_type: str) -> str:
        """Return the configured model for code or prose chunks."""
        if file_type == "code":
            return self._settings.code_embedding_model
        if file_type == "prose":
            return self._settings.prose_embedding_model
        raise ValueError(
            f"Unsupported file type for embeddings: {file_type!r}")

    def _embed(
        self,
        texts: list[str],
        *,
        file_type: str,
        input_type: str,
    ) -> EmbeddingBatch:
        model = self.model_for(file_type)
        if not texts:
            return EmbeddingBatch(vectors=[], model=model, total_tokens=0)
        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs cannot be empty")

        vectors: list[list[float]] = []
        total_tokens = 0
        batch_size = self._settings.embedding_batch_size
        try:
            for start in range(0, len(texts), batch_size):
                response = self._client.embed(
                    texts[start: start + batch_size],
                    model=model,
                    input_type=input_type,
                    truncation=False,
                    output_dimension=self._settings.embedding_dimension,
                    output_dtype="float",
                )
                response_vectors = response.embeddings
                if len(response_vectors) != len(texts[start: start + batch_size]):
                    raise EmbeddingError(
                        "Voyage returned a different number of vectors than inputs"
                    )
                vectors.extend(self._normalize(vector)
                               for vector in response_vectors)
                total_tokens += response.total_tokens
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Voyage embedding request failed: {exc}") from exc

        return EmbeddingBatch(vectors=vectors, model=model, total_tokens=total_tokens)

    def _normalize(self, vector: list[float]) -> list[float]:
        if len(vector) != self._settings.embedding_dimension:
            raise EmbeddingError(
                "Voyage returned an unexpected embedding dimension: "
                f"expected {self._settings.embedding_dimension}, got {len(vector)}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise EmbeddingError("Voyage returned a zero-length embedding")
        return [float(value / norm) for value in vector]

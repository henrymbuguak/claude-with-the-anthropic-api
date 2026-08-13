"""Orchestrate BM25, model-specific vector retrieval, and RRF."""

from __future__ import annotations

from app.rag.config import RagSettings
from app.rag.embed_voyage import VoyageEmbedder
from app.rag.index_bm25 import BM25Index
from app.rag.index_vector import VectorIndex
from app.rag.models import Chunk, SearchResult
from app.rag.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Search lexical, code-vector, and prose-vector rankings."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: VoyageEmbedder | None,
        code_index: VectorIndex | None,
        prose_index: VectorIndex | None,
    ) -> None:
        self._bm25 = BM25Index(chunks)
        self._embedder = embedder
        self._code_index = code_index
        self._prose_index = prose_index

    @classmethod
    def from_disk(
        cls,
        chunks: list[Chunk],
        settings: RagSettings,
        embedder: VoyageEmbedder | None = None,
    ) -> HybridRetriever:
        """Load model-compatible code and prose indexes from disk."""
        code_path = settings.index_dir / "code-index.json"
        prose_path = settings.index_dir / "prose-index.json"
        code_index = (
            VectorIndex.load(
                code_path,
                expected_model=settings.code_embedding_model,
                expected_dimension=settings.embedding_dimension,
            )
            if code_path.exists()
            else None
        )
        prose_index = (
            VectorIndex.load(
                prose_path,
                expected_model=settings.prose_embedding_model,
                expected_dimension=settings.embedding_dimension,
            )
            if prose_path.exists()
            else None
        )
        return cls(chunks, embedder, code_index, prose_index)

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        top_k_per_retriever: int = 30,
    ) -> list[SearchResult]:
        """Search one retrieval mode and return a unified ranked list."""
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("mode must be bm25, vector, or hybrid")
        if mode == "bm25":
            return self._bm25.search(query, top_k=top_k)

        rankings: list[list[SearchResult]] = []
        if mode == "hybrid":
            rankings.append(self._bm25.search(
                query, top_k=top_k_per_retriever))
        if self._code_index is not None:
            if self._embedder is None:
                raise ValueError(
                    "embedder is required to search the code vector index")
            code_query = self._embedder.embed_query(query, "code")
            rankings.append(
                self._code_index.search(code_query, top_k=top_k_per_retriever)
            )
        if self._prose_index is not None:
            if self._embedder is None:
                raise ValueError(
                    "embedder is required to search the prose vector index")
            prose_query = self._embedder.embed_query(query, "prose")
            rankings.append(
                self._prose_index.search(
                    prose_query, top_k=top_k_per_retriever)
            )
        if not rankings:
            return []
        fused = reciprocal_rank_fusion(rankings, k=60)
        return [
            SearchResult(chunk=result.chunk, score=result.score, rank=rank)
            for rank, result in enumerate(fused[:top_k], start=1)
        ]

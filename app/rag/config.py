"""Configuration for semantic retrieval with Voyage AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class RagConfigError(RuntimeError):
    """Raised when RAG configuration is missing or invalid."""


@dataclass(frozen=True)
class RagSettings:
    """Settings used only by semantic indexing and retrieval."""

    voyage_api_key: str | None
    code_embedding_model: str = "voyage-code-3"
    prose_embedding_model: str = "voyage-4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 64
    index_dir: Path = Path(".rag-index")
    enabled: bool = False
    chat_mode: str = "hybrid"
    chat_top_k: int = 5
    context_char_budget: int = 6000

    @classmethod
    def from_env(cls, *, require_api_key: bool = False) -> RagSettings:
        """Load and validate RAG settings without affecting BM25-only usage."""
        api_key = os.getenv("VOYAGE_API_KEY", "").strip() or None
        if require_api_key and api_key is None:
            raise RagConfigError(
                "VOYAGE_API_KEY is not set. Add it to .env before using semantic retrieval."
            )

        dimension_raw = os.getenv("RAG_EMBEDDING_DIMENSION", "1024")
        batch_size_raw = os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64")
        try:
            dimension = int(dimension_raw)
        except ValueError as exc:
            raise RagConfigError(
                f"RAG_EMBEDDING_DIMENSION must be an integer, got {dimension_raw!r}"
            ) from exc
        if dimension not in {256, 512, 1024, 2048}:
            raise RagConfigError(
                "RAG_EMBEDDING_DIMENSION must be one of 256, 512, 1024, or 2048"
            )

        try:
            batch_size = int(batch_size_raw)
        except ValueError as exc:
            raise RagConfigError(
                f"RAG_EMBEDDING_BATCH_SIZE must be an integer, got {batch_size_raw!r}"
            ) from exc
        if not 1 <= batch_size <= 1000:
            raise RagConfigError(
                "RAG_EMBEDDING_BATCH_SIZE must be between 1 and 1000"
            )

        enabled_raw = os.getenv("RAG_ENABLED", "false").strip().lower()
        if enabled_raw not in {"true", "false"}:
            raise RagConfigError(
                f"RAG_ENABLED must be 'true' or 'false', got {enabled_raw!r}"
            )

        chat_mode = os.getenv("RAG_CHAT_MODE", "hybrid").strip()
        if chat_mode not in {"bm25", "vector", "hybrid"}:
            raise RagConfigError(
                f"RAG_CHAT_MODE must be one of bm25, vector, or hybrid, got {chat_mode!r}"
            )

        chat_top_k_raw = os.getenv("RAG_CHAT_TOP_K", "5")
        try:
            chat_top_k = int(chat_top_k_raw)
        except ValueError as exc:
            raise RagConfigError(
                f"RAG_CHAT_TOP_K must be an integer, got {chat_top_k_raw!r}"
            ) from exc
        if chat_top_k < 1:
            raise RagConfigError("RAG_CHAT_TOP_K must be greater than zero")

        context_char_budget_raw = os.getenv("RAG_CONTEXT_CHAR_BUDGET", "6000")
        try:
            context_char_budget = int(context_char_budget_raw)
        except ValueError as exc:
            raise RagConfigError(
                "RAG_CONTEXT_CHAR_BUDGET must be an integer, "
                f"got {context_char_budget_raw!r}"
            ) from exc
        if context_char_budget < 500:
            raise RagConfigError(
                "RAG_CONTEXT_CHAR_BUDGET must be at least 500")

        return cls(
            voyage_api_key=api_key,
            code_embedding_model=os.getenv(
                "RAG_CODE_EMBEDDING_MODEL", "voyage-code-3"
            ).strip(),
            prose_embedding_model=os.getenv(
                "RAG_PROSE_EMBEDDING_MODEL", "voyage-4"
            ).strip(),
            embedding_dimension=dimension,
            embedding_batch_size=batch_size,
            index_dir=Path(os.getenv("RAG_INDEX_DIR", ".rag-index")),
            enabled=enabled_raw == "true",
            chat_mode=chat_mode,
            chat_top_k=chat_top_k,
            context_char_budget=context_char_budget,
        )

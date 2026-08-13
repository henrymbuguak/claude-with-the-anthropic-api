"""Make a minimal live Voyage request to verify local credentials and SDK setup."""

from __future__ import annotations

from app.rag.config import RagConfigError, RagSettings
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder


def main() -> None:
    try:
        settings = RagSettings.from_env(require_api_key=True)
        embedder = VoyageEmbedder(settings)
        document = embedder.embed_documents(
            ["Conversation history is persisted to a JSON file."], "code"
        )
        query = embedder.embed_query(
            "How is conversation history saved?", "code")
    except (RagConfigError, EmbeddingError) as exc:
        raise SystemExit(f"Voyage smoke test failed: {exc}") from exc

    similarity = sum(
        left * right for left, right in zip(document.vectors[0], query, strict=True)
    )
    print(f"Model: {document.model}")
    print(f"Dimensions: {len(query)}")
    print(f"Document tokens: {document.total_tokens}")
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()

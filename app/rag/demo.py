"""Command-line inspection of lexical, semantic, and hybrid retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.config import RagConfigError, RagSettings
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder
from app.rag.index_bm25 import BM25Index
from app.rag.index_vector import VectorIndexError
from app.rag.ingest import ingest_path
from app.rag.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search local source chunks with BM25 or Voyage embeddings.")
    parser.add_argument("query", help="Question or search terms")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Corpus root (default: current directory)",
    )
    parser.add_argument("--top-k", type=int, default=5,
                        help="Results to display")
    parser.add_argument(
        "--mode",
        choices=("bm25", "vector", "hybrid"),
        default="bm25",
        help="Retrieval mode (default: bm25)",
    )
    args = parser.parse_args()

    chunks = ingest_path(args.root)
    if args.mode == "bm25":
        results = BM25Index(chunks).search(args.query, top_k=args.top_k)
    else:
        try:
            settings = RagSettings.from_env(require_api_key=True)
            embedder = VoyageEmbedder(settings)
            retriever = HybridRetriever.from_disk(chunks, settings, embedder)
            results = retriever.search(
                args.query, mode=args.mode, top_k=args.top_k
            )
        except (RagConfigError, EmbeddingError, VectorIndexError) as exc:
            raise SystemExit(f"RAG retrieval error: {exc}") from exc

    print(f"Indexed {len(chunks)} chunks from {args.root.resolve()}\n")
    if not results:
        print("No matching chunks found.")
        return

    for result in results:
        preview = " ".join(result.chunk.text.split())[:240]
        print(
            f"{result.rank}. {result.chunk.chunk_id} "
            f"({args.mode} {result.score:.3f})\n   {preview}\n"
        )


if __name__ == "__main__":
    main()

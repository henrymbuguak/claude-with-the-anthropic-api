"""Build cached Voyage vector indexes for a local repository."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.rag.config import RagConfigError, RagSettings
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder
from app.rag.embedding_cache import EmbeddingCache
from app.rag.index_vector import VectorIndex
from app.rag.ingest import ingest_path
from app.rag.models import Chunk


@dataclass(frozen=True)
class BuildStats:
    """Observable indexing work and API usage."""

    total_chunks: int
    code_chunks: int
    prose_chunks: int
    embedded_chunks: int
    cache_hits: int
    total_tokens: int


def build_indexes(
    root: Path,
    settings: RagSettings,
    embedder: VoyageEmbedder,
    *,
    rebuild: bool = False,
) -> BuildStats:
    """Build separate code/prose indexes while reusing cached embeddings."""
    chunks = ingest_path(root)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.index_dir / "embedding-cache.json"
    if rebuild and cache_path.exists():
        cache_path.unlink()
    cache = EmbeddingCache(cache_path)
    embedded_chunks = 0
    cache_hits = 0
    total_tokens = 0

    for file_type, filename in (("code", "code-index.json"), ("prose", "prose-index.json")):
        selected = [chunk for chunk in chunks if chunk.file_type == file_type]
        if not selected:
            continue
        model = embedder.model_for(file_type)
        vectors: list[list[float] | None] = []
        missing_chunks: list[Chunk] = []
        missing_positions: list[int] = []
        keys: list[str] = []
        for position, chunk in enumerate(selected):
            key = EmbeddingCache.key(
                chunk.text, model=model, dimension=settings.embedding_dimension
            )
            keys.append(key)
            cached = cache.get(key)
            vectors.append(cached)
            if cached is None:
                missing_chunks.append(chunk)
                missing_positions.append(position)
            else:
                cache_hits += 1

        if missing_chunks:
            batch = embedder.embed_documents(
                [chunk.text for chunk in missing_chunks], file_type
            )
            embedded_chunks += len(missing_chunks)
            total_tokens += batch.total_tokens
            for position, vector in zip(missing_positions, batch.vectors, strict=True):
                vectors[position] = vector
                cache.put(keys[position], vector)

        complete_vectors = [vector for vector in vectors if vector is not None]
        if len(complete_vectors) != len(selected):
            raise EmbeddingError("Not every chunk received an embedding")
        VectorIndex(
            selected,
            complete_vectors,
            model=model,
            dimension=settings.embedding_dimension,
        ).save(settings.index_dir / filename)

    cache.save()
    return BuildStats(
        total_chunks=len(chunks),
        code_chunks=sum(chunk.file_type == "code" for chunk in chunks),
        prose_chunks=sum(chunk.file_type == "prose" for chunk in chunks),
        embedded_chunks=embedded_chunks,
        cache_hits=cache_hits,
        total_tokens=total_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local Voyage vector indexes.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore cached embeddings")
    args = parser.parse_args()

    try:
        settings = RagSettings.from_env(require_api_key=True)
        stats = build_indexes(
            args.root,
            settings,
            VoyageEmbedder(settings),
            rebuild=args.rebuild,
        )
    except (RagConfigError, EmbeddingError) as exc:
        raise SystemExit(f"RAG indexing error: {exc}") from exc

    print(f"Discovered {stats.total_chunks} chunks")
    print(f"Code: {stats.code_chunks}; prose: {stats.prose_chunks}")
    print(f"Embedded: {stats.embedded_chunks}; cache hits: {stats.cache_hits}")
    print(f"Voyage tokens used: {stats.total_tokens}")
    print(f"Saved indexes to {settings.index_dir.resolve()}")


if __name__ == "__main__":
    main()

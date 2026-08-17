# Build and search a vector index

## What you'll build

In this tutorial, you build the exact vector index this repository's semantic
retriever searches. You store chunk vectors, rank them by cosine similarity,
and persist an index that rejects vectors from an incompatible model or
dimension.

**Time:** About 30 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.
- [Implement Okapi BM25 from scratch](implement-okapi-bm25.md).
- Familiarity with the repository's [retrieval architecture](../architecture.md).

You do not need an API key. Computing real embeddings requires a paid Voyage
API call, so every command in this tutorial searches small, hand-written
vectors instead.

## See it work

Run the tests that already exercise the finished class, and read the three
behaviors they confirm:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_index_vector.py -v
```

<!-- verify expect match=contains -->

```text
test_vector_search_ranks_cosine_similarity_and_breaks_ties PASSED
test_vector_index_round_trips_and_validates_manifest PASSED
test_vector_index_rejects_invalid_vectors PASSED
3 passed
```

Each test name states a guarantee this tutorial's class provides: ranking by
similarity with a deterministic tie-break, surviving a save-and-reload round
trip while catching a stale configuration, and rejecting vectors that don't
fit the index. The rest of this tutorial rebuilds the class that already
passes these tests.

## How a vector index works

A **vector index** pairs each chunk with a fixed-length list of numbers, its
vector, so a query can be turned into the same kind of list and compared
against every stored entry. Unlike Okapi BM25's keyword counts, a vector
index says nothing about which words a chunk contains; it only stores
numbers, typically produced by an embedding model that maps similar meanings
to nearby vectors. This tutorial's `VectorIndex` never computes those numbers
itself. It exact-searches whichever vectors you hand it, which is why every
command in this tutorial runs without an API key.

**Cosine similarity** compares two vectors by the angle between them, not
their length. Given two vectors, cosine similarity is their dot product
divided by the product of their lengths. That division cancels out
completely only when both vectors already have length one, called *unit
vectors*. `VectorIndex` normalizes every vector once, when the index is
built, so `search()` computes a plain dot product instead of dividing on
every comparison. Cosine similarity is always contained in the range -1 to 1,
no matter how many vectors the index holds.

**Dimension compatibility** is the requirement that every vector entering the
index has the same length, and that a saved index is only ever compared
against vectors from the model that created it. Two vectors of different
lengths have no defined dot product, and two vectors from different
embedding models can share a length while measuring entirely different
notions of similarity. `VectorIndex` checks both: once per vector when it's
normalized, and once per file when a saved index reloads.

## Preserve the reference implementation

The finished implementation already exists so that you can run it before
rebuilding it. Work on a disposable branch or temporary copy of the
repository.

In PowerShell, preserve the file and clear the original:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```powershell
Copy-Item app/rag/index_vector.py app/rag/index_vector_reference.txt
Clear-Content app/rag/index_vector.py
```

In Bash, run the equivalent commands:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```bash
cp app/rag/index_vector.py app/rag/index_vector_reference.txt
: > app/rag/index_vector.py
```

## Build the index

1. In `app/rag/index_vector.py`, add the module docstring, imports, the schema
    version, and the error type:

    ```python
    """Exact vector search and persistence for small local corpora."""

    from __future__ import annotations

    import json
    import math
    from pathlib import Path

    from app.rag.models import Chunk, SearchResult

    _INDEX_SCHEMA_VERSION = 1


    class VectorIndexError(RuntimeError):
        """Raised when a vector index is malformed or incompatible."""
    ```

    `_INDEX_SCHEMA_VERSION` names the on-disk file format. Raising
    `VectorIndexError` for every malformed or incompatible index gives callers
    one exception type to catch, instead of several unrelated built-in errors.

2. Add the class and validate that every chunk has a matching vector:

    ```python
    class VectorIndex:
        """An exact inner-product index over L2-normalized vectors."""

        def __init__(
            self,
            chunks: list[Chunk],
            vectors: list[list[float]],
            *,
            model: str,
            dimension: int,
        ) -> None:
            if len(chunks) != len(vectors):
                raise VectorIndexError(
                    "chunks and vectors must have the same length")
    ```

3. Still in `__init__`, normalize and store each vector, then record the
    model and dimension for later comparisons:

    ```python
            self._chunks = list(chunks)
            self._vectors = [self._normalize(vector, dimension)
                             for vector in vectors]
            self.model = model
            self.dimension = dimension
    ```

    `_normalize()` doesn't exist yet; you add it in
    [Enforce dimension compatibility](#enforce-dimension-compatibility). Python
    resolves `self._normalize` when `__init__` runs, not when the class body
    is read, so the forward reference is safe once the method exists anywhere
    in the class.

4. Add `__len__`, so an index reports its own size like a built-in
    collection:

    ```python
        def __len__(self) -> int:
            return len(self._chunks)
    ```

## Search by cosine similarity

1. Add `search()` and guard against an invalid result count:

    ```python
        def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
            """Return chunks ordered by exact cosine similarity."""
            if top_k < 1:
                raise ValueError("top_k must be greater than zero")
            normalized_query = self._normalize(query_vector, self.dimension)
    ```

2. Score every stored vector against the normalized query:

    ```python
            scored = [
                (sum(left * right for left,
                 right in zip(normalized_query, vector, strict=True)), chunk)
                for chunk, vector in zip(self._chunks, self._vectors, strict=True)
            ]
    ```

    Both `normalized_query` and every stored vector already have length one,
    so their dot product equals the cosine of the angle between them. No
    separate division happens at search time.

3. Sort by descending score with a deterministic tie-break, and return the
    top results:

    ```python
            scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
            return [
                SearchResult(chunk=chunk, score=score, rank=rank)
                for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
            ]
    ```

    Sorting by `chunk_id` after the score keeps the ranking identical across
    runs whenever two chunks tie exactly.

## Persist and reload the index

1. Add `save()` to write vectors and chunk metadata as deterministic JSON:

    ```python
        def save(self, path: Path) -> None:
            """Persist vectors and source metadata as deterministic JSON."""
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": _INDEX_SCHEMA_VERSION,
                "model": self.model,
                "dimension": self.dimension,
                "entries": [
                    {
                        "chunk": {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "source_path": chunk.source_path,
                            "file_type": chunk.file_type,
                            "symbol": chunk.symbol,
                        },
                        "vector": vector,
                    }
                    for chunk, vector in zip(self._chunks, self._vectors, strict=True)
                ],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ```

2. Add the `load()` classmethod, and reject a file whose model or dimension
    doesn't match what the caller expects:

    ```python
        @classmethod
        def load(
            cls,
            path: Path,
            *,
            expected_model: str | None = None,
            expected_dimension: int | None = None,
        ) -> VectorIndex:
            """Load an index and reject stale model or dimension configurations."""
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["schema_version"] != _INDEX_SCHEMA_VERSION:
                    raise VectorIndexError(
                        "Unsupported vector index schema version")
                model = payload["model"]
                dimension = payload["dimension"]
                if expected_model is not None and model != expected_model:
                    raise VectorIndexError(
                        f"Vector index model mismatch: expected {expected_model}, got {model}"
                    )
                if expected_dimension is not None and dimension != expected_dimension:
                    raise VectorIndexError(
                        "Vector index dimension mismatch: "
                        f"expected {expected_dimension}, got {dimension}"
                    )
                chunks = [Chunk(**entry["chunk"]) for entry in payload["entries"]]
                vectors = [entry["vector"] for entry in payload["entries"]]
            except VectorIndexError:
                raise
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise VectorIndexError(
                    f"Could not load vector index: {exc}") from exc
            return cls(chunks, vectors, model=model, dimension=dimension)
    ```

    `expected_model` and `expected_dimension` default to `None`, so a caller
    that hasn't decided which embedder to require yet can still load the
    file. Passing both turns a silent mismatch into an immediate
    `VectorIndexError`.

## Enforce dimension compatibility

1. Add the static `_normalize()` helper that every constructor and search
    call relies on:

    ```python
        @staticmethod
        def _normalize(vector: list[float], dimension: int) -> list[float]:
            if len(vector) != dimension:
                raise VectorIndexError(
                    f"Expected a {dimension}-dimensional vector, got {len(vector)}"
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise VectorIndexError(
                    "Cannot index or search with a zero-length vector")
            return [float(value / norm) for value in vector]
    ```

    This one helper enforces dimension compatibility for every vector that
    enters the index and produces the unit vectors that make the search's
    plain dot product equal to cosine similarity. A zero-length vector has no
    direction to compare, so `_normalize()` rejects it instead of dividing by
    zero.

## Verify your work

Run the focused tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_index_vector.py -q
```

<!-- verify expect match=contains -->

```text
3 passed
```

Check the implementation and test with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check app/rag/index_vector.py tests/test_rag_index_vector.py
```

After your implementation matches the reference behavior, remove the
temporary copy:

<!-- verify manual reason="Deletes the tutorial's temporary reference copy" -->

```powershell
Remove-Item app/rag/index_vector_reference.txt
```

## Break it on purpose

Run the focused test that deliberately passes incompatible vectors:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_index_vector.py::test_vector_index_rejects_invalid_vectors -v
```

<!-- verify expect match=contains -->

```text
PASSED
```

The test first gives a two-dimensional index a one-dimensional vector and
asserts that `VectorIndexError` reports the dimension mismatch. It then gives
the index an all-zero vector and asserts that the missing direction is rejected.
Both failures protect the cosine-similarity calculation from inputs for which
it is undefined.

Even for valid vectors, cosine similarity stays in the range -1 to 1. Okapi
BM25 has no fixed ceiling, so lexical and vector scores remain incomparable.
Adding or averaging them directly would let whichever retriever produces larger
raw numbers dominate. Combining the rankings safely means comparing ranks, not
raw scores.

## Troubleshooting

| Symptom                                                             | Resolution                                                                 | Source    |
| --------------------------------------------------------------------| --------------------------------------------------------------------------- | --------- |
| `VectorIndexError: chunks and vectors must have the same length`    | Pass equal-length `chunks` and `vectors` lists to `VectorIndex`.             | Observed  |
| `VectorIndexError: Expected a <dimension>-dimensional vector, got <n>` | Confirm every vector matches the index's configured `dimension`.          | Observed  |
| `VectorIndexError: Cannot index or search with a zero-length vector`| Drop or re-embed chunks that produced an all-zero vector.                   | Observed  |
| `VectorIndexError: Vector index model mismatch: expected ... got ...` | Rebuild the index with the same embedding model you pass as `expected_model`. | Observed |
| `ModuleNotFoundError: No module named 'app'`                        | Run the command from the repository root with `uv run`.                     | Observed  |

## Next steps

A planned tutorial, Merge rankings with Reciprocal Rank Fusion, combines this
index's cosine-similarity ranking with the BM25 ranking from
[Implement Okapi BM25 from scratch](implement-okapi-bm25.md) by comparing
ranks instead of the incomparable raw scores this tutorial exposed.

For the underlying formula, read the
[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) reference.

## Tested against

| Item     | Value      |
| -------- | ---------- |
| Python   | 3.13.15; the repository requires 3.12 or later |
| pytest   | 9.1.1      |
| Verified | 2026-08-17 |

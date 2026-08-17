# Merge rankings with Reciprocal Rank Fusion

## What you'll build

In this tutorial, you build the Reciprocal Rank Fusion function used by this
repository's hybrid retriever. You combine lexical and vector rankings without
comparing their incompatible raw scores, promote chunks found by more than one
retriever, and return one deterministic result list.

**Time:** About 30 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.
- [Implement Okapi BM25 from scratch](implement-okapi-bm25.md).
- [Build and search a vector index](build-vector-index.md).

You do not need an API key. Every command uses small in-memory rankings.

## See it work

Run the focused RRF tests and read the behaviors they confirm:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_rrf.py -v
```

<!-- verify expect match=contains -->

```text
test_rrf_promotes_chunks_found_by_multiple_retrievers PASSED
test_rrf_deduplicates_within_one_ranking PASSED
test_rrf_uses_chunk_id_for_deterministic_ties PASSED
test_rrf_validates_configuration PASSED
4 passed
```

The tests prove that shared chunks gain influence from multiple rankings,
duplicate results do not count twice, ties remain stable, and invalid fusion
settings fail before scoring begins.

## How Reciprocal Rank Fusion works

A **rank fusion** method combines multiple ordered result lists into one. The
input lists may come from different retrieval systems, such as BM25 and cosine
similarity, whose raw scores are not comparable.

Reciprocal Rank Fusion uses a result's position instead of its source score. A
result at position $r$ contributes:

```text
weight / (k + r)
```

The constant `k` reduces the difference between nearby positions. This project
uses `k=60` by default. A chunk found by multiple retrievers receives one
contribution from each ranking, so repeated independent evidence can move it
above chunks found by only one retriever.

The input scores remain **incomparable score spaces**: BM25 scores depend on term
rarity and corpus statistics, while cosine similarity stays between -1 and 1.
RRF never adds those source scores. Its output score is a new reciprocal-rank
score used only for the fused ordering.

## Preserve the reference implementation

The finished implementation already exists so that you can run it before
rebuilding it. Work on a disposable branch or temporary copy of the repository.

In PowerShell, preserve the file and clear the original:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```powershell
Copy-Item app/rag/rrf.py app/rag/rrf_reference.txt
Clear-Content app/rag/rrf.py
```

In Bash, run the equivalent commands:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```bash
cp app/rag/rrf.py app/rag/rrf_reference.txt
: > app/rag/rrf.py
```

## Validate fusion settings

1. In `app/rag/rrf.py`, add the module imports:

    ```python
    """Reciprocal Rank Fusion for combining independent retrieval rankings."""

    from __future__ import annotations

    from collections.abc import Sequence

    from app.rag.models import Chunk, SearchResult
    ```

2. Add the function signature and validate `k`:

    ```python
    def reciprocal_rank_fusion(
        rankings: Sequence[Sequence[SearchResult]],
        *,
        k: int = 60,
        weights: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """Fuse rankings by chunk ID without comparing source scores."""
        if k < 1:
            raise ValueError("k must be greater than zero")
    ```

3. Create default weights and require one weight per ranking:

    ```python
        if weights is None:
            weights = [1.0] * len(rankings)
        if len(weights) != len(rankings):
            raise ValueError("weights must match the number of rankings")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights cannot be negative")
    ```

    A weight changes one retriever's influence without making its source scores
    comparable to another retriever's scores.

## Accumulate reciprocal ranks

1. Create dictionaries for source chunks, fused scores, and best positions:

    ```python
        chunks: dict[str, Chunk] = {}
        scores: dict[str, float] = {}
        best_ranks: dict[str, int] = {}
    ```

2. Iterate through each ranking and its weight:

    ```python
        for results, weight in zip(rankings, weights, strict=True):
            seen_in_ranking: set[str] = set()
            for position, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
    ```

3. Ignore a duplicate chunk within the same ranking:

    ```python
                if chunk_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(chunk_id)
    ```

    One retriever may return a duplicate by mistake. Counting it twice would
    imitate independent evidence that does not exist.

4. Store the chunk and add its reciprocal-rank contribution:

    ```python
                chunks[chunk_id] = result.chunk
                scores[chunk_id] = scores.get(
                    chunk_id, 0.0
                ) + weight / (k + position)
                best_ranks[chunk_id] = min(
                    best_ranks.get(chunk_id, position), position
                )
    ```

    A chunk present in two rankings receives two contributions. `best_ranks`
    records its strongest source position for deterministic tie-breaking.

## Build the fused ranking

1. Sort chunk IDs by descending fused score, best source position, and chunk ID:

    ```python
        ordered_ids = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                best_ranks[chunk_id],
                chunk_id,
            ),
        )
    ```

2. Convert the ordered IDs into new `SearchResult` values:

    ```python
        return [
            SearchResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered_ids, start=1)
        ]
    ```

    The returned score belongs to the fused ranking. It does not preserve or
    reinterpret any source retriever's score.

## Verify your work

Run the focused tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_rrf.py -q
```

<!-- verify expect match=contains -->

```text
4 passed
```

Check the implementation and tests with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check app/rag/rrf.py tests/test_rag_rrf.py
```

After your implementation matches the reference behavior, remove the temporary
copy:

<!-- verify manual reason="Deletes the tutorial's temporary reference copy" -->

```powershell
Remove-Item app/rag/rrf_reference.txt
```

## Break it on purpose

Run the test that deliberately passes incompatible configuration:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_rrf.py::test_rrf_validates_configuration -v
```

<!-- verify expect match=contains -->

```text
test_rrf_validates_configuration PASSED
1 passed
```

The test supplies one ranking but no matching weight and asserts that the
function raises `ValueError`. It also passes `k=0`, which would make the fusion
constant invalid, and asserts the second error. These failures keep
configuration mistakes from producing a plausible but misleading fused order.

RRF still has an important limit: it can promote agreement only among the
rankings it receives. If every retriever misses the relevant chunk, fusion
cannot recover it. Retrieval evaluation measures that end-to-end behavior.

## Troubleshooting

| Symptom | Resolution | Source |
| --- | --- | --- |
| `ValueError: weights must match the number of rankings` | Pass one weight for each ranking, or omit `weights` for equal weighting. | Observed |
| `ValueError: k must be greater than zero` | Use a positive fusion constant. | Observed |
| One duplicate chunk receives too much score | Keep `seen_in_ranking` inside the outer ranking loop. | Predicted |
| Equal scores change order between runs | Include best rank and chunk ID in the sort key. | Predicted |
| A strong source score has no effect | Expected: RRF uses rank positions, not source scores. | Predicted |

## Next steps

Continue with [Benchmark retrieval quality](benchmark-retrieval-quality.md) to
measure whether fused or individual rankings return relevant chunks at a useful
cutoff. A later hybrid-retriever tutorial wires BM25, vector search, and RRF into
one query path.

For the original method, read [Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

## Tested against

| Item | Value |
| --- | --- |
| Python | 3.13.14; the repository requires 3.12 or later |
| pytest | 9.1.1 |
| Verified | 2026-08-17 |

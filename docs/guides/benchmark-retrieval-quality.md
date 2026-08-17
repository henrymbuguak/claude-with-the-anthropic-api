# Benchmark retrieval quality

## What you'll build

In this tutorial, you turn six labeled retrieval questions into a repeatable
BM25 quality report. You calculate Hit@k, Recall@k, and mean reciprocal rank
(MRR), then use the report to identify questions that lexical retrieval misses.

**Time:** About 30 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.
- [Implement Okapi BM25 from scratch](implement-okapi-bm25.md).

You do not need an API key. The benchmark uses local chunks and deterministic
labels from `eval/rag_cases.jsonl`.

## See it work

Run the BM25 benchmark from the repository root:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m eval.rag_retrieval
```

<!-- verify expect match=contains -->

```text
Corpus: 267 chunks | Cases: 6 | k=5
Mode      Cases    Hit@k   Recall@k      MRR
bm25          6    0.500      0.417    0.333
bm25 misses: conversation-persistence, citation-deduplication, system-prompt-file
```

The run searches <!-- fact corpus.chunk_count -->267<!-- /fact --> chunks for
<!-- fact benchmark.bm25_at_5.case_count -->6<!-- /fact --> labeled questions.
The three metrics describe different parts of the resulting rankings.

## How retrieval metrics work

Each record in `eval/rag_cases.jsonl` contains a question and one or more chunk
IDs that count as relevant. The benchmark searches for the question, keeps the
first `k` results, and compares those result IDs with the labels.

| Metric | Question it answers | BM25 at k=5 |
| --- | --- | --- |
| Hit@k | Did at least one relevant chunk appear? | <!-- fact benchmark.bm25_at_5.hit_rate -->0.5<!-- /fact --> |
| Recall@k | What fraction of accepted relevant chunks appeared? | <!-- fact benchmark.bm25_at_5.recall -->0.417<!-- /fact --> |
| MRR | How early did the first relevant chunk appear? | <!-- fact benchmark.bm25_at_5.mean_reciprocal_rank -->0.333<!-- /fact --> |

Hit@k is binary for each case. Recall@k matters when a question has multiple
accepted chunks. Reciprocal rank is one divided by the position of the first
relevant result; a miss contributes zero. MRR is the mean across all cases.

## Load labeled cases

1. In `eval/rag_retrieval.py`, inspect `RetrievalCase`:

    ```python
    @dataclass(frozen=True)
    class RetrievalCase:
        id: str
        query: str
        relevant_chunk_ids: frozenset[str]
    ```

    A set of accepted IDs lets one question recognize multiple source chunks as
    relevant without requiring every chunk to appear.

2. Inspect `load_cases()` and `validate_case_labels()`.

    The loader rejects duplicate case IDs and empty label sets. Label validation
    rejects chunk IDs that no longer exist after ingestion changes, preventing a
    stale benchmark from silently reporting false misses.

## Score one ranking

1. In `score_ranking()`, collect the one-based positions of relevant results:

    ```python
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
        if chunk_id in case.relevant_chunk_ids
    ]
    ```

2. Calculate whether the ranking contains a hit:

    ```python
    hit = bool(relevant_ranks)
    ```

3. Divide retrieved relevant chunks by all accepted relevant chunks:

    ```python
    recall = len(relevant_ranks) / len(case.relevant_chunk_ids)
    ```

4. Invert the first relevant rank, or return zero for a miss:

    ```python
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    ```

The repository implementation also ignores duplicate result IDs before scoring,
so one repeated chunk cannot inflate recall.

## Aggregate the cases

1. Run every labeled query through the same retrieval mode and cutoff.

2. Average each metric in `summarize()`:

    ```python
    hit_rate = sum(metric.hit for metric in case_metrics) / case_count
    recall_at_k = sum(metric.recall for metric in case_metrics) / case_count
    mean_reciprocal_rank = (
        sum(metric.reciprocal_rank for metric in case_metrics) / case_count
    )
    ```

3. Keep the missed case IDs beside the aggregate values.

    Aggregate metrics support comparisons, while named misses identify the
    questions and code paths that need investigation.

## Verify your work

Run the focused metric tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_retrieval_eval.py -q
```

<!-- verify expect match=contains -->

```text
6 passed
```

Check the benchmark and its tests with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check eval/rag_retrieval.py tests/test_rag_retrieval_eval.py
```

## Break it on purpose

Reduce the cutoff from five results to one:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m eval.rag_retrieval --top-k 1
```

<!-- verify expect match=contains -->

```text
Corpus: 267 chunks | Cases: 6 | k=1
bm25          6    0.167      0.167    0.167
bm25 misses: conversation-persistence, citation-deduplication, system-prompt-file, untrusted-rag-context, hybrid-rank-fusion
```

Hit@1 falls to
<!-- fact benchmark.bm25_at_1.hit_rate -->0.167<!-- /fact --> because only one
case puts a relevant chunk first. The experiment shows why every retrieval metric
must name its cutoff: Hit@1 and Hit@5 describe different user experiences.

## Troubleshooting

| Symptom | Resolution | Source |
| --- | --- | --- |
| `unknown relevant chunk ids` | Update stale labels only after confirming the intended replacement chunk. | Observed |
| Metrics differ from this guide | Run from the repository root and confirm the committed fact ledger is current. | Observed |
| Vector or hybrid mode reports missing indexes | Build indexes and provide a Voyage key, or use the offline BM25 default. | Predicted |
| One case unexpectedly has full recall | Check whether it has one accepted relevant chunk; one hit then equals full recall. | Predicted |

## Next steps

Use the benchmark to compare BM25 with vector and hybrid retrieval on the same
case set. Keyed comparisons belong in a protected environment because they use
Voyage embeddings and can incur cost.

For metric definitions and evaluation context, read the
[Information Retrieval evaluation documentation](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29).

## Tested against

| Item | Value |
| --- | --- |
| Python | 3.13.14; the repository requires 3.12 or later |
| pytest | 9.1.1 |
| Corpus | <!-- fact corpus.chunk_count -->267<!-- /fact --> chunks |
| Cases | <!-- fact benchmark.bm25_at_5.case_count -->6<!-- /fact --> |
| Verified | 2026-08-17 |

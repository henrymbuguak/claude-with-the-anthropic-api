# Implement Okapi BM25 from scratch

## What you'll build

In this tutorial, you build the keyword-ranking function used by this repository's local retrieval pipeline. You preserve complete code identifiers, calculate Okapi BM25 scores, and return deterministic ranked results without using a search library.

**Time:** About 45 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.
- [Chunk Python and Markdown for retrieval](chunk-python-and-markdown.md).
- Familiarity with the repository's [retrieval architecture](../architecture.md).

You do not need an API key. Every command in this tutorial runs locally.

## See it work

From the repository root, search for three identifiers used by the web-search configuration:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m app.rag.demo "allowed_domains max_uses WebSearchTool20260209Param" --top-k 3
```

<!-- verify expect match=ordered ignore_decimals=true -->

```text
Indexed 267 chunks
1. app/tools/web_search.py#function-build_web_search_tool (bm25 55.306)
2. tests/test_claude_client.py#function-test_stream_message_enables_configured_web_search (bm25 29.813)
3. app/tools/web_search.py#module-1 (bm25 25.372)
```

The implementation function ranks first because it contains all three identifiers. BM25 uses lexical evidence: it rewards matching terms but does not compare the meaning of two sentences.

## How BM25 works

BM25 assigns each query term a score in each document, then adds those term scores. This tutorial uses a source-code chunk as a document.

Three ideas control the score:

- **Term frequency** rewards a term that appears more than once, with diminishing returns.
- **Inverse document frequency** rewards a term that appears in few chunks.
- **Length normalization** reduces the accidental advantage of long chunks.

For one term, the implementation calculates:

```text
IDF = log(1 + (N - document_frequency + 0.5) / (document_frequency + 0.5))

score = IDF * term_frequency * (k1 + 1)
        / (term_frequency + k1 * (1 - b + b * length / average_length))
```

`k1` controls term-frequency saturation. `b` controls length normalization. This project uses `k1=1.5` and `b=0.75`.

## Preserve the reference implementation

The finished implementation already exists so that you can run it before rebuilding it. Work on a disposable branch or temporary copy of the repository.

In PowerShell, preserve the file and clear the original:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```powershell
Copy-Item app/rag/index_bm25.py app/rag/index_bm25_reference.txt
Clear-Content app/rag/index_bm25.py
```

In Bash, run the equivalent commands:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```bash
cp app/rag/index_bm25.py app/rag/index_bm25_reference.txt
: > app/rag/index_bm25.py
```

## Build the tokenizer

1. In `app/rag/index_bm25.py`, add the module imports and token patterns:

    ```python
    """A small BM25 index with tokenization for prose and source code."""

    from __future__ import annotations

    import math
    import re
    from collections import Counter

    from app.rag.models import Chunk, SearchResult

    _TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
    _CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
    ```

2. Add `tokenize()` after the patterns:

    ```python
    def tokenize(text: str) -> list[str]:
        """Preserve identifiers while adding snake/camel-case components."""
        tokens: list[str] = []
        for match in _TOKEN.finditer(text):
            raw = match.group(0)
            tokens.append(raw.lower())
            parts = [
                part.lower()
                for snake_part in raw.split("_")
                for part in _CAMEL_BOUNDARY.split(snake_part)
                if part
            ]
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens
    ```

    The tokenizer stores both `WebSearchTool20260209Param` and its components. An exact identifier query and a natural `web search` query can therefore reach the same chunk.

3. Verify the split with a real identifier:

    <!-- verify cmd tier=offline -->

    ```powershell
    uv run python -c "from app.rag.index_bm25 import tokenize; print(tokenize('WebSearchTool20260209Param'))"
    ```

    <!-- verify expect match=exact -->

    ```text
    ['websearchtool20260209param', 'web', 'search', 'tool20260209', 'param']
    ```

## Build the index

1. Add the class and validate its parameters:

    ```python
    class BM25Index:
        """An in-memory Okapi BM25 index over immutable chunks."""

        def __init__(
            self,
            chunks: list[Chunk],
            *,
            k1: float = 1.5,
            b: float = 0.75,
        ) -> None:
            if k1 <= 0:
                raise ValueError("k1 must be greater than zero")
            if not 0 <= b <= 1:
                raise ValueError("b must be between zero and one")
    ```

2. In `__init__`, store the chunks and parameters:

    ```python
            self._chunks = list(chunks)
            self._k1 = k1
            self._b = b
    ```

3. Still in `__init__`, precompute term counts and document lengths:

    ```python
            self._term_frequencies = [
                Counter(tokenize(chunk.text)) for chunk in chunks
            ]
            self._document_lengths = [
                sum(counts.values()) for counts in self._term_frequencies
            ]
            self._average_document_length = (
                sum(self._document_lengths) / len(self._document_lengths)
                if self._document_lengths
                else 0.0
            )
    ```

4. Count each term once per document when calculating document frequency:

    ```python
            self._document_frequencies = Counter(
                term for counts in self._term_frequencies for term in counts
            )
    ```

    Iterating over a `Counter` yields its keys. A term that appears ten times in one chunk therefore contributes one document to document frequency, not ten.

## Score one term

1. Add `_term_score()` and return early when the chunk does not contain the term:

    ```python
        def _term_score(
            self,
            term: str,
            frequencies: Counter[str],
            document_length: int,
        ) -> float:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                return 0.0
    ```

2. Calculate inverse document frequency:

    ```python
            document_count = len(self._chunks)
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
    ```

3. Add saturation and length normalization, then return the term score:

    ```python
            length_ratio = document_length / self._average_document_length
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * length_ratio
            )
            return inverse_document_frequency * (
                term_frequency * (self._k1 + 1) / denominator
            )
    ```

## Rank the chunks

1. Add `search()` with guards for invalid result counts and an empty corpus:

    ```python
        def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
            """Return positive-scoring chunks ordered by BM25 relevance."""
            if top_k < 1:
                raise ValueError("top_k must be greater than zero")
            if not self._chunks:
                return []
    ```

2. Tokenize the query and score every chunk:

    ```python
            query_terms = set(tokenize(query))
            scored: list[tuple[float, Chunk]] = []
            for chunk, frequencies, document_length in zip(
                self._chunks,
                self._term_frequencies,
                self._document_lengths,
                strict=True,
            ):
                score = sum(
                    self._term_score(term, frequencies, document_length)
                    for term in query_terms
                )
                if score > 0:
                    scored.append((score, chunk))
    ```

3. Sort by descending score and use the chunk ID as a deterministic tie-breaker:

    ```python
            scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
            return [
                SearchResult(chunk=chunk, score=score, rank=rank)
                for rank, (score, chunk) in enumerate(
                    scored[:top_k], start=1
                )
            ]
    ```

4. Confirm that the complete module imports:

    <!-- verify cmd tier=offline -->

    ```powershell
    uv run python -c "from app.rag.index_bm25 import BM25Index; print('ok')"
    ```

    <!-- verify expect match=exact -->

    ```text
    ok
    ```

## Verify your work

Run the focused tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_bm25.py -q
```

<!-- verify expect match=contains -->

```text
4 passed
```

Check the implementation and test with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check app/rag/index_bm25.py tests/test_rag_bm25.py
```

After your implementation matches the reference behavior, remove the temporary copy:

<!-- verify manual reason="Deletes the tutorial's temporary reference copy" -->

```powershell
Remove-Item app/rag/index_bm25_reference.txt
```

## Break it on purpose

BM25 matches vocabulary, not meaning. Run three formulations of the same question and request enough results to see the target chunk:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m app.rag.demo "conversation save load json messages role content" --top-k 40
uv run python -m app.rag.demo "How is conversation history saved?" --top-k 40
uv run python -m app.rag.demo "how does the bot remember what I said earlier" --top-k 40
```

<!-- verify expect match=ordered -->

```text
1. app/conversation.py#class-Conversation
16. app/conversation.py#class-Conversation
34. app/conversation.py#class-Conversation
```

The exact score is not the lesson and can shift when the corpus changes. The
ranking demonstrates the limitation: using implementation vocabulary places the
target at rank <!-- fact bm25.conversation_query_ranks.implementation_vocabulary -->1<!-- /fact -->,
while ordinary wording pushes the same target to ranks
<!-- fact bm25.conversation_query_ranks.natural_question -->16<!-- /fact --> and
<!-- fact bm25.conversation_query_ranks.synonym_question -->34<!-- /fact -->. A
semantic retriever addresses this vocabulary mismatch by comparing vector
representations rather than shared words.

## Troubleshooting

| Symptom                                       | Resolution                                                                   | Source    |
| --------------------------------------------- | ---------------------------------------------------------------------------- | --------- |
| `ImportError: cannot import name 'BM25Index'` | Complete the class definition or restore the reference file.                 | Observed  |
| `ModuleNotFoundError: No module named 'app'`  | Run the command from the repository root with `uv run`.                      | Observed  |
| Scores differ from this guide                 | Confirm the ranked chunk IDs; BM25 scores change with corpus composition.    | Observed  |
| Search results change between identical runs  | Include `chunk_id` in the sort key so ties are deterministic.                | Predicted |
| An identifier query returns no results        | Confirm that `tokenize()` stores the complete identifier and its components. | Predicted |

## Next steps

Continue with [Build and search a vector index](build-vector-index.md) to replace
lexical overlap with cosine similarity over vectors. The later fusion tutorial
then combines both rankings without comparing their incompatible raw scores.

For the ranking model's derivation, read [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf).

## Tested against

| Item     | Value                                          |
| -------- | ---------------------------------------------- |
| Python   | 3.13.14; the repository requires 3.12 or later |
| pytest   | 9.1.1                                          |
| Corpus   | <!-- fact corpus.chunk_count -->267<!-- /fact --> chunks, excluding guide documentation and tooling |
| Verified | 2026-08-17                                     |

"""A small BM25 index with tokenization suitable for prose and source code."""

from __future__ import annotations

import math
import re
from collections import Counter

from app.rag.models import Chunk, SearchResult

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Preserve complete identifiers while adding snake/camel-case components."""
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        raw = match.group(0)
        whole = raw.lower()
        tokens.append(whole)
        parts = [
            part.lower()
            for snake_part in raw.split("_")
            for part in _CAMEL_BOUNDARY.split(snake_part)
            if part
        ]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


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

        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._term_frequencies = [
            Counter(tokenize(chunk.text)) for chunk in chunks]
        self._document_lengths = [sum(counts.values())
                                  for counts in self._term_frequencies]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = Counter(
            term for counts in self._term_frequencies for term in counts
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return positive-scoring chunks ordered by BM25 relevance."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if not self._chunks:
            return []

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

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _term_score(
        self,
        term: str,
        frequencies: Counter[str],
        document_length: int,
    ) -> float:
        term_frequency = frequencies.get(term, 0)
        if not term_frequency:
            return 0.0

        document_count = len(self._chunks)
        document_frequency = self._document_frequencies[term]
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) /
            (document_frequency + 0.5)
        )
        length_ratio = document_length / self._average_document_length
        denominator = term_frequency + self._k1 * (
            1 - self._b + self._b * length_ratio
        )
        return inverse_document_frequency * (
            term_frequency * (self._k1 + 1) / denominator
        )

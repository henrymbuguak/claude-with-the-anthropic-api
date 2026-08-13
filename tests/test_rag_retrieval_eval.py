"""Tests for offline RAG retrieval metrics."""

import json

import pytest

from app.rag.models import Chunk, SearchResult
from eval.rag_retrieval import (
    RetrievalCase,
    evaluate_mode,
    load_cases,
    score_ranking,
    summarize,
    validate_case_labels,
)


class FakeRetriever:
    def search(self, query: str, *, mode: str, top_k: int) -> list[SearchResult]:
        chunk_id = "relevant" if query == "hit" else "irrelevant"
        chunk = Chunk(chunk_id, query, "source.py", "code")
        return [SearchResult(chunk, 1.0, 1)][:top_k]


def test_score_ranking_calculates_recall_and_reciprocal_rank() -> None:
    case = RetrievalCase(
        "history", "How is history saved?", frozenset({"a", "c"}))

    metrics = score_ranking(case, ["x", "a", "b", "c"])

    assert metrics.hit is True
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 0.5


def test_score_ranking_returns_zero_metrics_for_a_miss() -> None:
    case = RetrievalCase("history", "How is history saved?", frozenset({"a"}))

    metrics = score_ranking(case, ["x", "y"])

    assert metrics.hit is False
    assert metrics.recall == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_summarize_averages_case_metrics() -> None:
    first = score_ranking(
        RetrievalCase("first", "first", frozenset({"a"})), ["a"]
    )
    second = score_ranking(
        RetrievalCase("second", "second", frozenset({"b"})), ["x", "b"]
    )

    summary = summarize("bm25", 5, [first, second])

    assert summary.case_count == 2
    assert summary.hit_rate == 1.0
    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 0.75


def test_load_cases_rejects_duplicate_ids(tmp_path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    records = [
        {"id": "duplicate", "query": "one", "relevant_chunk_ids": ["a"]},
        {"id": "duplicate", "query": "two", "relevant_chunk_ids": ["b"]},
    ]
    cases_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(cases_path)


def test_validate_case_labels_rejects_stale_chunk_ids() -> None:
    cases = [RetrievalCase("stale", "query", frozenset({"missing"}))]

    with pytest.raises(ValueError, match="unknown relevant chunk ids: missing"):
        validate_case_labels(cases, [])


def test_evaluate_mode_runs_queries_and_aggregates_metrics() -> None:
    cases = [
        RetrievalCase("hit", "hit", frozenset({"relevant"})),
        RetrievalCase("miss", "miss", frozenset({"relevant"})),
    ]

    summary, metrics = evaluate_mode(FakeRetriever(), cases, "bm25", 5)

    assert summary.hit_rate == 0.5
    assert summary.mean_reciprocal_rank == 0.5
    assert [metric.hit for metric in metrics] == [True, False]

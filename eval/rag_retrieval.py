"""Offline benchmark for evaluating ranked RAG retrieval results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from app.rag.config import RagConfigError, RagSettings
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder
from app.rag.index_vector import VectorIndexError
from app.rag.ingest import ingest_path
from app.rag.models import Chunk
from app.rag.retriever import HybridRetriever

DEFAULT_CASES_PATH = Path(__file__).parent / "rag_cases.jsonl"


@dataclass(frozen=True)
class RetrievalCase:
    """One query with the chunk ids considered relevant to its answer."""

    id: str
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class CaseMetrics:
    """Ranking metrics for one retrieval case."""

    case_id: str
    hit: bool
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval metrics for one mode at a fixed cutoff."""

    mode: str
    top_k: int
    case_count: int
    hit_rate: float
    recall_at_k: float
    mean_reciprocal_rank: float


def load_cases(path: Path) -> list[RetrievalCase]:
    """Load and validate retrieval cases from a JSONL file."""
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = payload["id"]
            relevant_ids = frozenset(payload["relevant_chunk_ids"])
            if case_id in seen_ids:
                raise ValueError(
                    f"duplicate case id {case_id!r} on line {line_number}")
            if not relevant_ids:
                raise ValueError(f"case {case_id!r} has no relevant chunk ids")
            seen_ids.add(case_id)
            cases.append(
                RetrievalCase(
                    id=case_id,
                    query=payload["query"],
                    relevant_chunk_ids=relevant_ids,
                )
            )
    if not cases:
        raise ValueError("retrieval case file is empty")
    return cases


def score_ranking(case: RetrievalCase, ranked_chunk_ids: list[str]) -> CaseMetrics:
    """Score one ranked result list against all accepted relevant chunks."""
    seen_ids: set[str] = set()
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
        if chunk_id not in seen_ids
        and not seen_ids.add(chunk_id)
        and chunk_id in case.relevant_chunk_ids
    ]
    return CaseMetrics(
        case_id=case.id,
        hit=bool(relevant_ranks),
        recall=len(relevant_ranks) / len(case.relevant_chunk_ids),
        reciprocal_rank=1 / relevant_ranks[0] if relevant_ranks else 0.0,
    )


def summarize(
    mode: str, top_k: int, case_metrics: list[CaseMetrics]
) -> EvaluationSummary:
    """Average case metrics for one retrieval mode."""
    if not case_metrics:
        raise ValueError("cannot summarize an empty evaluation")
    case_count = len(case_metrics)
    return EvaluationSummary(
        mode=mode,
        top_k=top_k,
        case_count=case_count,
        hit_rate=sum(metric.hit for metric in case_metrics) / case_count,
        recall_at_k=sum(metric.recall for metric in case_metrics) / case_count,
        mean_reciprocal_rank=(
            sum(metric.reciprocal_rank for metric in case_metrics) / case_count
        ),
    )


def validate_case_labels(cases: list[RetrievalCase], chunks: list[Chunk]) -> None:
    """Fail when a labeled chunk no longer exists in the ingested corpus."""
    corpus_ids = {chunk.chunk_id for chunk in chunks}
    missing_ids = sorted(
        chunk_id
        for case in cases
        for chunk_id in case.relevant_chunk_ids
        if chunk_id not in corpus_ids
    )
    if missing_ids:
        raise ValueError(
            f"unknown relevant chunk ids: {', '.join(missing_ids)}")


def evaluate_mode(
    retriever: HybridRetriever,
    cases: list[RetrievalCase],
    mode: str,
    top_k: int,
) -> tuple[EvaluationSummary, list[CaseMetrics]]:
    """Run all labeled queries through one retrieval mode."""
    metrics = []
    for case in cases:
        results = retriever.search(case.query, mode=mode, top_k=top_k)
        metrics.append(
            score_ranking(case, [result.chunk.chunk_id for result in results])
        )
    return summarize(mode, top_k, metrics), metrics


def print_report(
    summaries: list[EvaluationSummary],
    details: dict[str, list[CaseMetrics]],
) -> None:
    """Print a compact comparison table and optional per-case misses."""
    print(f"{'Mode':<9} {'Cases':>5} {'Hit@k':>8} {'Recall@k':>10} {'MRR':>8}")
    for summary in summaries:
        print(
            f"{summary.mode:<9} {summary.case_count:>5} "
            f"{summary.hit_rate:>8.3f} {summary.recall_at_k:>10.3f} "
            f"{summary.mean_reciprocal_rank:>8.3f}"
        )
    for mode, case_metrics in details.items():
        misses = [metric.case_id for metric in case_metrics if not metric.hit]
        if misses:
            print(f"\n{mode} misses: {', '.join(misses)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local RAG retrieval modes.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("bm25", "vector", "hybrid"),
        default=["bm25"],
        help="Modes to compare (default: bm25)",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")

    chunks = ingest_path(args.root)
    cases = load_cases(args.cases)
    try:
        validate_case_labels(cases, chunks)
        embedder = None
        settings = RagSettings.from_env()
        if any(mode != "bm25" for mode in args.modes):
            settings = RagSettings.from_env(require_api_key=True)
            missing_indexes = [
                path.name
                for path in (
                    settings.index_dir / "code-index.json",
                    settings.index_dir / "prose-index.json",
                )
                if not path.exists()
            ]
            if missing_indexes:
                raise RagConfigError(
                    "build the missing indexes first: " +
                    ", ".join(missing_indexes)
                )
            embedder = VoyageEmbedder(settings)
        retriever = HybridRetriever.from_disk(chunks, settings, embedder)
        evaluations = [
            evaluate_mode(retriever, cases, mode, args.top_k) for mode in args.modes
        ]
    except (ValueError, RagConfigError, EmbeddingError, VectorIndexError) as exc:
        raise SystemExit(f"RAG evaluation error: {exc}") from exc

    print(
        f"Corpus: {len(chunks)} chunks | Cases: {len(cases)} | k={args.top_k}\n")
    print_report(
        [summary for summary, _ in evaluations],
        {summary.mode: metrics for summary, metrics in evaluations},
    )


if __name__ == "__main__":
    main()

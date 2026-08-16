"""Measure deterministic facts cited by the guide series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag.index_bm25 import BM25Index
from app.rag.ingest import ingest_path
from eval.rag_retrieval import DEFAULT_CASES_PATH, load_cases, score_ranking, summarize

_IDENTIFIER_QUERY = "allowed_domains max_uses WebSearchTool20260209Param"
_CONVERSATION_TARGET = "app/conversation.py#class-Conversation"
_CONVERSATION_QUERIES = {
    "implementation_vocabulary": "conversation save load json messages role content",
    "natural_question": "How is conversation history saved?",
    "synonym_question": "how does the bot remember what I said earlier",
}


def measure_facts(root: Path) -> dict:
    """Return the stable offline facts used by published guides."""
    root = root.resolve()
    chunks = ingest_path(root)
    index = BM25Index(chunks)
    identifier_results = index.search(_IDENTIFIER_QUERY, top_k=3)
    conversation_ranks = {}
    for name, query in _CONVERSATION_QUERIES.items():
        conversation_ranks[name] = next(
            (
                result.rank
                for result in index.search(query, top_k=len(chunks))
                if result.chunk.chunk_id == _CONVERSATION_TARGET
            ),
            None,
        )

    cases = load_cases(root / DEFAULT_CASES_PATH)
    benchmark = {}
    for top_k in (1, 5):
        case_metrics = []
        misses = []
        for case in cases:
            ranked = [
                result.chunk.chunk_id
                for result in index.search(case.query, top_k=top_k)
            ]
            metrics = score_ranking(case, ranked)
            case_metrics.append(metrics)
            if not metrics.hit:
                misses.append(case.id)
        summary = summarize("bm25", top_k, case_metrics)
        benchmark[f"bm25_at_{top_k}"] = {
            "case_count": summary.case_count,
            "hit_rate": round(summary.hit_rate, 3),
            "mean_reciprocal_rank": round(summary.mean_reciprocal_rank, 3),
            "misses": misses,
            "recall": round(summary.recall_at_k, 3),
        }

    return {
        "benchmark": benchmark,
        "bm25": {
            "conversation_query_ranks": conversation_ranks,
            "identifier_top_chunk_ids": [
                result.chunk.chunk_id for result in identifier_results
            ],
        },
        "corpus": {"chunk_count": len(chunks)},
    }


def render_facts(facts: dict) -> str:
    """Serialize facts deterministically for reviewable commits."""
    return json.dumps(facts, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--write", type=Path, help="Write the measured ledger")
    output.add_argument("--check", type=Path, help="Compare with a committed ledger")
    args = parser.parse_args()

    rendered = render_facts(measure_facts(args.root))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered, encoding="utf-8")
        print(f"Wrote measured facts to {args.write}")
        return 0

    committed = json.loads(args.check.read_text(encoding="utf-8"))
    measured = json.loads(rendered)
    if committed != measured:
        print(f"Fact ledger differs: {args.check}")
        return 1
    print(f"Fact ledger current: {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

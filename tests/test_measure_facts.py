"""Tests for the deterministic guide fact ledger."""

from pathlib import Path

from harness.measure_facts import measure_facts, render_facts


def test_measure_facts_matches_published_offline_baseline() -> None:
    facts = measure_facts(Path.cwd())

    assert facts["corpus"]["chunk_count"] == 267
    assert facts["bm25"]["identifier_top_chunk_ids"] == [
        "app/tools/web_search.py#function-build_web_search_tool",
        "tests/test_claude_client.py#function-test_stream_message_enables_configured_web_search",
        "app/tools/web_search.py#module-1",
    ]
    assert facts["bm25"]["conversation_query_ranks"] == {
        "implementation_vocabulary": 1,
        "natural_question": 16,
        "synonym_question": 34,
    }
    assert facts["benchmark"]["bm25_at_5"] == {
        "case_count": 6,
        "hit_rate": 0.5,
        "mean_reciprocal_rank": 0.333,
        "misses": [
            "conversation-persistence",
            "citation-deduplication",
            "system-prompt-file",
        ],
        "recall": 0.417,
    }
    assert facts["benchmark"]["bm25_at_1"] == {
        "case_count": 6,
        "hit_rate": 0.167,
        "mean_reciprocal_rank": 0.167,
        "misses": [
            "conversation-persistence",
            "citation-deduplication",
            "system-prompt-file",
            "untrusted-rag-context",
            "hybrid-rank-fusion",
        ],
        "recall": 0.167,
    }


def test_render_facts_is_stable_and_ends_with_newline() -> None:
    rendered = render_facts({"z": 1, "a": {"value": 2}})

    assert rendered == '{\n  "a": {\n    "value": 2\n  },\n  "z": 1\n}\n'

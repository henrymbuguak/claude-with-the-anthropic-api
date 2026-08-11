"""Tests for CLI response formatting."""

from app.claude_client import Citation
from main import format_sources


def test_format_sources_returns_empty_string_without_citations() -> None:
    assert format_sources([]) == ""


def test_format_sources_numbers_titles_and_falls_back_to_url() -> None:
    citations = [
        Citation(
            url="https://example.com/report",
            title="Example report",
            cited_text="Evidence",
        ),
        Citation(
            url="https://example.org/data",
            title=None,
            cited_text=None,
        ),
    ]

    assert format_sources(citations) == (
        "Sources:\n"
        "[1] Example report - https://example.com/report\n"
        "[2] https://example.org/data - https://example.org/data"
    )

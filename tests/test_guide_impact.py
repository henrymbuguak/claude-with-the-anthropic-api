import json
from pathlib import Path

from harness.guide_impact import impacted_guides, render_summary


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "guides": [
                    {
                        "id": 9,
                        "output": "docs/guides/chunking.md",
                        "primary_code": ["app/rag/ingest.py"],
                        "status": "published",
                        "title": "Chunk source files",
                    },
                    {
                        "id": 10,
                        "output": "docs/guides/bm25.md",
                        "primary_code": ["app/rag/index_bm25.py"],
                        "status": "draft",
                        "title": "Implement BM25",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_impact_reports_changed_primary_code_for_published_guides(tmp_path: Path) -> None:
    impacts = impacted_guides(
        _plan(tmp_path), ["app\\rag\\ingest.py", "tests/test_rag_ingest.py"]
    )

    assert impacts[0].guide_id == 9
    assert impacts[0].changed_paths == ("app/rag/ingest.py",)
    assert "[Chunk source files](docs/guides/chunking.md)" in render_summary(impacts)


def test_impact_ignores_unmapped_and_unpublished_guides(tmp_path: Path) -> None:
    impacts = impacted_guides(_plan(tmp_path), ["app/rag/index_bm25.py", "README.md"])

    assert impacts == ()
    assert "No published guides" in render_summary(impacts)
"""Tests for source-aware RAG ingestion."""

from pathlib import Path

import pytest

from app.rag.ingest import chunk_file, chunk_markdown, chunk_python, ingest_path


def test_chunk_python_preserves_module_code_decorators_and_definitions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.py"
    path.write_text(
        '"""Module docs."""\n\nVALUE = 1\n\n'
        "@decorator\n"
        "def greet(name: str) -> str:\n"
        '    return f"Hello {name}"\n\n'
        "class Greeter:\n"
        "    def greet(self) -> str:\n"
        '        return "Hello"\n',
        encoding="utf-8",
    )

    chunks = chunk_python(path, root=tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == [
        "sample.py#module-1",
        "sample.py#function-greet",
        "sample.py#class-Greeter",
    ]
    assert "VALUE = 1" in chunks[0].text
    assert chunks[1].text.startswith("@decorator")
    assert "class Greeter" in chunks[2].text


def test_chunk_markdown_uses_headings_and_disambiguates_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "Introduction.\n\n# Setup\nFirst.\n\n# Setup\nSecond.\n",
        encoding="utf-8",
    )

    chunks = chunk_markdown(path, root=tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == [
        "notes.md#heading-preamble",
        "notes.md#heading-setup",
        "notes.md#heading-setup-2",
    ]
    assert chunks[1].symbol == "Setup"
    assert chunks[2].text == "# Setup\nSecond."


def test_chunk_file_rejects_unsupported_files(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        chunk_file(path)


def test_ingest_path_discovers_supported_files_and_excludes_non_corpus_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "feature.py").write_text(
        "def feature():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Guide\nContent.\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Ignore me.", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(
        "# Documentation\nIgnore me.\n", encoding="utf-8"
    )
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "verify.py").write_text(
        "def verify():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_verify_guides.py").write_text(
        "def test_linter():\n    assert True\n", encoding="utf-8"
    )

    chunks = ingest_path(tmp_path)

    assert [chunk.source_path for chunk in chunks] == [
        "README.md",
        "app/feature.py",
    ]

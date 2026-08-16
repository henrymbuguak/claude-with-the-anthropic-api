"""Deterministic, source-aware chunking for Python and Markdown files."""

from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

from app.rag.models import Chunk

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".rag-index",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "docs",
    "dist",
    "harness",
}
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".py"}


def _source_path(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            resolved = resolved.relative_to(root.resolve())
        except ValueError:
            pass
    return PurePosixPath(resolved).as_posix()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _make_chunk(
    source_path: str,
    text: str,
    file_type: str,
    identity: str,
    symbol: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{source_path}#{identity}",
        text=text.strip(),
        source_path=source_path,
        file_type=file_type,
        symbol=symbol,
    )


def chunk_python(path: Path, root: Path | None = None) -> list[Chunk]:
    """Chunk a Python file at top-level function and class boundaries."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    source_path = _source_path(path, root)
    chunks: list[Chunk] = []
    next_module_line = 1

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in definitions:
        start_line = min(
            [node.lineno, *
                (decorator.lineno for decorator in node.decorator_list)]
        )
        module_text = "\n".join(
            lines[next_module_line - 1: start_line - 1]).strip()
        if module_text:
            chunks.append(
                _make_chunk(
                    source_path,
                    module_text,
                    "code",
                    f"module-{next_module_line}",
                    "<module>",
                )
            )

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        end_line = node.end_lineno or node.lineno
        chunks.append(
            _make_chunk(
                source_path,
                "\n".join(lines[start_line - 1: end_line]),
                "code",
                f"{kind}-{node.name}",
                node.name,
            )
        )
        next_module_line = end_line + 1

    trailing_text = "\n".join(lines[next_module_line - 1:]).strip()
    if trailing_text:
        chunks.append(
            _make_chunk(
                source_path,
                trailing_text,
                "code",
                f"module-{next_module_line}",
                "<module>",
            )
        )

    if not chunks and source.strip():
        chunks.append(
            _make_chunk(source_path, source, "code", "module-1", "<module>")
        )
    return chunks


def chunk_markdown(path: Path, root: Path | None = None) -> list[Chunk]:
    """Chunk Markdown into sections beginning at each heading."""
    lines = path.read_text(encoding="utf-8").splitlines()
    source_path = _source_path(path, root)
    chunks: list[Chunk] = []
    section_lines: list[str] = []
    section_name = "preamble"
    identity_counts: dict[str, int] = {}

    def append_section() -> None:
        text = "\n".join(section_lines).strip()
        if not text:
            return
        base_identity = f"heading-{_slug(section_name)}"
        identity_counts[base_identity] = identity_counts.get(
            base_identity, 0) + 1
        occurrence = identity_counts[base_identity]
        identity = base_identity if occurrence == 1 else f"{base_identity}-{occurrence}"
        chunks.append(
            _make_chunk(source_path, text, "prose", identity, section_name)
        )

    for line in lines:
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            append_section()
            section_lines = [line]
            section_name = heading.group(2)
        else:
            section_lines.append(line)
    append_section()
    return chunks


def chunk_file(path: Path, root: Path | None = None) -> list[Chunk]:
    """Dispatch a supported source file to its chunker."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return chunk_python(path, root)
    if suffix in {".md", ".markdown"}:
        return chunk_markdown(path, root)
    raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")


def ingest_path(root: Path) -> list[Chunk]:
    """Recursively chunk supported repository files in deterministic order."""
    chunks: list[Chunk] = []
    paths = sorted(root.rglob(
        "*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in _EXCLUDED_DIRECTORIES for part in relative_parts):
            continue
        chunks.extend(chunk_file(path, root=root))
    return chunks

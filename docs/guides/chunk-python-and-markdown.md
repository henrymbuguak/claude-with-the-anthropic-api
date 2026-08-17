# Chunk Python and Markdown for retrieval

## What you'll build

In this tutorial, you build the ingestion pipeline that turns this repository's
Python and Markdown files into small, retrievable units called chunks. You give
every chunk a stable, citable ID, and you assemble those chunks into the corpus
that later retrieval guides search.

**Time:** About 40 minutes.

## Before you begin

You need:

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/).
- A local clone with dependencies installed.

You do not need an API key. Every command in this tutorial runs locally.

## See it work

From the repository root, search for vocabulary that names the chunk builder itself:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m app.rag.demo "chunk_id source_path identity chunk_python chunk_markdown" --top-k 3
```

<!-- verify expect match=ordered ignore_decimals=true -->

```text
Indexed 267 chunks
1. app/rag/ingest.py#function-_make_chunk (bm25 30.047)
2. app/rag/ingest.py#function-chunk_markdown (bm25 30.002)
3. tests/test_rag_ingest.py#module-1 (bm25 28.474)
```

The top result, `app/rag/ingest.py#function-_make_chunk`, is the one function that
builds every chunk ID in this repository. The rest of this tutorial rebuilds the
module that produced this result and the 267 chunks it was chosen from.

## How chunking works

A **chunk** is a small, self-contained unit of source text that a retriever can
rank and a chat session can cite: one Python function, one class, the leftover
module-level code around them, or one Markdown section starting at a heading.
Splitting a file into chunks, instead of indexing the whole file, lets a search
return the one function that answers a question instead of an entire module.

Every chunk carries a **chunk ID**: a string built from its source path, a `#`
separator, and an identity such as `function-greet` or `heading-setup`. A chunk
ID is stable across runs, so the same identifier can move from a search result,
to a citation, to a spot check in the file, without translation.

Both chunkers in this tutorial build `Chunk` records, defined once in
`app/rag/models.py`:

```python
@dataclass(frozen=True)
class Chunk:
    """A source unit that can be indexed and cited."""

    chunk_id: str
    text: str
    source_path: str
    file_type: str
    symbol: str | None = None
```

This tutorial does not modify `Chunk`. It stays fixed so that the BM25, vector,
and rank-fusion guides that follow can all consume the same chunk shape.

The third concept is **corpus composition**: the rules that decide which files
become chunks at all. Not every file in this repository is source material.
Configuration, generated indexes, and the guide-generation tooling itself never
become chunks, so a search only ever returns application code and documentation
that a reader would recognize as content.

## Preserve the reference implementation

The finished implementation already exists so that you can run it before
rebuilding it. Work on a disposable branch or temporary copy of the repository.

In PowerShell, preserve the file and clear the original:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```powershell
Copy-Item app/rag/ingest.py app/rag/ingest_reference.txt
Clear-Content app/rag/ingest.py
```

In Bash, run the equivalent commands:

<!-- verify manual reason="Copies and clears a source file in the tutorial workspace" -->

```bash
cp app/rag/ingest.py app/rag/ingest_reference.txt
: > app/rag/ingest.py
```

## Build the path and identity helpers

1. In `app/rag/ingest.py`, add the module docstring, imports, and shared patterns:

    ```python
    """Deterministic, source-aware chunking for Python and Markdown files."""

    from __future__ import annotations

    import ast
    import re
    from pathlib import Path, PurePosixPath

    from app.rag.models import Chunk

    _MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    _EXCLUDED_DIRECTORIES = {
        ".claude",
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
    _EXCLUDED_FILES = {
        "test_generation_assets.py",
        "test_measure_facts.py",
        "test_prepare_generation.py",
        "test_validate_generation.py",
        "test_verify_facts.py",
        "test_verify_guides.py",
    }
    _SUPPORTED_SUFFIXES = {".md", ".markdown", ".py"}
    ```

    These constants define corpus composition before any file is read.
    `_SUPPORTED_SUFFIXES` limits the corpus to Python and Markdown.
    `_EXCLUDED_DIRECTORIES` removes tooling and generated paths such as `.venv`
    and `docs`. `_EXCLUDED_FILES` removes this project's own guide-generation
    tests, so the corpus never indexes the tooling that builds it.

2. Add `_source_path()`:

    ```python
    def _source_path(path: Path, root: Path | None) -> str:
        resolved = path.resolve()
        if root is not None:
            try:
                resolved = resolved.relative_to(root.resolve())
            except ValueError:
                pass
        return PurePosixPath(resolved).as_posix()
    ```

    `_source_path()` returns a forward-slash path relative to the corpus root,
    such as `app/rag/ingest.py`, regardless of the operating system running the
    ingestion.

3. Add `_slug()`:

    ```python
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "section"
    ```

    `_slug()` turns a Markdown heading such as "Getting Started" into
    `getting-started` for use inside a chunk ID.

4. Add `_make_chunk()`:

    ```python
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
    ```

    `_make_chunk()` assembles the chunk ID from the source path, a `#`
    separator, and an identity such as `function-greet` or `heading-setup`.
    Every chunker in this module calls `_make_chunk()`, so every chunk ID
    follows the same `path#identity` shape no matter which file type produced
    it.

## Chunk Python source at definition boundaries

1. Add `chunk_python()` and parse the file into an AST:

    ```python
    def chunk_python(path: Path, root: Path | None = None) -> list[Chunk]:
        """Chunk a Python file at top-level function and class boundaries."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        source_path = _source_path(path, root)
        chunks: list[Chunk] = []
        next_module_line = 1
    ```

    Parsing with [`ast`](https://docs.python.org/3/library/ast.html), instead of
    splitting on blank lines, means a chunk boundary always lands on a real
    function or class, even next to a multi-line string or a decorator.

2. Collect the top-level definitions in source order:

    ```python
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    ```

    Reading `tree.body`, instead of walking the entire tree, keeps nested
    functions and methods out of this list. They stay inside the text of their
    enclosing chunk instead of becoming chunks of their own.

3. Start the loop and emit any preceding module-level code as its own chunk:

    ```python
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
    ```

    Starting from the first decorator line, rather than the `def` or `class`
    line, keeps a decorator attached to the definition it modifies instead of
    leaking into the preceding module chunk.

4. Still inside the loop, emit the definition itself and advance past it:

    ```python
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
    ```

    A class chunk's identity is `class-<name>`, and a function chunk's identity
    is `function-<name>`. Both keep the definition's own name as the `symbol`.

5. After the loop, emit trailing module code and guard against files with no
    definitions:

    ```python
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
    ```

    The final guard keeps a non-empty file that defines no function or class,
    such as a constants-only module, as a single chunk instead of disappearing
    from the corpus.

## Chunk Markdown at heading boundaries

1. Add `chunk_markdown()` and its section state:

    ```python
    def chunk_markdown(path: Path, root: Path | None = None) -> list[Chunk]:
        """Chunk Markdown into sections beginning at each heading."""
        lines = path.read_text(encoding="utf-8").splitlines()
        source_path = _source_path(path, root)
        chunks: list[Chunk] = []
        section_lines: list[str] = []
        section_name = "preamble"
        identity_counts: dict[str, int] = {}
    ```

    Text before the first heading becomes a `preamble` section instead of being
    dropped.

2. Add the closure that turns buffered lines into a chunk:

    ```python
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
    ```

    `identity_counts` disambiguates repeated headings: the first "Setup"
    section becomes `heading-setup`, and a second "Setup" section becomes
    `heading-setup-2` instead of colliding with the first.

3. Walk the file, starting a new section at each heading:

    ```python
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
    ```

    The final `append_section()` call flushes the last section, which never
    reaches a following heading.

4. Confirm both chunkers split at the right boundaries:

    <!-- verify cmd tier=offline -->

    ```powershell
    uv run pytest tests/test_rag_ingest.py -v
    ```

    <!-- verify expect match=contains -->

    ```text
    tests/test_rag_ingest.py::test_chunk_python_preserves_module_code_decorators_and_definitions PASSED
    tests/test_rag_ingest.py::test_chunk_markdown_uses_headings_and_disambiguates_duplicates PASSED
    ```

## Dispatch by file type and assemble the corpus

1. Add `chunk_file()` to route a path to the matching chunker:

    ```python
    def chunk_file(path: Path, root: Path | None = None) -> list[Chunk]:
        """Dispatch a supported source file to its chunker."""
        suffix = path.suffix.lower()
        if suffix == ".py":
            return chunk_python(path, root)
        if suffix in {".md", ".markdown"}:
            return chunk_markdown(path, root)
        raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")
    ```

    An unsupported suffix raises immediately, so a mistaken call fails loudly
    instead of silently returning no chunks.

2. Add `ingest_path()` to walk the corpus root in deterministic order:

    ```python
    def ingest_path(root: Path) -> list[Chunk]:
        """Recursively chunk supported repository files in deterministic order."""
        chunks: list[Chunk] = []
        paths = sorted(root.rglob(
            "*"), key=lambda path: path.relative_to(root).as_posix())
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            if path.name in _EXCLUDED_FILES:
                continue
            relative_parts = path.relative_to(root).parts
            if any(part in _EXCLUDED_DIRECTORIES for part in relative_parts):
                continue
            chunks.extend(chunk_file(path, root=root))
        return chunks
    ```

    Sorting by relative path before chunking means two ingestion runs over an
    unchanged tree always produce chunks in the same order. `ingest_path()`
    checks the suffix, file name, and every parent directory name before
    calling `chunk_file()`, so an excluded path never reaches the point where
    an unsupported suffix could raise.

3. Confirm the assembled module reproduces the current corpus:

    <!-- verify cmd tier=offline -->

    ```powershell
    uv run python -m app.rag.demo "chunk_id source_path identity chunk_python chunk_markdown" --top-k 1
    ```

    <!-- verify expect match=contains -->

    ```text
    Indexed 267 chunks
    ```

    The repository's current combination of supported suffixes and excluded
    directories produces
    <!-- fact corpus.chunk_count -->267<!-- /fact --> chunks. Add, remove, or
    move a Python or Markdown file, and this count changes the next time
    `ingest_path()` runs.

## Verify your work

Run the focused tests:

<!-- verify cmd tier=offline -->

```powershell
uv run pytest tests/test_rag_ingest.py -q
```

<!-- verify expect match=contains -->

```text
4 passed
```

Check the implementation and tests with Ruff:

<!-- verify cmd tier=offline output=none -->

```powershell
uv run ruff check app/rag/ingest.py app/rag/models.py tests/test_rag_ingest.py
```

After your implementation matches the reference behavior, remove the temporary
copy:

<!-- verify manual reason="Deletes the tutorial's temporary reference copy" -->

```powershell
Remove-Item app/rag/ingest_reference.txt
```

## Break it on purpose

`chunk_python()` only splits at *top-level* definitions. A method defined inside
a class stays inside that class's single chunk. Search for vocabulary spread
across three different methods of the same class:

<!-- verify cmd tier=offline -->

```powershell
uv run python -m app.rag.demo "from_disk classmethod code_path prose_path top_k_per_retriever" --top-k 1
```

<!-- verify expect match=contains ignore_decimals=true -->

```text
1. app/rag/retriever.py#class-HybridRetriever
```

`__init__`, `from_disk`, and `search` are three separate methods in
`app/rag/retriever.py`, but they share one chunk ID:
`class-HybridRetriever`. A class with many methods becomes one large chunk, so
a query that names one specific method still returns the entire class, and a
retriever has no way to rank that one method above the rest of the class. This
tutorial's boundary rule is deliberately simple; ranking within a chunk, rather
than splitting it further, is the next tutorial's job.

## Troubleshooting

| Symptom | Resolution | Source |
| --- | --- | --- |
| `ImportError: cannot import name 'chunk_python'` | Complete the function definitions or restore the reference file. | Observed |
| `ModuleNotFoundError: No module named 'app'` | Run the command from the repository root with `uv run`. | Observed |
| Chunk count differs from this guide | Confirm no Python or Markdown files were added, removed, or moved; corpus composition depends on the current file tree. | Observed |
| Two Markdown sections produce the same chunk ID | Check whether `append_section()` runs before `section_lines` is reset for the new heading. | Predicted |
| `ingest_path()` returns fewer chunks than expected | Confirm the path is not under an excluded directory such as `.venv` or `docs`, and that its suffix is `.py`, `.md`, or `.markdown`. | Predicted |

## Next steps

The next tutorial ranks these chunks with Okapi BM25, so a search over this
same corpus returns the most relevant chunk first instead of in file order. Continue
with [Implement Okapi BM25 from scratch](guide-10.md).

For the module used to build chunk boundaries, read the
[`ast` module documentation](https://docs.python.org/3/library/ast.html).

## Tested against

| Item | Value |
| --- | --- |
| Python | 3.13.15; the repository requires 3.12 or later |
| pytest | 9.1.1 |
| Corpus | <!-- fact corpus.chunk_count -->267<!-- /fact --> chunks, excluding guide documentation and tooling |
| Verified | 2026-08-17 |

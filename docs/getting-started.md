# Get started

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- An Anthropic API key
- A Voyage AI API key only for vector or hybrid retrieval

## Install

Clone the repository and install its locked dependencies:

```powershell
uv sync
```

Copy the environment template and add your Anthropic key:

```powershell
Copy-Item .env.example .env
```

```dotenv
ANTHROPIC_API_KEY=your-key
```

Never commit `.env`. The repository tracks only the secret-free `.env.example` template.

## Run the CLI

```powershell
uv run python main.py
```

Enter a question at the prompt. Type `exit` or press Ctrl+C to stop. Conversation history is saved after every completed turn and restored on the next run.

## Enable local retrieval

BM25 mode runs locally and needs no additional API key:

```dotenv
RAG_ENABLED=true
RAG_CHAT_MODE=bm25
RAG_CHAT_TOP_K=5
```

Run the CLI again:

```powershell
uv run python main.py
```

The answer is followed by the local chunk IDs used as sources. Retrieved source text is sent only with the current request and is not written into conversation history.

## Enable hybrid retrieval

Add a Voyage AI key and build the local vector indexes:

```dotenv
VOYAGE_API_KEY=your-key
RAG_ENABLED=true
RAG_CHAT_MODE=hybrid
```

```powershell
uv run python -m app.rag.build_index
uv run python main.py
```

Generated indexes and the embedding cache are stored in `.rag-index/`, which is excluded from Git.

## Verify the project

The complete test suite uses mocked Anthropic and Voyage clients, so it makes no paid API calls:

```powershell
uv run pytest -q
uv run ruff check .
```

For the deterministic BM25 retrieval benchmark:

```powershell
uv run python -m eval.rag_retrieval
```

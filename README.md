# Build with Anthropic API

A small Python project for experimenting with the Anthropic (Claude) API, including multi-turn conversations and optional grounded web search.

## What's Here

An interactive, multi-turn chat CLI for Claude, split into small, testable modules:

- **`app/config.py`** — Loads and validates all settings from environment variables (`.env`) into a `Settings` dataclass. Fails fast with a clear error if `ANTHROPIC_API_KEY` is missing or a numeric setting is malformed.
- **`app/conversation.py`** — `Conversation` holds the message history (`add_user_message`/`add_assistant_message`) and can `save()`/`load()` it to/from JSON so a session can be resumed later.
- **`app/claude_client.py`** — `ClaudeChatClient` wraps the Anthropic SDK: streams responses token-by-token, optionally enables native web search, extracts citations and usage, and translates SDK exceptions (`RateLimitError`, `APIConnectionError`, `APIStatusError`) into a single friendly `ChatClientError`.
- **`app/tools/web_search.py`** — Builds the versioned Anthropic server-side web search definition from validated settings.
- **`main.py`** — The CLI entry point. Loads settings, resumes any saved conversation, then runs an input loop until the user types `exit` (or presses Ctrl+C).

## Project Layout

```
app/            # core library: config, conversation state, Claude API client
main.py         # CLI entry point
prompts/        # versioned system prompt files
eval/           # prompt evaluation harness (cases, evaluators, runner, results)
tests/          # unit tests (pytest)
```

## Testing

Unit tests cover `app/config.py`, `app/conversation.py`, `app/claude_client.py`, and `eval/evaluators.py`, mocking the Anthropic SDK so no real API calls are made:

```powershell
uv run pytest
```

## Dependencies

This project is managed with [uv](https://github.com/astral-sh/uv). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` (both committed for reproducible installs). A `requirements.txt` is also generated for anyone using plain `pip`:

```powershell
uv export --no-hashes --no-dev -o requirements.txt
```

## Setup

1. **Install dependencies** (using uv):

   ```powershell
   uv sync
   ```

   Or with pip:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configure your environment** — copy `.env.example` to `.env` and fill in your API key:

   ```powershell
   Copy-Item .env.example .env
   ```

   | Variable                               | Required | Default                     | Description                                              |
   | -------------------------------------- | -------- | --------------------------- | -------------------------------------------------------- |
   | `ANTHROPIC_API_KEY`                    | Yes      | —                           | Your Anthropic API key                                   |
   | `ANTHROPIC_MODEL`                      | No       | `claude-sonnet-4-6`         | Model used for chat completions                          |
   | `ANTHROPIC_MAX_TOKENS`                 | No       | `500`                       | Max tokens generated per response                        |
   | `ANTHROPIC_TEMPERATURE`                | No       | `1.0`                       | Sampling temperature                                     |
   | `ANTHROPIC_WEB_SEARCH_ENABLED`         | No       | `false`                     | Enable Anthropic's native server-side web search         |
   | `ANTHROPIC_WEB_SEARCH_MAX_USES`        | No       | `3`                         | Maximum web searches allowed in one API request          |
   | `ANTHROPIC_WEB_SEARCH_ALLOWED_DOMAINS` | No       | _(all)_                     | Optional comma-separated domain allowlist                |
   | `ANTHROPIC_SYSTEM_PROMPT_FILE`         | No       | _(none)_                    | Path to a versioned system prompt file (see `prompts/`)  |
   | `ANTHROPIC_SYSTEM_PROMPT`              | No       | _(none)_                    | Inline system prompt, used only if the file var is unset |
   | `LOG_LEVEL`                            | No       | `INFO`                      | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)  |
   | `CONVERSATION_HISTORY_FILE`            | No       | `conversation_history.json` | Where the conversation is saved/resumed from             |

   > Never commit your real API key. `.env` is already excluded via `.gitignore`; only `.env.example` (no secrets) is tracked.

3. **Run the CLI**:
   ```powershell
   uv run main.py
   ```
   Type a question, get a streamed reply, and keep chatting — type `exit` (or press Ctrl+C) to quit. The conversation is saved to `CONVERSATION_HISTORY_FILE` after every turn and automatically resumed on the next run.

## Web Search

Set `ANTHROPIC_WEB_SEARCH_ENABLED=true` to make Anthropic's native web search tool available to Claude. Claude automatically decides whether a question needs current web information; ordinary questions can still be answered without searching.

Anthropic executes this server-side tool, so the application does not call a separate search provider. When Claude cites web results, the CLI prints a numbered `Sources` list after the streamed answer. Search counts appear in debug logs when `LOG_LEVEL=DEBUG`.

Use `ANTHROPIC_WEB_SEARCH_MAX_USES` to limit searches per request. For research constrained to trusted sites, set a comma-separated allowlist such as:

```dotenv
ANTHROPIC_WEB_SEARCH_ALLOWED_DOMAINS=who.int,cdc.gov
```

Web search availability depends on the Anthropic account and model, and search requests may incur additional charges. Keep the feature disabled when it is not needed.

## Prompt Evaluation Workflow

System prompts are versioned as plain text files in `prompts/` (e.g. `prompts/scientist_v1.txt`) instead of being buried in `.env`, so changes are diffable in git and testable before being adopted.

- **`eval/cases.jsonl`** — a fixed set of test questions (JSONL), covering edge cases like emotionally charged topics, casual phrasing, and uncertain science.
- **`eval/evaluators.py`** — two evaluator types:
  - Rule-based checks (`run_rule_checks`): cheap, deterministic checks for banned hype words, exclamation-mark overuse, and response length.
  - LLM-as-judge (`run_llm_judge`): asks Claude to score a response 1–5 on `active_voice`, `calm_tone`, and `evidence_grounded`, with a justification — used for subjective/linguistic criteria that regex can't reliably catch.
- **`eval/run_eval.py`** — the harness. Runs every case in `eval/cases.jsonl` through a given prompt file, applies both evaluator types, prints a summary table, and saves a JSON report to `eval/results/`:
  ```powershell
  uv run eval/run_eval.py prompts/scientist_v1.txt
  ```
- **`eval/compare.py`** — runs the same case set against two prompt versions side-by-side, to check whether an edit actually improves scores before adopting it:
  ```powershell
  uv run eval/compare.py prompts/scientist_v1.txt prompts/scientist_v2.txt
  ```

To adopt a new prompt version, point `ANTHROPIC_SYSTEM_PROMPT_FILE` in `.env` at the new file once its eval scores look good.

## Notes

- Conversation history is kept as a list of `{"role": ..., "content": ...}` dicts, persisted to JSON, and replayed on each turn to `client.messages.create(...)`, which is how Claude "remembers" prior turns (the API itself is stateless).
- Streaming uses `client.messages.stream(...)`; token and web-search usage are logged at `DEBUG` level for cost visibility.
- API errors (rate limits, connection issues, non-2xx responses) surface as a single friendly `ChatClientError` instead of a raw SDK traceback, and the unanswered user turn is rolled back so the conversation stays consistent.
- `.gitignore` covers Python caches, virtual environments (`.venv`), `.env` secrets, and the local `conversation_history.json`, while `pyproject.toml`, `uv.lock`, `requirements.txt`, and `.env.example` are tracked for reproducible, secret-free setup.

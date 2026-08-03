# Build with Anthropic API

A small Python project for experimenting with the Anthropic (Claude) API, including multi-turn conversations.

## What's Here

- **`main.py`** — Demonstrates a multi-turn chat with Claude:
  - `add_user_message(messages, user_input)` — appends a user turn to the conversation history
  - `add_assistant_message(messages, assistant_response)` — appends an assistant turn to the conversation history
  - `chat(messages)` — sends the full message history to the Claude API and returns the reply text
  - Example flow: ask a question, capture the assistant's reply, add it back to history, then ask a follow-up so Claude retains context across turns

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

2. **Configure your API key** — create a `.env` file in the project root:

   ```dotenv
   ANTHROPIC_API_KEY=your-api-key-here
   ```

   > Never commit your real API key. `.env` is already excluded via `.gitignore`.

3. **Run the script**:
   ```powershell
   uv run main.py
   ```

## Model

The script currently uses `claude-sonnet-4-6` (configured via the `model` variable in [main.py](main.py)).

## Notes

- Conversation history is kept as a list of `{"role": ..., "content": ...}` dicts and passed to `client.messages.create(...)` on each turn, which is how Claude "remembers" prior turns (the API itself is stateless).
- `.gitignore` covers Python caches, virtual environments (`.venv`), and `.env` secrets so they aren't accidentally committed, while `pyproject.toml`, `uv.lock`, and `requirements.txt` are tracked for reproducible installs.

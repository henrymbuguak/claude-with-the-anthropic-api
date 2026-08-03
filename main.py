"""Interactive multi-turn chat CLI backed by the Anthropic Messages API."""

from __future__ import annotations

import logging
from pathlib import Path

from app.claude_client import ChatClientError, ClaudeChatClient
from app.config import ConfigError, Settings
from app.conversation import Conversation

logger = logging.getLogger(__name__)


def run_chat_loop(client: ClaudeChatClient, conversation: Conversation, history_path: Path) -> None:
    print("Type your question, or 'exit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding the conversation. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            break

        conversation.add_user_message(user_input)

        print("Claude: ", end="", flush=True)
        try:
            response = client.stream_message(
                conversation.messages,
                on_chunk=lambda chunk: print(chunk, end="", flush=True),
            )
        except ChatClientError as exc:
            print()
            logger.error("%s", exc)
            conversation.messages.pop()  # drop the unanswered user turn
            continue

        print("\n")
        logger.debug(
            "Tokens used - input: %s, output: %s", response.input_tokens, response.output_tokens
        )
        conversation.add_assistant_message(response.text)
        conversation.save(history_path)


def main() -> None:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    logging.basicConfig(level=settings.log_level,
                        format="%(levelname)s: %(message)s")
    if settings.log_level != "DEBUG":
        # Keep third-party HTTP client noise out of normal runs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    history_path = Path(settings.history_file)
    conversation = Conversation.load(history_path)
    client = ClaudeChatClient(settings)

    run_chat_loop(client, conversation, history_path)


if __name__ == "__main__":
    main()

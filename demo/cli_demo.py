"""Deterministic CLI fixture used only to render the README demo."""

import sys
import time

ANSWER = (
    "Conversation history is loaded from JSON when the CLI starts. After each "
    "successful turn, the user and assistant messages are saved back to the "
    "configured history file, so the next run can replay the conversation."
)


def stream(text: str) -> None:
    for word in text.split():
        print(f"{word} ", end="", flush=True)
        time.sleep(0.035)


def main() -> None:
    print("Type your question, or 'exit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding the conversation. Goodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            return

        print("Claude: ", end="", flush=True)
        stream(ANSWER)
        print("\n")
        print("Local sources:")
        print("[1] app/conversation.py#class-Conversation")
        print("[2] main.py#function-run_chat_loop\n")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
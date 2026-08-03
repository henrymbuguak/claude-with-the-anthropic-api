"""Conversation history management for multi-turn chats with Claude."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Conversation:
    """Holds the message history sent to the Anthropic Messages API."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def save(self, path: Path) -> None:
        """Persist the conversation to disk as JSON so it can be resumed later."""
        path.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Conversation":
        """Load a previously saved conversation, or start a fresh one if none exists."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        return cls(messages=data)

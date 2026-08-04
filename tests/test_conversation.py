"""Tests for app.conversation.Conversation."""

from __future__ import annotations

import json
from pathlib import Path

from app.conversation import Conversation


def test_add_user_message_appends_role_and_content() -> None:
    conversation = Conversation()

    conversation.add_user_message("Hello")

    assert conversation.messages == [{"role": "user", "content": "Hello"}]


def test_add_assistant_message_appends_role_and_content() -> None:
    conversation = Conversation()

    conversation.add_assistant_message("Hi there")

    assert conversation.messages == [
        {"role": "assistant", "content": "Hi there"}]


def test_save_writes_messages_as_json(tmp_path: Path) -> None:
    conversation = Conversation()
    conversation.add_user_message("Hello")
    conversation.add_assistant_message("Hi there")
    path = tmp_path / "history.json"

    conversation.save(path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == conversation.messages


def test_load_returns_empty_conversation_when_file_missing(tmp_path: Path) -> None:
    conversation = Conversation.load(tmp_path / "does-not-exist.json")

    assert conversation.messages == []


def test_load_returns_empty_conversation_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not valid json", encoding="utf-8")

    conversation = Conversation.load(path)

    assert conversation.messages == []


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    original = Conversation()
    original.add_user_message("What is entropy?")
    original.add_assistant_message("A measure of disorder.")
    original.save(path)

    loaded = Conversation.load(path)

    assert loaded.messages == original.messages

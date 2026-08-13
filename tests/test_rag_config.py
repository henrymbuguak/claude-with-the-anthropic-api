"""Tests for Voyage semantic retrieval configuration."""

from pathlib import Path

import pytest

from app.rag.config import RagConfigError, RagSettings


@pytest.fixture(autouse=True)
def clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VOYAGE_API_KEY",
        "RAG_CODE_EMBEDDING_MODEL",
        "RAG_PROSE_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIMENSION",
        "RAG_EMBEDDING_BATCH_SIZE",
        "RAG_INDEX_DIR",
        "RAG_ENABLED",
        "RAG_CHAT_MODE",
        "RAG_CHAT_TOP_K",
        "RAG_CONTEXT_CHAR_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)


def test_rag_settings_defaults_do_not_require_api_key() -> None:
    settings = RagSettings.from_env()

    assert settings.voyage_api_key is None
    assert settings.code_embedding_model == "voyage-code-3"
    assert settings.prose_embedding_model == "voyage-4"
    assert settings.embedding_dimension == 1024
    assert settings.embedding_batch_size == 64
    assert settings.index_dir == Path(".rag-index")
    assert settings.enabled is False
    assert settings.chat_mode == "hybrid"
    assert settings.chat_top_k == 5
    assert settings.context_char_budget == 6000


def test_rag_settings_requires_key_only_for_semantic_usage() -> None:
    with pytest.raises(RagConfigError, match="VOYAGE_API_KEY"):
        RagSettings.from_env(require_api_key=True)


def test_rag_settings_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("RAG_CODE_EMBEDDING_MODEL", "code-model")
    monkeypatch.setenv("RAG_PROSE_EMBEDDING_MODEL", "prose-model")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSION", "512")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RAG_INDEX_DIR", "custom-index")

    settings = RagSettings.from_env(require_api_key=True)

    assert settings.voyage_api_key == "test-voyage-key"
    assert settings.code_embedding_model == "code-model"
    assert settings.prose_embedding_model == "prose-model"
    assert settings.embedding_dimension == 512
    assert settings.embedding_batch_size == 32
    assert settings.index_dir == Path("custom-index")


@pytest.mark.parametrize("value", ["bad", "0", "768"])
def test_rag_settings_rejects_invalid_dimension(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSION", value)

    with pytest.raises(RagConfigError, match="RAG_EMBEDDING_DIMENSION"):
        RagSettings.from_env()


@pytest.mark.parametrize("value", ["bad", "0", "1001"])
def test_rag_settings_rejects_invalid_batch_size(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", value)

    with pytest.raises(RagConfigError, match="RAG_EMBEDDING_BATCH_SIZE"):
        RagSettings.from_env()


def test_rag_settings_reads_chat_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_CHAT_MODE", "bm25")
    monkeypatch.setenv("RAG_CHAT_TOP_K", "3")
    monkeypatch.setenv("RAG_CONTEXT_CHAR_BUDGET", "2000")

    settings = RagSettings.from_env()

    assert settings.enabled is True
    assert settings.chat_mode == "bm25"
    assert settings.chat_top_k == 3
    assert settings.context_char_budget == 2000


@pytest.mark.parametrize("value", ["yes", "1", "enabled"])
def test_rag_settings_rejects_invalid_enabled_flag(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RAG_ENABLED", value)

    with pytest.raises(RagConfigError, match="RAG_ENABLED"):
        RagSettings.from_env()


def test_rag_settings_rejects_invalid_chat_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CHAT_MODE", "bad-mode")

    with pytest.raises(RagConfigError, match="RAG_CHAT_MODE"):
        RagSettings.from_env()


@pytest.mark.parametrize("value", ["bad", "0"])
def test_rag_settings_rejects_invalid_chat_top_k(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RAG_CHAT_TOP_K", value)

    with pytest.raises(RagConfigError, match="RAG_CHAT_TOP_K"):
        RagSettings.from_env()


@pytest.mark.parametrize("value", ["bad", "100"])
def test_rag_settings_rejects_invalid_context_char_budget(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RAG_CONTEXT_CHAR_BUDGET", value)

    with pytest.raises(RagConfigError, match="RAG_CONTEXT_CHAR_BUDGET"):
        RagSettings.from_env()

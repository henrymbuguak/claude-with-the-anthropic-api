"""Interactive multi-turn chat CLI backed by the Anthropic Messages API."""

from __future__ import annotations

import logging
from pathlib import Path

from app.claude_client import ChatClientError, Citation, ClaudeChatClient
from app.config import ConfigError, Settings
from app.conversation import Conversation
from app.rag.config import RagConfigError, RagSettings
from app.rag.context import build_augmented_message
from app.rag.embed_voyage import EmbeddingError, VoyageEmbedder
from app.rag.index_vector import VectorIndexError
from app.rag.ingest import ingest_path
from app.rag.models import SearchResult
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def format_sources(citations: list[Citation]) -> str:
    """Format web citations as a compact numbered source list."""
    if not citations:
        return ""
    lines = ["Sources:"]
    lines.extend(
        f"[{index}] {citation.title or citation.url} - {citation.url}"
        for index, citation in enumerate(citations, start=1)
    )
    return "\n".join(lines)


def format_local_sources(results: list[SearchResult]) -> str:
    """Format retrieved local chunk ids as a compact numbered source list."""
    if not results:
        return ""
    lines = ["Local sources:"]
    lines.extend(
        f"[{index}] {result.chunk.chunk_id}"
        for index, result in enumerate(results, start=1)
    )
    return "\n".join(lines)


def run_chat_loop(
    client: ClaudeChatClient,
    conversation: Conversation,
    history_path: Path,
    retriever: HybridRetriever | None = None,
    rag_settings: RagSettings | None = None,
) -> None:
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

        api_messages = conversation.messages
        local_sources: list[SearchResult] = []
        if retriever is not None and rag_settings is not None:
            try:
                local_sources = retriever.search(
                    user_input,
                    mode=rag_settings.chat_mode,
                    top_k=rag_settings.chat_top_k,
                )
            except (EmbeddingError, VectorIndexError) as exc:
                logger.error(
                    "RAG retrieval failed, answering without local context: %s", exc
                )
                local_sources = []
            if local_sources:
                augmented = build_augmented_message(
                    user_input, local_sources, rag_settings.context_char_budget
                )
                api_messages = conversation.messages[:-1] + [
                    {"role": "user", "content": augmented}
                ]

        print("Claude: ", end="", flush=True)
        try:
            response = client.stream_message(
                api_messages,
                on_chunk=lambda chunk: print(chunk, end="", flush=True),
            )
        except ChatClientError as exc:
            print()
            logger.error("%s", exc)
            conversation.messages.pop()  # drop the unanswered user turn
            continue

        print("\n")
        sources = format_sources(response.citations)
        if sources:
            print(f"{sources}\n")
        local_sources_text = format_local_sources(local_sources)
        if local_sources_text:
            print(f"{local_sources_text}\n")
        logger.debug(
            "Usage - input tokens: %s, output tokens: %s, web searches: %s",
            response.input_tokens,
            response.output_tokens,
            response.web_search_requests,
        )
        conversation.add_assistant_message(response.text)
        conversation.save(history_path)


def main() -> None:
    try:
        settings = Settings.from_env()
        rag_settings = RagSettings.from_env()
    except (ConfigError, RagConfigError) as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from exc

    logging.basicConfig(level=settings.log_level,
                        format="%(levelname)s: %(message)s")
    if settings.log_level != "DEBUG":
        # Keep third-party HTTP client noise out of normal runs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    retriever: HybridRetriever | None = None
    if rag_settings.enabled:
        try:
            chunks = ingest_path(Path.cwd())
            embedder = None
            if rag_settings.chat_mode != "bm25":
                if not rag_settings.voyage_api_key:
                    raise RagConfigError(
                        "VOYAGE_API_KEY is required for RAG_CHAT_MODE="
                        f"{rag_settings.chat_mode!r}"
                    )
                embedder = VoyageEmbedder(rag_settings)
            retriever = HybridRetriever.from_disk(
                chunks, rag_settings, embedder)
        except (RagConfigError, EmbeddingError, VectorIndexError) as exc:
            print(f"Configuration error: {exc}")
            raise SystemExit(1) from exc

    history_path = Path(settings.history_file)
    conversation = Conversation.load(history_path)
    client = ClaudeChatClient(settings)

    run_chat_loop(client, conversation, history_path, retriever, rag_settings)


if __name__ == "__main__":
    main()

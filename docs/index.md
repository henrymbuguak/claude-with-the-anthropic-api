# Claude Chat CLI with Hybrid RAG Retrieval

Build and inspect a multi-turn Claude CLI with streaming responses, native web search, hybrid local retrieval, source grounding, and prompt evaluation.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Explore the architecture](architecture.md){ .md-button }

<figure markdown="span">
  ![The CLI answering a question with retrieved local sources](cli-demo.gif){ width="900" }
  <figcaption>The deterministic demo follows the same prompts, streaming behavior, and local-source format as the real CLI.</figcaption>
</figure>

## What the repository demonstrates

- A streaming Anthropic Messages API client with persistent conversation history.
- Optional extended thinking, prompt caching, and native web search.
- AST-aware Python and heading-aware Markdown chunking.
- Okapi BM25 and exact cosine vector search implemented without a search framework.
- Reciprocal Rank Fusion across lexical and semantic rankings.
- Grounded prompts with citations, abstention behavior, and prompt-injection boundaries.
- Deterministic retrieval evaluation and LLM-as-judge prompt evaluation.

## Choose a path

| Goal                                           | Start here                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| Run the finished assistant                     | [Get started](getting-started.md)                                                     |
| Understand retrieval and generation boundaries | [Architecture](architecture.md)                                                       |
| Inspect the implementation                     | [Repository on GitHub](https://github.com/henrymbuguak/claude-with-the-anthropic-api) |

!!! note "Guide automation comes later"
This first documentation phase publishes stable project documentation only.
Executable tutorial verification and generated guides will be added after
their verification harness is independently tested.

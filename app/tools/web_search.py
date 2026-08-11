"""Configuration for Anthropic's server-side web search tool."""

from __future__ import annotations

from anthropic.types import WebSearchTool20260209Param

from app.config import Settings


def build_web_search_tool(settings: Settings) -> WebSearchTool20260209Param:
    """Build the versioned native web-search definition for an API request."""
    tool: WebSearchTool20260209Param = {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": settings.web_search_max_uses,
    }
    if settings.web_search_allowed_domains:
        tool["allowed_domains"] = settings.web_search_allowed_domains
    return tool

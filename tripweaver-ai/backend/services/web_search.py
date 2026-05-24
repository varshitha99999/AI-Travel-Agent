"""
Web Search Service
------------------
Uses DuckDuckGo (free, no API key) to search for travel-related information.
Results are filtered and formatted for the travel agent context.
"""

from __future__ import annotations

import re
from typing import List, Dict

from agent.error_handler import with_retry, ToolError


def _clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Run a DuckDuckGo text search and return a list of result dicts.
    Each dict has keys: title, url, snippet.
    Raises ToolError on failure.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ToolError("duckduckgo-search package is not installed. Run: pip install duckduckgo-search")

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        raise ToolError(f"DuckDuckGo search failed: {exc}") from exc

    results = []
    for item in raw:
        results.append({
            "title":   _clean(item.get("title", "")),
            "url":     item.get("href", ""),
            "snippet": _clean(item.get("body", "")),
        })
    return results


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No results found."

    lines = ["🔍 **Web Search Results:**\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"**[{i}] {r['title']}**")
        if r["snippet"]:
            lines.append(r["snippet"])
        if r["url"]:
            lines.append(f"🔗 {r['url']}")
        lines.append("")

    return "\n".join(lines).strip()


@with_retry(max_attempts=2, delay=1.0)
def get_web_search(query: str) -> str:
    """
    Search the web for travel information and return formatted results.
    Retries once on transient failures.
    """
    results = search_web(query, max_results=5)
    return format_search_results(results)

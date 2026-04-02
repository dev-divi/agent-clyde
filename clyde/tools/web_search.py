"""Web search tool — search the web and return results."""

from __future__ import annotations

from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web and return results. Uses DuckDuckGo (no API key needed). Returns titles, URLs, and snippets for the top results."
    category = "web"
    requires_permission = True

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return (default: 8)",
                required=False,
                default=8,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        max_results = kwargs.get("max_results", 8)

        # Try DuckDuckGo HTML search (no API key, no dependencies beyond httpx)
        try:
            return _ddg_search(query, max_results)
        except Exception as e:
            return f"Search failed: {e}"


def _ddg_search(query: str, max_results: int) -> str:
    """Search DuckDuckGo via their HTML interface."""
    import re
    import httpx

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    with httpx.Client(timeout=15, follow_redirects=True, headers=headers) as client:
        response = client.post(url, data={"q": query, "b": ""})
        response.raise_for_status()
        html = response.text

    # Parse results from DuckDuckGo HTML
    results = []

    # Match result blocks: <a class="result__a" href="...">title</a>
    # and <a class="result__snippet" ...>snippet</a>
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title) in enumerate(links[:max_results]):
        # Clean HTML tags from title and snippet
        clean_title = re.sub(r'<[^>]+>', '', title).strip()
        clean_snippet = ""
        if i < len(snippets):
            clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

        # Decode DuckDuckGo redirect URL
        clean_url = href
        if "uddg=" in href:
            import urllib.parse
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            if "uddg" in parsed:
                clean_url = parsed["uddg"][0]

        results.append(f"[{i+1}] {clean_title}\n    {clean_url}\n    {clean_snippet}")

    if not results:
        return f"No results found for: {query}"

    return f"Search results for: {query}\n\n" + "\n\n".join(results)

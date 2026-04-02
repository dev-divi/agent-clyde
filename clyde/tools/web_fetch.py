"""Web fetch tool — retrieve content from URLs."""

from __future__ import annotations

from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "Fetch content from a URL. Returns the response body as text."
    category = "web"
    requires_permission = True

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL to fetch",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        url = kwargs["url"]
        timeout = kwargs.get("timeout", 30)

        try:
            import httpx
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.text
                # Truncate very large responses
                if len(content) > 100_000:
                    content = content[:100_000] + "\n... (truncated)"
                return content
        except Exception as e:
            return f"Error fetching {url}: {e}"

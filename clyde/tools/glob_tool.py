"""Glob tool — find files by pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class GlobTool(BaseTool):
    name = "glob"
    description = "Find files matching a glob pattern. Returns matching file paths sorted by modification time."
    category = "search"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description='Glob pattern (e.g. "**/*.py", "src/**/*.ts")',
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Directory to search in (default: current directory)",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum results to return (default: 100)",
                required=False,
                default=100,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        pattern = kwargs["pattern"]
        base = Path(kwargs.get("path", ".")).resolve()
        limit = kwargs.get("limit", 100)

        if not base.exists():
            return f"Error: Directory not found: {base}"

        try:
            matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError as e:
            return f"Error: {e}"

        if not matches:
            return f"No files found matching '{pattern}' in {base}"

        total = len(matches)
        results = [str(m.relative_to(base)) for m in matches[:limit]]
        output = "\n".join(results)

        if total > limit:
            output += f"\n... ({total - limit} more matches)"

        return output

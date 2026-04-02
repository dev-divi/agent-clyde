"""Grep tool — search file contents with regex."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents using regex patterns. Returns matching lines with file paths and line numbers."
    category = "search"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="Regex pattern to search for",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="File or directory to search (default: current dir)",
                required=False,
            ),
            ToolParameter(
                name="glob",
                type="string",
                description='File pattern filter (e.g. "*.py", "*.ts")',
                required=False,
            ),
            ToolParameter(
                name="case_insensitive",
                type="boolean",
                description="Case-insensitive search (default: false)",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum results (default: 50)",
                required=False,
                default=50,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        pattern = kwargs["pattern"]
        base = Path(kwargs.get("path", ".")).resolve()
        file_glob = kwargs.get("glob", "**/*")
        case_insensitive = kwargs.get("case_insensitive", False)
        limit = kwargs.get("limit", 50)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        results: list[str] = []

        if base.is_file():
            files = [base]
        else:
            try:
                files = [f for f in base.glob(file_glob) if f.is_file()]
            except OSError as e:
                return f"Error: {e}"

        for filepath in files:
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = filepath.relative_to(base) if not base.is_file() else filepath.name
                    results.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        if not results:
            return f"No matches for '{pattern}'"

        output = "\n".join(results)
        if len(results) >= limit:
            output += f"\n... (limited to {limit} results)"
        return output

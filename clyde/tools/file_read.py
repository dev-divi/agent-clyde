"""File read tool — read files from the filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read a file's contents. Returns the file content with line numbers. Use this to understand code before modifying it."
    category = "filesystem"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Absolute or relative path to the file",
                required=True,
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description="Line number to start reading from (1-based)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of lines to read",
                required=False,
                default=2000,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        file_path = Path(kwargs["file_path"]).resolve()
        offset = max(1, kwargs.get("offset", 1))
        limit = kwargs.get("limit", 2000)

        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        if not file_path.is_file():
            return f"Error: Not a file: {file_path}"

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        lines = text.splitlines()
        total = len(lines)

        # Apply offset and limit
        start_idx = offset - 1
        end_idx = start_idx + limit
        selected = lines[start_idx:end_idx]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset):
            numbered.append(f"{i}\t{line}")

        result = "\n".join(numbered)
        if end_idx < total:
            result += f"\n... ({total - end_idx} more lines)"

        return result

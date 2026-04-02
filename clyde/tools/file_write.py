"""File write tool — create or overwrite files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories as needed."
    category = "filesystem"
    requires_permission = True

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to write",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="The content to write to the file",
                required=True,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        file_path = Path(kwargs["file_path"]).resolve()
        content = kwargs["content"]

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Successfully wrote {len(content)} bytes ({line_count} lines) to {file_path}"
        except OSError as e:
            return f"Error writing file: {e}"

"""File edit tool — surgical string replacements in files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class FileEditTool(BaseTool):
    name = "file_edit"
    description = "Make exact string replacements in a file. The old_string must match exactly (including whitespace). Use this for precise edits instead of rewriting entire files."
    category = "filesystem"
    requires_permission = True

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to edit",
                required=True,
            ),
            ToolParameter(
                name="old_string",
                type="string",
                description="The exact string to find and replace",
                required=True,
            ),
            ToolParameter(
                name="new_string",
                type="string",
                description="The replacement string",
                required=True,
            ),
            ToolParameter(
                name="replace_all",
                type="boolean",
                description="Replace all occurrences (default: false, fails if not unique)",
                required=False,
                default=False,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        file_path = Path(kwargs["file_path"]).resolve()
        old_string = kwargs["old_string"]
        new_string = kwargs["new_string"]
        replace_all = kwargs.get("replace_all", False)

        if not file_path.exists():
            return f"Error: File not found: {file_path}"

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading file: {e}"

        if old_string == new_string:
            return "Error: old_string and new_string are identical"

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}"
        if count > 1 and not replace_all:
            return f"Error: old_string found {count} times. Use replace_all=true or provide more context to make it unique."

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        try:
            file_path.write_text(new_content, encoding="utf-8")
            replacements = count if replace_all else 1
            return f"Successfully replaced {replacements} occurrence(s) in {file_path}"
        except OSError as e:
            return f"Error writing file: {e}"

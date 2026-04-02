"""Default tool registration — assembles the built-in tool set."""

from __future__ import annotations

from .bash import BashTool
from .file_edit import FileEditTool
from .file_read import FileReadTool
from .file_write import FileWriteTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .registry import ToolRegistry
from .todo import TodoWriteTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    registry.register(TodoWriteTool())
    return registry

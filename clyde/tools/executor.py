"""Tool executor — runs tools and collects results.

Step 7-8: detect tool_use blocks, permission-check, execute,
collect results, and prepare them for the next API turn.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from ..models import ToolResultBlock, ToolUseBlock
from .permissions import PermissionManager
from .registry import ToolRegistry

logger = logging.getLogger("clyde.tools")


class ToolExecutor:
    """Executes tool calls from the model's response."""

    def __init__(self, registry: ToolRegistry, permissions: PermissionManager):
        self.registry = registry
        self.permissions = permissions

    def execute_tool_calls(
        self, tool_uses: list[ToolUseBlock]
    ) -> list[ToolResultBlock]:
        """Execute a batch of tool calls and return results.

        Each tool_use block gets permission-checked and executed.
        Results are returned in the same order.
        """
        results: list[ToolResultBlock] = []

        for tu in tool_uses:
            result = self._execute_single(tu)
            results.append(result)

        return results

    def _execute_single(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        """Execute a single tool call with permission checking."""
        tool = self.registry.get(tool_use.name)

        if tool is None:
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Error: Unknown tool '{tool_use.name}'. Available tools: {', '.join(self.registry.get_names())}",
                is_error=True,
            )

        # Permission check
        if not self.permissions.can_execute(tool_use.name, tool_use.input):
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Permission denied for tool '{tool_use.name}'.",
                is_error=True,
            )

        # Execute
        start = time.monotonic()
        try:
            result = tool.execute(**tool_use.input)
            elapsed = time.monotonic() - start
            logger.debug(f"Tool {tool_use.name} completed in {elapsed:.2f}s")
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=str(result),
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"Tool {tool_use.name} failed after {elapsed:.2f}s: {e}")
            tb = traceback.format_exc()
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Error executing {tool_use.name}: {e}\n{tb}",
                is_error=True,
            )

"""Tool executor — runs tools and collects results.

Step 7-8: detect tool_use blocks, permission-check, validate inputs,
execute, collect results, and prepare them for the next API turn.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

from ..models import ToolParameter, ToolResultBlock, ToolUseBlock
from .base import BaseTool
from .permissions import PermissionManager
from .registry import ToolRegistry

logger = logging.getLogger("clyde.tools")

# Python type string → accepted Python types
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolExecutor:
    """Executes tool calls from the model's response."""

    def __init__(self, registry: ToolRegistry, permissions: PermissionManager):
        self.registry = registry
        self.permissions = permissions

    def execute_tool_calls(
        self, tool_uses: list[ToolUseBlock]
    ) -> list[ToolResultBlock]:
        """Execute a batch of tool calls and return results.

        Each tool_use block gets validated, permission-checked, and executed.
        Results are returned in the same order.
        """
        results: list[ToolResultBlock] = []

        for tu in tool_uses:
            result = self._execute_single(tu)
            results.append(result)

        return results

    def _execute_single(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        """Execute a single tool call with validation and permission checking."""
        tool = self.registry.get(tool_use.name)

        if tool is None:
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Error: Unknown tool '{tool_use.name}'. Available tools: {', '.join(self.registry.get_names())}",
                is_error=True,
            )

        # --- Input validation ---
        validation_error = _validate_tool_input(tool, tool_use.input)
        if validation_error:
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Validation error for '{tool_use.name}': {validation_error}",
                is_error=True,
            )

        # --- Permission check ---
        if not self.permissions.can_execute(tool_use.name, tool_use.input):
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Permission denied for tool '{tool_use.name}'.",
                is_error=True,
            )

        # --- Execute ---
        start = time.monotonic()
        try:
            result = tool.execute(**tool_use.input)
            elapsed = time.monotonic() - start
            logger.debug(f"Tool {tool_use.name} completed in {elapsed:.2f}s")
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=str(result),
            )
        except TypeError as e:
            # Catch argument mismatches (extra kwargs, wrong types at call site)
            elapsed = time.monotonic() - start
            logger.error(f"Tool {tool_use.name} argument error after {elapsed:.2f}s: {e}")
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Argument error for {tool_use.name}: {e}",
                is_error=True,
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


def _validate_tool_input(tool: BaseTool, input_data: dict[str, Any]) -> str | None:
    """Validate tool input against parameter definitions.

    Returns an error message string if invalid, None if valid.
    """
    params = tool.get_parameters()
    param_map: dict[str, ToolParameter] = {p.name: p for p in params}
    allowed_names = set(param_map.keys())

    # Check for required parameters
    for p in params:
        if p.required and p.name not in input_data:
            return f"Missing required parameter '{p.name}' ({p.description})"

    # Check for unknown parameters
    unknown = set(input_data.keys()) - allowed_names
    if unknown:
        return f"Unknown parameter(s): {', '.join(sorted(unknown))}. Expected: {', '.join(sorted(allowed_names))}"

    # Type check each provided parameter
    for key, value in input_data.items():
        param = param_map.get(key)
        if param is None:
            continue  # Already caught above

        expected_types = _TYPE_MAP.get(param.type)
        if expected_types and not isinstance(value, expected_types):
            actual = type(value).__name__
            return f"Parameter '{key}' expected type '{param.type}', got '{actual}'"

        # Enum validation
        if param.enum and value not in param.enum:
            return f"Parameter '{key}' must be one of {param.enum}, got '{value}'"

    return None

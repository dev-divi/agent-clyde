"""Bash tool — execute shell commands."""

from __future__ import annotations

import subprocess
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


class BashTool(BaseTool):
    name = "bash"
    description = "Execute a shell command and return its output. Use for system operations, git commands, running scripts, and anything that requires shell access."
    category = "execution"
    requires_permission = True

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The shell command to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds (default 120)",
                required=False,
                default=120,
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        command = kwargs["command"]
        timeout = kwargs.get("timeout", 120)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None,  # Uses current working directory
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")

            output = "\n".join(output_parts) if output_parts else "(no output)"
            # Truncate very long outputs
            if len(output) > 50_000:
                output = output[:50_000] + "\n... (truncated)"
            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

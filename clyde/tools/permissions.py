"""Tool permission system — gatekeeps tool execution."""

from __future__ import annotations

from typing import Callable


class PermissionManager:
    """Manages which tools can execute and prompts for approval when needed."""

    def __init__(
        self,
        mode: str = "prompt",
        allowed: list[str] | None = None,
        denied: list[str] | None = None,
        prompt_fn: Callable[[str, dict], bool] | None = None,
    ):
        self.mode = mode  # "auto" | "prompt" | "deny"
        self._allowed = set(allowed or [])
        self._denied = set(denied or [])
        self._session_allowed: set[str] = set()
        self._prompt_fn = prompt_fn or self._default_prompt

    def can_execute(self, tool_name: str, tool_input: dict) -> bool:
        """Check if a tool is allowed to execute.

        Returns True if allowed, False if denied.
        In "prompt" mode, asks the user for permission.
        """
        if tool_name in self._denied:
            return False

        if self.mode == "auto":
            return True

        if self.mode == "deny":
            return tool_name in self._allowed

        # "prompt" mode
        if tool_name in self._allowed or tool_name in self._session_allowed:
            return True

        approved = self._prompt_fn(tool_name, tool_input)
        if approved:
            self._session_allowed.add(tool_name)
        return approved

    def allow_tool(self, name: str) -> None:
        self._allowed.add(name)

    def deny_tool(self, name: str) -> None:
        self._denied.add(name)

    @staticmethod
    def _default_prompt(tool_name: str, tool_input: dict) -> bool:
        """Default permission prompt via stdin."""
        import json
        preview = json.dumps(tool_input, indent=2)[:500]
        print(f"\n⚡ Tool: {tool_name}")
        print(f"   Input: {preview}")
        response = input("   Allow? [y/N/a(lways)]: ").strip().lower()
        if response == "a":
            return True  # Caller adds to session_allowed
        return response in ("y", "yes")

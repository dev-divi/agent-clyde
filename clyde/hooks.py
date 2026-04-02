"""Post-sampling hooks — Step 10 of the agent loop.

Hooks that run after the model response is complete:
- Auto-compact long conversations
- Memory extraction and persistence
- Custom user-defined hooks
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import TurnResult

logger = logging.getLogger("clyde.hooks")


class HookRunner:
    """Manages and executes post-sampling hooks."""

    def __init__(self):
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def run_all(self, result: TurnResult, context: dict[str, Any]) -> None:
        """Run all registered hooks with the turn result."""
        for hook in self._hooks:
            try:
                hook.execute(result, context)
            except Exception as e:
                logger.error(f"Hook '{hook.name}' failed: {e}")


class Hook:
    """A named post-sampling hook."""

    def __init__(self, name: str, fn: Callable[[TurnResult, dict[str, Any]], None]):
        self.name = name
        self._fn = fn

    def execute(self, result: TurnResult, context: dict[str, Any]) -> None:
        self._fn(result, context)

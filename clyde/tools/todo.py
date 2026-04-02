"""Todo tool — task tracking within sessions.

Maintains a structured task list the agent can update as it works
through multi-step tasks. Persists to a JSON file so tasks survive
across turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ToolParameter
from .base import BaseTool


@dataclass
class TodoItem:
    content: str
    status: str  # "pending" | "in_progress" | "completed"
    id: int = 0


class TodoStore:
    """In-memory + file-backed todo list."""

    def __init__(self):
        self.items: list[TodoItem] = []
        self._next_id = 1

    def add(self, content: str) -> TodoItem:
        item = TodoItem(content=content, status="pending", id=self._next_id)
        self._next_id += 1
        self.items.append(item)
        return item

    def update_status(self, item_id: int, status: str) -> TodoItem | None:
        for item in self.items:
            if item.id == item_id:
                item.status = status
                return item
        return None

    def remove(self, item_id: int) -> bool:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                return True
        return False

    def clear_completed(self) -> int:
        before = len(self.items)
        self.items = [i for i in self.items if i.status != "completed"]
        return before - len(self.items)

    def render(self) -> str:
        if not self.items:
            return "No tasks."

        lines = []
        status_icons = {"pending": "○", "in_progress": "◉", "completed": "✓"}
        for item in self.items:
            icon = status_icons.get(item.status, "?")
            lines.append(f"  {icon} [{item.id}] {item.content} ({item.status})")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [{"id": i.id, "content": i.content, "status": i.status} for i in self.items]

    def from_dict(self, data: list[dict]) -> None:
        self.items = []
        for d in data:
            item = TodoItem(content=d["content"], status=d["status"], id=d.get("id", 0))
            self.items.append(item)
            if item.id >= self._next_id:
                self._next_id = item.id + 1


# Global store instance (shared across tool calls within a session)
_store = TodoStore()


def get_todo_store() -> TodoStore:
    """Get the global todo store."""
    return _store


class TodoWriteTool(BaseTool):
    name = "todo"
    description = (
        "Manage a task list. Actions: 'add' to create a task, 'update' to change status, "
        "'remove' to delete, 'list' to show all, 'clear' to remove completed tasks. "
        "Use this to track progress on multi-step work."
    )
    category = "planning"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform",
                required=True,
                enum=["add", "update", "remove", "list", "clear"],
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Task description (for 'add' action)",
                required=False,
            ),
            ToolParameter(
                name="id",
                type="integer",
                description="Task ID (for 'update' and 'remove' actions)",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="New status (for 'update' action)",
                required=False,
                enum=["pending", "in_progress", "completed"],
            ),
        ]

    def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]
        store = get_todo_store()

        if action == "list":
            return store.render()

        elif action == "add":
            content = kwargs.get("content", "")
            if not content:
                return "Error: 'content' is required for 'add' action."
            item = store.add(content)
            return f"Added task [{item.id}]: {item.content}\n\n{store.render()}"

        elif action == "update":
            item_id = kwargs.get("id")
            status = kwargs.get("status")
            if item_id is None or status is None:
                return "Error: 'id' and 'status' are required for 'update' action."
            item = store.update_status(item_id, status)
            if item is None:
                return f"Error: No task with ID {item_id}."
            return f"Updated [{item.id}] → {item.status}\n\n{store.render()}"

        elif action == "remove":
            item_id = kwargs.get("id")
            if item_id is None:
                return "Error: 'id' is required for 'remove' action."
            if store.remove(item_id):
                return f"Removed task [{item_id}].\n\n{store.render()}"
            return f"Error: No task with ID {item_id}."

        elif action == "clear":
            count = store.clear_completed()
            return f"Cleared {count} completed task(s).\n\n{store.render()}"

        return f"Unknown action: {action}"

"""Base class for all Clyde tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import ToolDefinition, ToolParameter


class BaseTool(ABC):
    """Base class that all tools must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this tool does."""
        ...

    @property
    def category(self) -> str:
        return "general"

    @property
    def requires_permission(self) -> bool:
        return False

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Define the tool's input parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result."""
        ...

    def get_definition(self) -> ToolDefinition:
        """Convert to a ToolDefinition for the system prompt and API."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
            category=self.category,
            requires_permission=self.requires_permission,
        )

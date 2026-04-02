"""Core data models for Clyde agent."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    MAX_TURNS = "max_turns"
    BUDGET_EXCEEDED = "budget_exceeded"
    USER_CANCEL = "user_cancel"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    CUSTOM = "custom"  # Any OpenAI-compatible endpoint


# ---------------------------------------------------------------------------
# Content blocks — mirrors Anthropic's content block types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: Role
    content: list[ContentBlock]
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Extract concatenated text from all TextBlocks."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        return self


# ---------------------------------------------------------------------------
# Turn result — what the agent loop returns per iteration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurnResult:
    message: Message
    usage: Usage
    stop_reason: StopReason
    tool_calls_made: int = 0
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]
    category: str = "general"
    requires_permission: bool = False

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema for API tool definitions."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_api_format(self) -> dict[str, Any]:
        """Format for Anthropic/OpenAI tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_schema(),
        }


# ---------------------------------------------------------------------------
# Stream events — typed events emitted during streaming
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


# Common event constructors
def event_message_start(session_id: str) -> StreamEvent:
    return StreamEvent("message_start", {"session_id": session_id})

def event_text_delta(text: str) -> StreamEvent:
    return StreamEvent("text_delta", {"text": text})

def event_tool_use(tool_name: str, tool_id: str) -> StreamEvent:
    return StreamEvent("tool_use", {"name": tool_name, "id": tool_id})

def event_tool_result(tool_id: str, result: str, is_error: bool = False) -> StreamEvent:
    return StreamEvent("tool_result", {"id": tool_id, "result": result, "is_error": is_error})

def event_message_stop(stop_reason: StopReason, usage: Usage) -> StreamEvent:
    return StreamEvent("message_stop", {
        "stop_reason": stop_reason.value,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    })

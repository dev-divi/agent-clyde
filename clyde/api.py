"""Multi-provider API streaming layer — Step 5 of the agent loop.

Abstracts Anthropic, OpenAI, Ollama, and custom endpoints behind
a single streaming interface. Tokens arrive as typed StreamEvents.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Generator

from .config import ProviderConfig
from .message import message_to_api
from .models import (
    ContentBlock,
    Message,
    Provider,
    Role,
    StreamEvent,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    Usage,
    event_message_start,
    event_message_stop,
    event_text_delta,
    event_tool_use,
    StopReason,
)

logger = logging.getLogger("clyde.api")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class APIClient(ABC):
    """Base class for provider API clients."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        session_id: str = "",
    ) -> Generator[StreamEvent, None, tuple[list[ContentBlock], Usage, StopReason]]:
        """Stream a response from the API.

        Yields StreamEvents as tokens arrive.
        Returns (content_blocks, usage, stop_reason) when complete.
        """
        ...

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> tuple[list[ContentBlock], Usage, StopReason]:
        """Non-streaming completion. Returns full response at once."""
        ...


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(APIClient):
    """Client for Anthropic's Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.config.resolve_api_key(),
            )
        return self._client

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        session_id: str = "",
    ) -> Generator[StreamEvent, None, tuple[list[ContentBlock], Usage, StopReason]]:
        client = self._get_client()

        api_messages = [message_to_api(m, "anthropic") for m in messages]
        api_tools = [t.to_api_format() for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        content_blocks: list[ContentBlock] = []
        usage = Usage()
        stop_reason = StopReason.END_TURN

        yield event_message_start(session_id)

        with client.messages.stream(**kwargs) as stream:
            current_text = ""
            current_tool: dict[str, Any] | None = None

            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        current_text = ""
                    elif block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name, "input_json": ""}
                        yield event_tool_use(block.name, block.id)

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_text += delta.text
                        yield event_text_delta(delta.text)
                    elif delta.type == "input_json_delta" and current_tool:
                        current_tool["input_json"] += delta.partial_json

                elif event.type == "content_block_stop":
                    if current_text:
                        content_blocks.append(TextBlock(text=current_text))
                        current_text = ""
                    if current_tool:
                        try:
                            tool_input = json.loads(current_tool["input_json"])
                        except json.JSONDecodeError:
                            tool_input = {}
                        content_blocks.append(ToolUseBlock(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            input=tool_input,
                        ))
                        current_tool = None

                elif event.type == "message_delta":
                    sr = getattr(event, "delta", None)
                    if sr and hasattr(sr, "stop_reason") and sr.stop_reason:
                        stop_reason = _map_anthropic_stop(sr.stop_reason)
                    u = getattr(event, "usage", None)
                    if u:
                        usage.output_tokens = getattr(u, "output_tokens", 0)

                elif event.type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        u = getattr(msg, "usage", None)
                        if u:
                            usage.input_tokens = getattr(u, "input_tokens", 0)
                            usage.cache_read_tokens = getattr(u, "cache_read_input_tokens", 0)
                            usage.cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0)

        yield event_message_stop(stop_reason, usage)
        return content_blocks, usage, stop_reason

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> tuple[list[ContentBlock], Usage, StopReason]:
        client = self._get_client()
        api_messages = [message_to_api(m, "anthropic") for m in messages]
        api_tools = [t.to_api_format() for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        response = client.messages.create(**kwargs)

        from .message import parse_anthropic_content
        content_blocks = parse_anthropic_content(
            [{"type": b.type, **b.model_dump()} for b in response.content]
        )
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        stop_reason = _map_anthropic_stop(response.stop_reason)
        return content_blocks, usage, stop_reason


def _map_anthropic_stop(reason: str) -> StopReason:
    mapping = {
        "end_turn": StopReason.END_TURN,
        "tool_use": StopReason.TOOL_USE,
        "max_tokens": StopReason.MAX_TOKENS,
    }
    return mapping.get(reason, StopReason.END_TURN)


# ---------------------------------------------------------------------------
# OpenAI-compatible client (works with OpenAI, Ollama, LM Studio, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient(APIClient):
    """Client for any OpenAI-compatible API (OpenAI, Ollama, LM Studio, vLLM, etc.)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {
                "api_key": self.config.resolve_api_key() or "not-needed",
            }
            base_url = self.config.resolve_base_url()
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        session_id: str = "",
    ) -> Generator[StreamEvent, None, tuple[list[ContentBlock], Usage, StopReason]]:
        client = self._get_client()

        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            converted = message_to_api(m, "openai")
            if isinstance(converted, list):
                api_messages.extend(converted)
            else:
                api_messages.append(converted)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description,
                    "parameters": t.to_schema(),
                }}
                for t in tools
            ]

        content_blocks: list[ContentBlock] = []
        usage = Usage()
        stop_reason = StopReason.END_TURN
        accumulated_text = ""
        tool_calls_acc: dict[int, dict] = {}

        yield event_message_start(session_id)

        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            if not chunk.choices:
                # Usage chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage.input_tokens = chunk.usage.prompt_tokens or 0
                    usage.output_tokens = chunk.usage.completion_tokens or 0
                continue

            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            if delta.content:
                accumulated_text += delta.content
                yield event_text_delta(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": "",
                        }
                        if tool_calls_acc[idx]["name"]:
                            yield event_tool_use(
                                tool_calls_acc[idx]["name"],
                                tool_calls_acc[idx]["id"],
                            )
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["arguments"] += tc.function.arguments
                    if tc.id and not tool_calls_acc[idx]["id"]:
                        tool_calls_acc[idx]["id"] = tc.id

            if finish:
                if finish == "tool_calls":
                    stop_reason = StopReason.TOOL_USE
                elif finish == "length":
                    stop_reason = StopReason.MAX_TOKENS
                else:
                    stop_reason = StopReason.END_TURN

        if accumulated_text:
            content_blocks.append(TextBlock(text=accumulated_text))

        for _, tc in sorted(tool_calls_acc.items()):
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            content_blocks.append(ToolUseBlock(id=tc["id"], name=tc["name"], input=args))

        yield event_message_stop(stop_reason, usage)
        return content_blocks, usage, stop_reason

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> tuple[list[ContentBlock], Usage, StopReason]:
        client = self._get_client()

        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            converted = message_to_api(m, "openai")
            if isinstance(converted, list):
                api_messages.extend(converted)
            else:
                api_messages.append(converted)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description,
                    "parameters": t.to_schema(),
                }}
                for t in tools
            ]

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        from .message import parse_openai_response
        content_blocks = parse_openai_response(choice.model_dump())
        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        finish = choice.finish_reason
        if finish == "tool_calls":
            stop_reason = StopReason.TOOL_USE
        elif finish == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.END_TURN

        return content_blocks, usage, stop_reason


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(config: ProviderConfig) -> APIClient:
    """Create the appropriate API client for the configured provider."""
    if config.provider == Provider.ANTHROPIC:
        return AnthropicClient(config)
    # OpenAI, Ollama, and Custom all use OpenAI-compatible API
    return OpenAICompatibleClient(config)

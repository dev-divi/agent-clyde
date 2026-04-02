"""Message creation — Step 2 of the agent loop.

Wraps user text, tool results, and assistant responses into the
internal Message format used throughout the pipeline.
"""

from __future__ import annotations

from .models import (
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def create_user_message(text: str) -> Message:
    """Create a user message from raw text input."""
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def create_assistant_message(content: list[ContentBlock]) -> Message:
    """Create an assistant message from content blocks."""
    return Message(role=Role.ASSISTANT, content=content)


def create_tool_result_message(results: list[ToolResultBlock]) -> Message:
    """Create a tool-result message from executed tool outputs."""
    return Message(role=Role.USER, content=results)


def create_system_message(text: str) -> Message:
    """Create a system message."""
    return Message(role=Role.SYSTEM, content=[TextBlock(text=text)])


# ---------------------------------------------------------------------------
# Serialization — convert to/from API wire format
# ---------------------------------------------------------------------------

def message_to_api(msg: Message, provider: str = "anthropic") -> dict:
    """Convert internal Message to provider API format.

    Anthropic and OpenAI use slightly different schemas for tool use.
    This normalizes to the target provider's expected shape.
    """
    if provider == "anthropic":
        return _to_anthropic(msg)
    return _to_openai(msg)


def _to_anthropic(msg: Message) -> dict:
    content = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif isinstance(block, ToolResultBlock):
            content.append({
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": block.is_error,
            })
    return {"role": msg.role.value, "content": content}


def _to_openai(msg: Message) -> dict:
    """Convert to OpenAI chat completion format."""
    # OpenAI uses a different structure for tool calls
    if msg.role == Role.ASSISTANT and msg.tool_uses:
        tool_calls = []
        for tu in msg.tool_uses:
            tool_calls.append({
                "id": tu.id,
                "type": "function",
                "function": {"name": tu.name, "arguments": _json_dumps(tu.input)},
            })
        text = msg.text
        return {
            "role": "assistant",
            "content": text if text else None,
            "tool_calls": tool_calls,
        }

    if msg.role == Role.USER and msg.tool_results:
        # OpenAI expects individual tool messages, but we batch them.
        # Return first result; caller should split if needed.
        results = []
        for tr in msg.tool_results:
            results.append({
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr.content,
            })
        return results  # type: ignore  — caller handles list

    return {"role": msg.role.value, "content": msg.text}


def _json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Deserialization — parse API response into internal blocks
# ---------------------------------------------------------------------------

def parse_anthropic_content(content_blocks: list[dict]) -> list[ContentBlock]:
    """Parse Anthropic API response content blocks."""
    result: list[ContentBlock] = []
    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            result.append(TextBlock(text=block["text"]))
        elif btype == "tool_use":
            result.append(ToolUseBlock(
                id=block["id"],
                name=block["name"],
                input=block.get("input", {}),
            ))
    return result


def parse_openai_response(choice: dict) -> list[ContentBlock]:
    """Parse OpenAI chat completion response."""
    msg = choice.get("message", choice)
    result: list[ContentBlock] = []

    if msg.get("content"):
        result.append(TextBlock(text=msg["content"]))

    for tc in msg.get("tool_calls", []):
        import json
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            args = {}
        result.append(ToolUseBlock(
            id=tc["id"],
            name=tc["function"]["name"],
            input=args,
        ))

    return result

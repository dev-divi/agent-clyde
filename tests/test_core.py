"""Core tests for Clyde agent."""

import pytest
from clyde.models import (
    TextBlock, ToolUseBlock, ToolResultBlock, Message, Role, Usage,
    ToolParameter, ToolDefinition,
)
from clyde.message import (
    create_user_message, create_assistant_message, create_tool_result_message,
    message_to_api, parse_anthropic_content, parse_openai_response,
)
from clyde.history import ConversationHistory
from clyde.config import ClydeConfig, ProviderConfig
from clyde.tools.registry import ToolRegistry
from clyde.tools.permissions import PermissionManager


# --- Models ---

def test_text_block():
    b = TextBlock(text="hello")
    assert b.text == "hello"
    assert b.type == "text"


def test_message_text():
    msg = Message(role=Role.USER, content=[TextBlock(text="hi"), TextBlock(text=" there")])
    assert msg.text == "hi there"


def test_message_tool_uses():
    tu = ToolUseBlock(id="t1", name="bash", input={"command": "ls"})
    msg = Message(role=Role.ASSISTANT, content=[TextBlock(text="ok"), tu])
    assert len(msg.tool_uses) == 1
    assert msg.tool_uses[0].name == "bash"


def test_usage_add():
    u1 = Usage(input_tokens=100, output_tokens=50)
    u2 = Usage(input_tokens=200, output_tokens=100)
    u1 += u2
    assert u1.input_tokens == 300
    assert u1.output_tokens == 150
    assert u1.total_tokens == 450


def test_tool_definition_schema():
    td = ToolDefinition(
        name="test",
        description="Test tool",
        parameters=[
            ToolParameter(name="x", type="string", description="input", required=True),
            ToolParameter(name="y", type="integer", description="optional", required=False, default=5),
        ],
    )
    schema = td.to_schema()
    assert schema["properties"]["x"]["type"] == "string"
    assert "x" in schema["required"]
    assert "y" not in schema["required"]
    assert schema["properties"]["y"]["default"] == 5


# --- Message ---

def test_create_user_message():
    msg = create_user_message("hello")
    assert msg.role == Role.USER
    assert msg.text == "hello"


def test_message_to_api_anthropic():
    msg = create_user_message("test")
    result = message_to_api(msg, "anthropic")
    assert result["role"] == "user"
    assert result["content"][0]["type"] == "text"


def test_message_to_api_openai():
    msg = create_user_message("test")
    result = message_to_api(msg, "openai")
    assert result["role"] == "user"
    assert result["content"] == "test"


def test_parse_anthropic_content():
    blocks = parse_anthropic_content([
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
    ])
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)
    assert isinstance(blocks[1], ToolUseBlock)


# --- History ---

def test_history_append():
    h = ConversationHistory()
    h.append(create_user_message("hi"))
    assert h.length == 1


def test_history_compact():
    h = ConversationHistory()
    for i in range(20):
        h.append(create_user_message(f"message {i}"))
    summary = h.compact(keep_last=5)
    assert summary is not None
    assert h.length == 6  # 1 summary + 5 kept


def test_history_clear():
    h = ConversationHistory()
    h.append(create_user_message("hi"))
    h.clear()
    assert h.length == 0


# --- Config ---

def test_config_defaults():
    cfg = ClydeConfig()
    assert cfg.provider.model == "claude-sonnet-4-20250514"
    assert cfg.agent.max_turns == 25


def test_config_from_dict():
    cfg = ClydeConfig._from_dict({
        "provider": {"provider": "ollama", "model": "llama3"},
        "agent": {"max_turns": 10},
    })
    assert cfg.provider.provider.value == "ollama"
    assert cfg.provider.model == "llama3"
    assert cfg.agent.max_turns == 10


# --- Tool Registry ---

def test_registry():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    assert len(reg) == 7
    assert reg.has("bash")
    assert reg.has("file_read")
    assert reg.get("bash") is not None


# --- Permissions ---

def test_permissions_auto():
    pm = PermissionManager(mode="auto")
    assert pm.can_execute("bash", {"command": "ls"}) is True


def test_permissions_deny():
    pm = PermissionManager(mode="deny", allowed=["file_read"])
    assert pm.can_execute("file_read", {}) is True
    assert pm.can_execute("bash", {"command": "rm -rf /"}) is False


def test_permissions_denied_list():
    pm = PermissionManager(mode="auto", denied=["bash"])
    assert pm.can_execute("bash", {}) is False
    assert pm.can_execute("file_read", {}) is True

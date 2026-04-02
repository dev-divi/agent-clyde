"""Core tests for Clyde agent."""

import json
import tempfile
from pathlib import Path

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
from clyde.tools.executor import ToolExecutor, _validate_tool_input
from clyde.session import SessionStore, _serialize_message, _deserialize_message


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Message creation & serialization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

def test_registry():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    assert len(reg) == 7
    assert reg.has("bash")
    assert reg.has("file_read")
    assert reg.get("bash") is not None


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Input validation (NEW)
# ---------------------------------------------------------------------------

def test_validate_missing_required():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    bash = reg.get("bash")
    # Missing required 'command' parameter
    err = _validate_tool_input(bash, {})
    assert err is not None
    assert "Missing required" in err
    assert "command" in err


def test_validate_unknown_param():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    bash = reg.get("bash")
    err = _validate_tool_input(bash, {"command": "ls", "nonexistent": True})
    assert err is not None
    assert "Unknown parameter" in err


def test_validate_wrong_type():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    bash = reg.get("bash")
    # 'command' should be string, not integer
    err = _validate_tool_input(bash, {"command": 42})
    assert err is not None
    assert "expected type" in err


def test_validate_correct_input():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    bash = reg.get("bash")
    err = _validate_tool_input(bash, {"command": "echo hello"})
    assert err is None


def test_validate_optional_params():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    fr = reg.get("file_read")
    # Only required param is file_path
    err = _validate_tool_input(fr, {"file_path": "/tmp/test.txt"})
    assert err is None
    # With optional params too
    err = _validate_tool_input(fr, {"file_path": "/tmp/test.txt", "offset": 10, "limit": 50})
    assert err is None


def test_validate_optional_wrong_type():
    from clyde.tools.defaults import create_default_registry
    reg = create_default_registry()
    fr = reg.get("file_read")
    # 'offset' should be integer, not string
    err = _validate_tool_input(fr, {"file_path": "/tmp/test.txt", "offset": "bad"})
    assert err is not None
    assert "expected type" in err


# ---------------------------------------------------------------------------
# Session persistence & resume (NEW)
# ---------------------------------------------------------------------------

def test_session_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        store.save("test_session", {"turn_count": 5, "session_id": "test_session"})
        data = store.load("test_session")
        assert data is not None
        assert data["turn_count"] == 5


def test_session_message_serialization_roundtrip():
    """Test that messages survive serialize → deserialize."""
    # User message
    user_msg = create_user_message("hello world")
    serialized = _serialize_message(user_msg)
    deserialized = _deserialize_message(serialized)
    assert deserialized.role == Role.USER
    assert deserialized.text == "hello world"

    # Assistant message with tool use
    assistant_msg = Message(
        role=Role.ASSISTANT,
        content=[
            TextBlock(text="I'll run that"),
            ToolUseBlock(id="tu_123", name="bash", input={"command": "ls"}),
        ],
    )
    serialized = _serialize_message(assistant_msg)
    deserialized = _deserialize_message(serialized)
    assert deserialized.role == Role.ASSISTANT
    assert deserialized.text == "I'll run that"
    assert len(deserialized.tool_uses) == 1
    assert deserialized.tool_uses[0].name == "bash"
    assert deserialized.tool_uses[0].input == {"command": "ls"}

    # Tool result message
    tool_msg = create_tool_result_message([
        ToolResultBlock(tool_use_id="tu_123", content="file1.py\nfile2.py"),
    ])
    serialized = _serialize_message(tool_msg)
    deserialized = _deserialize_message(serialized)
    assert len(deserialized.tool_results) == 1
    assert deserialized.tool_results[0].content == "file1.py\nfile2.py"


def test_session_full_save_and_load_messages():
    """Test save_full + load_messages for session resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))

        messages = [
            create_user_message("hello"),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="hi there")]),
            create_user_message("run ls"),
            Message(role=Role.ASSISTANT, content=[
                ToolUseBlock(id="tu_1", name="bash", input={"command": "ls"}),
            ]),
        ]
        store.save_full("resume_test", messages, {"turn_count": 2})

        loaded = store.load_messages("resume_test")
        assert loaded is not None
        assert len(loaded) == 4
        assert loaded[0].role == Role.USER
        assert loaded[0].text == "hello"
        assert loaded[3].tool_uses[0].name == "bash"


def test_session_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        store.save("session_old", {"session_id": "session_old", "turn_count": 1})
        import time
        time.sleep(0.05)
        store.save("session_new", {"session_id": "session_new", "turn_count": 3})
        latest = store.get_latest_session_id()
        assert latest == "session_new"


def test_session_load_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        assert store.load("nonexistent") is None
        assert store.load_messages("nonexistent") is None


# ---------------------------------------------------------------------------
# Plugin system (NEW)
# ---------------------------------------------------------------------------

def test_plugin_mcp_config_parse():
    from clyde.plugins import PluginManager
    configs = PluginManager.parse_mcp_config([
        {"name": "test", "command": ["echo"], "args": ["hello"]},
        {"name": "fs", "command": ["npx", "mcp-fs"], "env": {"HOME": "/tmp"}},
    ])
    assert len(configs) == 2
    assert configs[0].name == "test"
    assert configs[0].command == ["echo"]
    assert configs[1].env == {"HOME": "/tmp"}


def test_plugin_loader_empty_dir():
    from clyde.plugins import PluginLoader
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = PluginLoader(plugin_dirs=[Path(tmpdir)])
        tools = loader.discover()
        assert tools == []


def test_plugin_loader_discovers_tool():
    """Test that a plugin .py file gets discovered and loaded."""
    from clyde.plugins import PluginLoader
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_file = Path(tmpdir) / "hello_tool.py"
        plugin_file.write_text(
            "from clyde.tools.base import BaseTool\n"
            "from clyde.models import ToolParameter\n"
            "\n"
            "class HelloTool(BaseTool):\n"
            "    name = 'hello'\n"
            "    description = 'Says hello'\n"
            "    def get_parameters(self):\n"
            "        return [ToolParameter(name='name', type='string', description='Name')]\n"
            "    def execute(self, **kwargs):\n"
            "        return f\"Hello, {kwargs.get('name', 'world')}!\"\n",
            encoding="utf-8",
        )
        loader = PluginLoader(plugin_dirs=[Path(tmpdir)])
        tools = loader.discover()
        assert len(tools) == 1
        assert tools[0].name == "hello"
        assert tools[0].execute(name="Tyler") == "Hello, Tyler!"

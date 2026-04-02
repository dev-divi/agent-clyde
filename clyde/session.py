"""Session persistence — save and resume conversations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import (
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


@dataclass
class SessionMetadata:
    session_id: str
    created_at: float
    updated_at: float
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    model: str
    provider: str


class SessionStore:
    """Persists sessions to disk as JSON files."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict) -> Path:
        """Save session data to disk, including full message history for resume."""
        path = self.sessions_dir / f"{session_id}.json"
        data["updated_at"] = time.time()
        if "created_at" not in data:
            data["created_at"] = time.time()
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def save_full(self, session_id: str, messages: list[Message], metadata: dict) -> Path:
        """Save a session with full message serialization (for resume)."""
        serialized_messages = [_serialize_message(m) for m in messages]
        data = {
            **metadata,
            "session_id": session_id,
            "messages_full": serialized_messages,
            "updated_at": time.time(),
        }
        if "created_at" not in data:
            data["created_at"] = time.time()
        path = self.sessions_dir / f"{session_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def load(self, session_id: str) -> dict | None:
        """Load a session from disk."""
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def load_messages(self, session_id: str) -> list[Message] | None:
        """Load full message history from a saved session (for resume)."""
        data = self.load(session_id)
        if data is None:
            return None
        raw_messages = data.get("messages_full")
        if not raw_messages:
            return None
        try:
            return [_deserialize_message(m) for m in raw_messages]
        except (KeyError, TypeError, ValueError):
            return None

    def get_latest_session_id(self) -> str | None:
        """Get the most recently updated session ID."""
        sessions = self.list_sessions(limit=1)
        return sessions[0]["session_id"] if sessions else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions."""
        sessions = []
        for path in sorted(
            self.sessions_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "updated_at": data.get("updated_at", 0),
                    "turn_count": data.get("turn_count", 0),
                    "model": data.get("model", ""),
                })
            except (json.JSONDecodeError, OSError):
                continue
            if len(sessions) >= limit:
                break
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Message serialization / deserialization for full session resume
# ---------------------------------------------------------------------------

def _serialize_message(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a JSON-safe dict."""
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
    return {
        "role": msg.role.value,
        "content": content,
        "id": msg.id,
        "timestamp": msg.timestamp,
    }


def _deserialize_message(data: dict[str, Any]) -> Message:
    """Deserialize a dict back into a Message."""
    role = Role(data["role"])
    blocks: list[ContentBlock] = []
    for raw in data["content"]:
        btype = raw["type"]
        if btype == "text":
            blocks.append(TextBlock(text=raw["text"]))
        elif btype == "tool_use":
            blocks.append(ToolUseBlock(
                id=raw["id"], name=raw["name"], input=raw.get("input", {}),
            ))
        elif btype == "tool_result":
            blocks.append(ToolResultBlock(
                tool_use_id=raw["tool_use_id"],
                content=raw["content"],
                is_error=raw.get("is_error", False),
            ))
    return Message(
        role=role,
        content=blocks,
        id=data.get("id", ""),
        timestamp=data.get("timestamp", 0.0),
    )

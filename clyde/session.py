"""Session persistence — save and resume conversations."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .history import ConversationHistory


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
        """Save session data to disk."""
        path = self.sessions_dir / f"{session_id}.json"
        data["updated_at"] = time.time()
        if "created_at" not in data:
            data["created_at"] = time.time()
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

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions."""
        sessions = []
        for path in sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "updated_at": data.get("updated_at", 0),
                    "turn_count": data.get("turn_count", 0),
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

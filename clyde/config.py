"""Configuration management for Clyde agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import Provider

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 25
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_COMPACT_AFTER = 40  # messages before auto-compaction


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""  # For custom/Ollama endpoints
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    def resolve_api_key(self) -> str:
        """Resolve API key from config or environment."""
        if self.api_key:
            return self.api_key
        env_map = {
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.OLLAMA: "",  # Ollama doesn't need a key
            Provider.CUSTOM: "CLYDE_API_KEY",
        }
        env_var = env_map.get(self.provider, "")
        return os.environ.get(env_var, "") if env_var else ""

    def resolve_base_url(self) -> str:
        """Resolve base URL for the provider."""
        if self.base_url:
            return self.base_url
        url_map = {
            Provider.ANTHROPIC: "https://api.anthropic.com",
            Provider.OPENAI: "https://api.openai.com/v1",
            Provider.OLLAMA: "http://localhost:11434/v1",
            Provider.CUSTOM: os.environ.get("CLYDE_BASE_URL", "http://localhost:8080/v1"),
        }
        return url_map.get(self.provider, "")


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    max_turns: int = DEFAULT_MAX_TURNS
    compact_after: int = DEFAULT_COMPACT_AFTER
    auto_compact: bool = True
    stream: bool = True
    tool_permission_mode: str = "prompt"  # "auto" | "prompt" | "deny"
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session config
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    sessions_dir: Path = Path(".clyde_sessions")
    memory_dir: Path = Path(".clyde_memory")
    auto_save: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ClydeConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    identity_file: str = "CLYDE.md"  # Agent identity/instructions file

    @classmethod
    def load(cls, path: Path | None = None) -> ClydeConfig:
        """Load config from clyde.json, falling back to defaults."""
        if path is None:
            path = Path("clyde.json")
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls._from_dict(raw)
        except (json.JSONDecodeError, KeyError):
            return cls()

    @classmethod
    def _from_dict(cls, d: dict) -> ClydeConfig:
        cfg = cls()
        if "provider" in d:
            p = d["provider"]
            cfg.provider.provider = Provider(p.get("provider", "anthropic"))
            cfg.provider.model = p.get("model", cfg.provider.model)
            cfg.provider.api_key = p.get("api_key", "")
            cfg.provider.base_url = p.get("base_url", "")
            cfg.provider.temperature = p.get("temperature", cfg.provider.temperature)
            cfg.provider.max_tokens = p.get("max_tokens", cfg.provider.max_tokens)
        if "agent" in d:
            a = d["agent"]
            cfg.agent.max_turns = a.get("max_turns", cfg.agent.max_turns)
            cfg.agent.compact_after = a.get("compact_after", cfg.agent.compact_after)
            cfg.agent.auto_compact = a.get("auto_compact", cfg.agent.auto_compact)
            cfg.agent.stream = a.get("stream", cfg.agent.stream)
            cfg.agent.tool_permission_mode = a.get("tool_permission_mode", "prompt")
            cfg.agent.allowed_tools = a.get("allowed_tools", [])
            cfg.agent.denied_tools = a.get("denied_tools", [])
        if "session" in d:
            s = d["session"]
            cfg.session.sessions_dir = Path(s.get("sessions_dir", ".clyde_sessions"))
            cfg.session.memory_dir = Path(s.get("memory_dir", ".clyde_memory"))
            cfg.session.auto_save = s.get("auto_save", True)
        if "identity_file" in d:
            cfg.identity_file = d["identity_file"]
        return cfg

    def to_dict(self) -> dict:
        return {
            "provider": {
                "provider": self.provider.provider.value,
                "model": self.provider.model,
                "base_url": self.provider.base_url,
                "temperature": self.provider.temperature,
                "max_tokens": self.provider.max_tokens,
            },
            "agent": {
                "max_turns": self.agent.max_turns,
                "compact_after": self.agent.compact_after,
                "auto_compact": self.agent.auto_compact,
                "stream": self.agent.stream,
                "tool_permission_mode": self.agent.tool_permission_mode,
                "allowed_tools": self.agent.allowed_tools,
                "denied_tools": self.agent.denied_tools,
            },
            "session": {
                "sessions_dir": str(self.session.sessions_dir),
                "memory_dir": str(self.session.memory_dir),
                "auto_save": self.session.auto_save,
            },
            "identity_file": self.identity_file,
        }

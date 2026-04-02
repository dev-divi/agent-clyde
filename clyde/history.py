"""Conversation history — Step 3 of the agent loop.

Manages the in-memory conversation array. Handles appending,
compaction (context window management), and replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Message, Role, TextBlock, Usage


@dataclass
class ConversationHistory:
    """In-memory conversation history that grows over the session."""

    messages: list[Message] = field(default_factory=list)
    _total_usage: Usage = field(default_factory=Usage)

    @property
    def length(self) -> int:
        return len(self.messages)

    @property
    def usage(self) -> Usage:
        return self._total_usage

    def append(self, message: Message) -> None:
        """Push a message onto the conversation array."""
        self.messages.append(message)

    def add_usage(self, usage: Usage) -> None:
        """Track cumulative token usage."""
        self._total_usage += usage

    def get_messages(self) -> list[Message]:
        """Return full conversation history."""
        return list(self.messages)

    def get_last_n(self, n: int) -> list[Message]:
        """Return the last N messages."""
        return self.messages[-n:]

    def compact(self, keep_last: int = 10) -> str | None:
        """Compact history by summarizing older messages.

        Keeps the last `keep_last` messages intact and replaces
        earlier messages with a summary. Returns the summary text
        or None if no compaction was needed.
        """
        if len(self.messages) <= keep_last:
            return None

        # Messages to summarize
        to_summarize = self.messages[:-keep_last]
        kept = self.messages[-keep_last:]

        # Build a simple summary of compacted messages
        summary_parts = []
        for msg in to_summarize:
            role = msg.role.value
            text = msg.text[:200] if msg.text else "[tool interaction]"
            tool_count = len(msg.tool_uses)
            if tool_count:
                summary_parts.append(f"[{role}] used {tool_count} tool(s)")
            elif text:
                summary_parts.append(f"[{role}] {text}")

        summary_text = (
            "=== Conversation compacted ===\n"
            "Earlier messages summarized:\n"
            + "\n".join(summary_parts)
            + "\n=== End summary ==="
        )

        # Replace history: summary + kept messages
        summary_msg = Message(
            role=Role.USER,
            content=[TextBlock(text=summary_text)],
        )
        self.messages = [summary_msg] + kept
        return summary_text

    def replay(self) -> list[dict]:
        """Get all messages as simple dicts for logging/debugging."""
        result = []
        for msg in self.messages:
            result.append({
                "role": msg.role.value,
                "text": msg.text[:500] if msg.text else None,
                "tool_uses": len(msg.tool_uses),
                "tool_results": len(msg.tool_results),
                "timestamp": msg.timestamp,
            })
        return result

    def clear(self) -> None:
        """Reset history."""
        self.messages.clear()
        self._total_usage = Usage()

"""The Agent Loop — Clyde's core orchestrator.

Implements the full 11-step pipeline:
  1. Input        → receive user text
  2. Message      → wrap into Message format
  3. History      → append to conversation array
  4. System       → assemble system prompt
  5. API          → stream to LLM provider
  6. Tokens       → track token usage
  7. Tools?       → detect tool_use blocks
  8. Loop         → execute tools, re-submit if needed
  9. Render       → output to user
  10. Hooks       → post-sampling hooks (compact, memory)
  11. Await       → wait for next input
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Generator

from .api import APIClient, create_client
from .config import ClydeConfig
from .history import ConversationHistory
from .message import create_assistant_message, create_tool_result_message, create_user_message
from .models import (
    ContentBlock,
    Message,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TurnResult,
    Usage,
    event_message_start,
    event_message_stop,
    event_text_delta,
)
from .system import build_system_prompt
from .tools.defaults import create_default_registry
from .tools.executor import ToolExecutor
from .tools.permissions import PermissionManager
from .tools.registry import ToolRegistry

logger = logging.getLogger("clyde")


class Agent:
    """Clyde agent — the main orchestrator.

    Manages the full agent loop from user input through tool execution
    to rendered output.
    """

    def __init__(
        self,
        config: ClydeConfig | None = None,
        registry: ToolRegistry | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ):
        self.config = config or ClydeConfig.load()
        self.session_id = f"session_{uuid.uuid4().hex[:12]}"
        self.history = ConversationHistory()
        self.turn_count = 0

        # --- Tool system ---
        self.registry = registry or create_default_registry()
        self.permissions = PermissionManager(
            mode=self.config.agent.tool_permission_mode,
            allowed=self.config.agent.allowed_tools,
            denied=self.config.agent.denied_tools,
        )
        self.executor = ToolExecutor(self.registry, self.permissions)

        # --- API client ---
        self.client: APIClient = create_client(self.config.provider)

        # --- Event callback for rendering ---
        self._on_event = on_event or (lambda e: None)

        # --- Memory ---
        self._memory_entries: list[str] = []

        # --- System prompt cache ---
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        """Build or return cached system prompt."""
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                identity_file=self.config.identity_file,
                tool_definitions=self.registry.get_definitions(),
                memory_entries=self._memory_entries if self._memory_entries else None,
            )
        return self._system_prompt

    def invalidate_system_prompt(self) -> None:
        """Force system prompt rebuild on next turn."""
        self._system_prompt = None

    def load_memory(self, entries: list[str]) -> None:
        """Load memory entries for the system prompt."""
        self._memory_entries = entries
        self.invalidate_system_prompt()

    # -----------------------------------------------------------------------
    # The Agent Loop
    # -----------------------------------------------------------------------

    def submit(self, user_input: str) -> TurnResult:
        """Submit user input and run the full agent loop.

        This is the main entry point. It:
        1. Creates a user message
        2. Appends to history
        3. Streams to the API
        4. Detects and executes tool calls
        5. Loops until the model stops using tools or max turns hit
        6. Returns the final result
        """
        start = time.monotonic()

        # Step 1-2: Input → Message creation
        user_msg = create_user_message(user_input)

        # Step 3: History append
        self.history.append(user_msg)

        # Step 7-8: Tool loop
        total_usage = Usage()
        total_tool_calls = 0
        loop_turns = 0
        max_loop_turns = self.config.agent.max_turns

        while loop_turns < max_loop_turns:
            loop_turns += 1

            # Step 4: System prompt assembly (cached)
            system = self.system_prompt

            # Step 5-6: API streaming + token tracking
            content_blocks, usage, stop_reason = self._call_api(system)
            total_usage += usage

            # Create assistant message and append to history
            assistant_msg = create_assistant_message(content_blocks)
            self.history.append(assistant_msg)
            self.history.add_usage(usage)

            # Step 7: Tool detection
            tool_uses = assistant_msg.tool_uses
            if not tool_uses or stop_reason != StopReason.TOOL_USE:
                # No tools — we're done
                self.turn_count += 1
                elapsed = (time.monotonic() - start) * 1000

                # Step 10: Post-sampling hooks
                self._run_hooks()

                return TurnResult(
                    message=assistant_msg,
                    usage=total_usage,
                    stop_reason=stop_reason,
                    tool_calls_made=total_tool_calls,
                    duration_ms=elapsed,
                )

            # Step 8: Execute tools and loop
            self._on_event(StreamEvent("tool_execution_start", {
                "count": len(tool_uses),
                "tools": [tu.name for tu in tool_uses],
            }))

            results = self.executor.execute_tool_calls(tool_uses)
            total_tool_calls += len(results)

            for r in results:
                self._on_event(StreamEvent("tool_result", {
                    "id": r.tool_use_id,
                    "is_error": r.is_error,
                    "preview": r.content[:200],
                }))

            # Append tool results to history and loop
            tool_msg = create_tool_result_message(results)
            self.history.append(tool_msg)

        # Max turns exceeded
        elapsed = (time.monotonic() - start) * 1000
        self._run_hooks()

        return TurnResult(
            message=Message(role=Role.ASSISTANT, content=[
                TextBlock(text=f"[Clyde: Reached max loop turns ({max_loop_turns})]")
            ]),
            usage=total_usage,
            stop_reason=StopReason.MAX_TURNS,
            tool_calls_made=total_tool_calls,
            duration_ms=elapsed,
        )

    def _call_api(self, system: str) -> tuple[list[ContentBlock], Usage, StopReason]:
        """Call the API (streaming or non-streaming) and return results."""
        messages = self.history.get_messages()
        tools = self.registry.get_definitions()

        if self.config.agent.stream:
            return self._stream_api(system, messages, tools)
        else:
            return self.client.complete(system=system, messages=messages, tools=tools)

    def _stream_api(
        self,
        system: str,
        messages: list[Message],
        tools: list[Any],
    ) -> tuple[list[ContentBlock], Usage, StopReason]:
        """Stream from API, emitting events and collecting the result."""
        gen = self.client.stream(
            system=system,
            messages=messages,
            tools=tools,
            session_id=self.session_id,
        )

        # Consume the generator, forwarding events
        content_blocks = []
        usage = Usage()
        stop_reason = StopReason.END_TURN

        try:
            while True:
                event = next(gen)
                self._on_event(event)
        except StopIteration as e:
            # Generator returns (content_blocks, usage, stop_reason)
            if e.value:
                content_blocks, usage, stop_reason = e.value

        return content_blocks, usage, stop_reason

    # -----------------------------------------------------------------------
    # Step 10: Post-sampling hooks
    # -----------------------------------------------------------------------

    def _run_hooks(self) -> None:
        """Run post-sampling hooks: auto-compact, memory extraction."""
        # Auto-compact if conversation is getting long
        if (
            self.config.agent.auto_compact
            and self.history.length > self.config.agent.compact_after
        ):
            summary = self.history.compact(keep_last=self.config.agent.compact_after // 2)
            if summary:
                logger.info(f"Auto-compacted conversation (kept last {self.config.agent.compact_after // 2} messages)")
                self._on_event(StreamEvent("compact", {"summary_length": len(summary)}))

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    def get_session_data(self) -> dict:
        """Export session data for persistence."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "history": self.history.replay(),
            "usage": {
                "input_tokens": self.history.usage.input_tokens,
                "output_tokens": self.history.usage.output_tokens,
                "total_tokens": self.history.usage.total_tokens,
            },
        }

    def reset(self) -> None:
        """Reset agent state for a new session."""
        self.session_id = f"session_{uuid.uuid4().hex[:12]}"
        self.history.clear()
        self.turn_count = 0
        self.invalidate_system_prompt()

"""Response rendering — Step 9 of the agent loop.

Renders streamed responses and tool interactions to the terminal
using Rich for markdown formatting and styled output.
"""

from __future__ import annotations

import sys
from typing import TextIO

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .models import StreamEvent, StopReason, TurnResult


class Renderer:
    """Terminal renderer for Clyde agent output."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._current_text = ""
        self._live: Live | None = None

    def on_event(self, event: StreamEvent) -> None:
        """Handle a stream event and render it."""
        handler = getattr(self, f"_handle_{event.type}", None)
        if handler:
            handler(event.data)

    def _handle_message_start(self, data: dict) -> None:
        self._current_text = ""
        self._live = Live("", console=self.console, refresh_per_second=15)
        self._live.start()

    def _handle_text_delta(self, data: dict) -> None:
        text = data.get("text", "")
        self._current_text += text
        if self._live:
            self._live.update(Markdown(self._current_text))

    def _handle_tool_use(self, data: dict) -> None:
        name = data.get("name", "?")
        # Pause live rendering to show tool info
        if self._live and self._live.is_started:
            self._live.update(Markdown(self._current_text))
            self._live.stop()
            self._live = None
        self.console.print(
            Text(f"  >> {name}", style="bold yellow"),
        )

    def _handle_tool_execution_start(self, data: dict) -> None:
        tools = data.get("tools", [])
        count = data.get("count", 0)
        self.console.print(
            Text(f"\n  [{count} tool(s): {', '.join(tools)}]", style="dim cyan"),
        )

    def _handle_tool_result(self, data: dict) -> None:
        is_error = data.get("is_error", False)
        preview = data.get("preview", "")
        style = "red" if is_error else "dim green"
        prefix = "  !! " if is_error else "  <- "
        self.console.print(Text(f"{prefix}{preview}", style=style))

    def _handle_message_stop(self, data: dict) -> None:
        if self._live and self._live.is_started:
            self._live.update(Markdown(self._current_text))
            self._live.stop()
            self._live = None
        elif self._current_text:
            # If live wasn't active (tool calls happened), print remaining text
            self.console.print(Markdown(self._current_text))
        self._current_text = ""

    def _handle_compact(self, data: dict) -> None:
        length = data.get("summary_length", 0)
        self.console.print(
            Text(f"\n  [conversation compacted, {length} chars summarized]", style="dim"),
        )

    def render_turn_result(self, result: TurnResult) -> None:
        """Render a completed turn result (for non-streaming mode)."""
        text = result.message.text
        if text:
            self.console.print(Markdown(text))

        if result.tool_calls_made > 0:
            self.console.print(
                Text(f"\n  [{result.tool_calls_made} tool call(s)]", style="dim cyan"),
            )

    def render_usage(self, result: TurnResult) -> None:
        """Show token usage after a turn."""
        u = result.usage
        parts = []
        if u.input_tokens:
            parts.append(f"in:{u.input_tokens}")
        if u.output_tokens:
            parts.append(f"out:{u.output_tokens}")
        if result.duration_ms:
            parts.append(f"{result.duration_ms:.0f}ms")

        if parts:
            self.console.print(
                Text(f"  ({', '.join(parts)})", style="dim"),
            )

    def print_welcome(self) -> None:
        """Print Clyde welcome banner."""
        self.console.print(
            Panel(
                "[bold]Clyde[/bold] — lightweight AI agent\n"
                "[dim]Type your message. Ctrl+C to exit.[/dim]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def print_error(self, msg: str) -> None:
        self.console.print(Text(f"Error: {msg}", style="bold red"))

    def print_info(self, msg: str) -> None:
        self.console.print(Text(msg, style="dim"))

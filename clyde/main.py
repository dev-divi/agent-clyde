"""Clyde CLI entry point and REPL — Step 1 & 11 of the agent loop.

Provides the interactive terminal interface and command routing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .agent import Agent
from .config import ClydeConfig, ProviderConfig
from .memory.store import MemoryStore
from .models import Provider, StopReason
from .render import Renderer
from .session import SessionStore


def main():
    """Main entry point for Clyde CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "version":
        print(f"Clyde v{__version__}")
        return

    if args.command == "init":
        _init_project(args)
        return

    if args.command == "sessions":
        _list_sessions(args)
        return

    # Default: run the REPL
    _run_repl(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clyde",
        description="Clyde — lightweight, provider-agnostic AI agent",
    )
    parser.add_argument("--version", action="version", version=f"Clyde v{__version__}")
    parser.add_argument("--config", type=str, default="clyde.json", help="Config file path")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--provider", type=str, choices=["anthropic", "openai", "ollama", "custom"],
                        help="Override provider")
    parser.add_argument("--base-url", type=str, help="Override API base URL")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--auto-tools", action="store_true",
                        help="Auto-approve all tool executions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("prompt", nargs="*", help="Initial prompt (non-interactive mode)")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Show version")

    init_parser = subparsers.add_parser("init", help="Initialize clyde.json config")
    init_parser.add_argument("--provider", type=str, default="anthropic")
    init_parser.add_argument("--model", type=str)

    subparsers.add_parser("sessions", help="List saved sessions")

    return parser


def _run_repl(args: argparse.Namespace):
    """Run the interactive REPL."""
    console = Console()
    renderer = Renderer(console)

    # Load config with CLI overrides
    config = ClydeConfig.load(Path(args.config) if args.config else None)
    _apply_overrides(config, args)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    # Initialize stores
    memory_store = MemoryStore(config.session.memory_dir)
    session_store = SessionStore(config.session.sessions_dir)

    # Create agent with renderer as event handler
    agent = Agent(config=config, on_event=renderer.on_event)

    # Load memory into system prompt
    memory_entries = memory_store.get_context_entries()
    if memory_entries:
        agent.load_memory(memory_entries)

    # Non-interactive mode: single prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
        result = agent.submit(prompt)
        if not config.agent.stream:
            renderer.render_turn_result(result)
        renderer.render_usage(result)
        if config.session.auto_save:
            session_store.save(agent.session_id, agent.get_session_data())
        return

    # Interactive REPL
    renderer.print_welcome()

    provider_info = f"{config.provider.provider.value}/{config.provider.model}"
    renderer.print_info(f"  Provider: {provider_info}")
    renderer.print_info(f"  Tools: {len(agent.registry)} available")
    renderer.print_info(f"  Permissions: {config.agent.tool_permission_mode} mode")
    console.print()

    try:
        while True:
            try:
                user_input = _get_input(console)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input.strip():
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = _handle_command(user_input, agent, renderer, session_store, memory_store)
                if handled == "exit":
                    break
                if handled:
                    continue

            # Submit to agent
            try:
                result = agent.submit(user_input)
                if not config.agent.stream:
                    renderer.render_turn_result(result)
                renderer.render_usage(result)
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                renderer.print_error(str(e))
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    finally:
        # Save session on exit
        if config.session.auto_save:
            session_store.save(agent.session_id, agent.get_session_data())
            renderer.print_info(f"  Session saved: {agent.session_id}")


def _get_input(console: Console) -> str:
    """Get user input with prompt."""
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.history import InMemoryHistory

        if not hasattr(_get_input, "_history"):
            _get_input._history = InMemoryHistory()  # type: ignore

        return prompt(
            "clyde > ",
            history=_get_input._history,  # type: ignore
            multiline=False,
        )
    except ImportError:
        return input("clyde > ")


def _handle_command(
    cmd: str,
    agent: Agent,
    renderer: Renderer,
    session_store: SessionStore,
    memory_store: MemoryStore,
) -> str | bool:
    """Handle slash commands. Returns 'exit' to quit, True if handled, False if not."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "/q"):
        return "exit"

    if command == "/help":
        renderer.print_info(
            "Commands:\n"
            "  /help          Show this help\n"
            "  /exit           Exit Clyde\n"
            "  /reset          Reset conversation\n"
            "  /compact        Compact conversation history\n"
            "  /history        Show conversation summary\n"
            "  /sessions       List saved sessions\n"
            "  /memory         Show memory entries\n"
            "  /tools          List available tools\n"
            "  /model <name>   Switch model\n"
            "  /provider <p>   Switch provider\n"
            "  /usage          Show token usage\n"
            "  /config         Show current config\n"
        )
        return True

    if command == "/reset":
        agent.reset()
        renderer.print_info("Conversation reset.")
        return True

    if command == "/compact":
        summary = agent.history.compact()
        if summary:
            renderer.print_info(f"Compacted. Summary: {len(summary)} chars")
        else:
            renderer.print_info("Nothing to compact.")
        return True

    if command == "/history":
        replay = agent.history.replay()
        for entry in replay[-10:]:
            role = entry["role"]
            text = (entry.get("text") or "[tool interaction]")[:100]
            renderer.print_info(f"  [{role}] {text}")
        return True

    if command == "/sessions":
        sessions = session_store.list_sessions()
        if not sessions:
            renderer.print_info("No saved sessions.")
        for s in sessions:
            renderer.print_info(f"  {s['session_id']} ({s['turn_count']} turns)")
        return True

    if command == "/memory":
        entries = memory_store.load_all()
        if not entries:
            renderer.print_info("No memory entries.")
        for e in entries:
            renderer.print_info(f"  [{e.memory_type}] {e.name}: {e.description}")
        return True

    if command == "/tools":
        for t in agent.registry.get_all():
            perm = " [requires permission]" if t.requires_permission else ""
            renderer.print_info(f"  {t.name} ({t.category}){perm}")
        return True

    if command == "/model" and arg:
        agent.config.provider.model = arg
        agent.client = __import__("clyde.api", fromlist=["create_client"]).create_client(
            agent.config.provider
        )
        agent.invalidate_system_prompt()
        renderer.print_info(f"Model switched to: {arg}")
        return True

    if command == "/provider" and arg:
        try:
            agent.config.provider.provider = Provider(arg)
            agent.client = __import__("clyde.api", fromlist=["create_client"]).create_client(
                agent.config.provider
            )
            agent.invalidate_system_prompt()
            renderer.print_info(f"Provider switched to: {arg}")
        except ValueError:
            renderer.print_error(f"Unknown provider: {arg}")
        return True

    if command == "/usage":
        u = agent.history.usage
        renderer.print_info(
            f"  Input tokens:  {u.input_tokens}\n"
            f"  Output tokens: {u.output_tokens}\n"
            f"  Total tokens:  {u.total_tokens}\n"
            f"  Turns:         {agent.turn_count}"
        )
        return True

    if command == "/config":
        renderer.print_info(json.dumps(agent.config.to_dict(), indent=2))
        return True

    renderer.print_info(f"Unknown command: {command}. Type /help for commands.")
    return True


def _apply_overrides(config: ClydeConfig, args: argparse.Namespace):
    """Apply CLI argument overrides to config."""
    if args.model:
        config.provider.model = args.model
    if args.provider:
        config.provider.provider = Provider(args.provider)
    if args.base_url:
        config.provider.base_url = args.base_url
    if args.no_stream:
        config.agent.stream = False
    if args.auto_tools:
        config.agent.tool_permission_mode = "auto"


def _init_project(args: argparse.Namespace):
    """Initialize a clyde.json config file."""
    config = ClydeConfig()
    if hasattr(args, "provider") and args.provider:
        config.provider.provider = Provider(args.provider)
    if hasattr(args, "model") and args.model:
        config.provider.model = args.model

    path = Path("clyde.json")
    if path.exists():
        print(f"Config already exists: {path}")
        return

    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"Created {path}")

    # Create CLYDE.md if it doesn't exist
    clyde_md = Path("CLYDE.md")
    if not clyde_md.exists():
        clyde_md.write_text(
            "# CLYDE.md\n\n"
            "Agent instructions go here. Clyde reads this file at startup\n"
            "and includes it in the system prompt.\n",
            encoding="utf-8",
        )
        print(f"Created {clyde_md}")


def _list_sessions(args: argparse.Namespace):
    """List saved sessions."""
    config = ClydeConfig.load(Path(args.config) if args.config else None)
    store = SessionStore(config.session.sessions_dir)
    sessions = store.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    for s in sessions:
        print(f"  {s['session_id']} ({s['turn_count']} turns)")


def __main__():
    main()


if __name__ == "__main__":
    main()

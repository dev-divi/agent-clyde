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
from .api import create_client
from .config import ClydeConfig
from .memory.store import MemoryStore
from .models import Provider, StopReason, TextBlock, ToolResultBlock, ToolUseBlock
from .plugins import PluginManager
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
    parser.add_argument("--resume", type=str, metavar="SESSION_ID",
                        help="Resume a previous session (use 'latest' for most recent)")
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

    # Load plugins (MCP servers + Python plugins)
    plugin_manager = PluginManager()
    _load_plugins(config, agent, plugin_manager, renderer)

    # Load memory into system prompt
    memory_entries = memory_store.get_context_entries()
    if memory_entries:
        agent.load_memory(memory_entries)

    # Handle --resume
    if args.resume:
        _handle_resume(args.resume, agent, session_store, renderer)

    # Non-interactive mode: single prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
        result = agent.submit(prompt)
        if not config.agent.stream:
            renderer.render_turn_result(result)
        renderer.render_usage(result)
        _save_session(agent, session_store, config)
        return

    # Interactive REPL
    renderer.print_welcome()

    provider_info = f"{config.provider.provider.value}/{config.provider.model}"
    renderer.print_info(f"  Provider: {provider_info}")
    renderer.print_info(f"  Tools: {len(agent.registry)} available")
    renderer.print_info(f"  Permissions: {config.agent.tool_permission_mode} mode")
    if args.resume:
        renderer.print_info(f"  Resumed: {agent.session_id} ({agent.turn_count} turns)")
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
                handled = _handle_command(
                    user_input, agent, renderer, session_store, memory_store, plugin_manager,
                )
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
        _save_session(agent, session_store, config)
        renderer.print_info(f"  Session saved: {agent.session_id}")
        plugin_manager.shutdown()


def _save_session(agent: Agent, store: SessionStore, config: ClydeConfig) -> None:
    """Save session with full message history for resume capability."""
    if config.session.auto_save:
        store.save_full(
            agent.session_id,
            agent.history.get_messages(),
            agent.get_session_data(),
        )


def _handle_resume(
    session_arg: str,
    agent: Agent,
    store: SessionStore,
    renderer: Renderer,
) -> None:
    """Resume a previous session."""
    if session_arg == "latest":
        session_id = store.get_latest_session_id()
        if not session_id:
            renderer.print_error("No sessions found to resume.")
            return
    else:
        session_id = session_arg

    messages = store.load_messages(session_id)
    if messages is None:
        renderer.print_error(f"Could not load session '{session_id}' (missing or no full history).")
        return

    agent.resume_session(session_id, messages)
    renderer.print_info(f"  Resumed session: {session_id} ({len(messages)} messages, {agent.turn_count} turns)")


def _load_plugins(
    config: ClydeConfig,
    agent: Agent,
    plugin_manager: PluginManager,
    renderer: Renderer,
) -> None:
    """Load MCP servers and Python plugins into the agent's registry."""
    count = 0

    # Load MCP servers from config
    config_path = Path("clyde.json")
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            mcp_configs_raw = raw.get("mcp_servers", [])
            if mcp_configs_raw:
                mcp_configs = PluginManager.parse_mcp_config(mcp_configs_raw)
                for tool in plugin_manager.load_mcp_servers(mcp_configs):
                    agent.registry.register(tool)
                    count += 1
        except (json.JSONDecodeError, OSError, KeyError) as e:
            renderer.print_error(f"Failed to load MCP config: {e}")

    # Load Python plugins
    plugin_dirs = [Path(".clyde_plugins")]
    for tool in plugin_manager.load_plugins(plugin_dirs):
        agent.registry.register(tool)
        count += 1

    if count > 0:
        renderer.print_info(f"  Plugins: {count} external tool(s) loaded")
        agent.invalidate_system_prompt()


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
    plugin_manager: PluginManager,
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
            "  /help              Show this help\n"
            "  /exit              Exit Clyde\n"
            "  /reset             Reset conversation\n"
            "  /resume [id]       Resume a session (default: latest)\n"
            "  /compact           Compact conversation history\n"
            "  /history           Show conversation summary\n"
            "  /sessions          List saved sessions\n"
            "  /memory            Show memory entries\n"
            "  /tools             List available tools\n"
            "  /model <name>      Switch model\n"
            "  /provider <name>   Switch provider\n"
            "  /structured <msg>  Force JSON structured output\n"
            "  /usage             Show token usage\n"
            "  /config            Show current config\n"
            "  /todo              Show current task list\n"
            "  /diff              Show changes made this session\n"
            "  /export [path]     Export conversation to JSON file\n"
        )
        return True

    if command == "/reset":
        agent.reset()
        renderer.print_info("Conversation reset.")
        return True

    if command == "/resume":
        session_id = arg.strip() if arg.strip() else "latest"
        _handle_resume(session_id, agent, session_store, renderer)
        return True

    if command == "/compact":
        agent._smart_compact()
        renderer.print_info("Compaction complete.")
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
            model = s.get("model", "")
            model_str = f" [{model}]" if model else ""
            renderer.print_info(f"  {s['session_id']} ({s['turn_count']} turns){model_str}")
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
        agent.switch_provider(model=arg)
        renderer.print_info(f"Model switched to: {arg}")
        return True

    if command == "/provider" and arg:
        try:
            agent.switch_provider(provider_name=arg)
            renderer.print_info(f"Provider switched to: {arg}")
        except ValueError:
            renderer.print_error(f"Unknown provider: {arg}. Options: anthropic, openai, ollama, custom")
        return True

    if command == "/structured" and arg:
        # Force structured JSON output for the prompt
        schema = {"type": "object"}  # Generic schema — model decides structure
        result = agent.submit_structured(arg, schema)
        if result is not None:
            renderer.print_info(json.dumps(result, indent=2))
        else:
            renderer.print_error("Failed to get structured output.")
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

    if command == "/todo":
        from .tools.todo import get_todo_store
        renderer.print_info(get_todo_store().render())
        return True

    if command == "/diff":
        # Show tool-use activity from this session (files touched, commands run)
        messages = agent.history.get_messages()
        changes: list[str] = []
        for msg in messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    tool_name = block.name
                    inp = block.input
                    if tool_name in ("file_write", "file_edit"):
                        path = inp.get("file_path", inp.get("path", "?"))
                        changes.append(f"  ✏️  {tool_name}: {path}")
                    elif tool_name == "bash":
                        cmd_text = inp.get("command", "")[:80]
                        changes.append(f"  ▶  bash: {cmd_text}")
                    elif tool_name in ("file_read", "glob", "grep"):
                        target = inp.get("file_path", inp.get("pattern", inp.get("path", "")))
                        changes.append(f"  👁  {tool_name}: {target}")
        if changes:
            renderer.print_info(f"Session activity ({len(changes)} tool calls):\n" + "\n".join(changes))
        else:
            renderer.print_info("No tool activity in this session.")
        return True

    if command == "/export":
        # Export conversation history to a JSON file
        export_path = arg.strip() if arg.strip() else f"clyde_export_{agent.session_id}.json"
        messages = agent.history.get_messages()
        export_data = {
            "session_id": agent.session_id,
            "turn_count": agent.turn_count,
            "model": agent.config.provider.model,
            "provider": agent.config.provider.provider.value,
            "messages": [],
        }
        for msg in messages:
            entry: dict = {"role": msg.role.value, "content": []}
            for block in msg.content:
                if isinstance(block, TextBlock):
                    entry["content"].append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    entry["content"].append({
                        "type": "tool_use", "name": block.name,
                        "id": block.id, "input": block.input,
                    })
                elif isinstance(block, ToolResultBlock):
                    entry["content"].append({
                        "type": "tool_result", "tool_use_id": block.tool_use_id,
                        "content": block.content,
                    })
            export_data["messages"].append(entry)
        try:
            Path(export_path).write_text(json.dumps(export_data, indent=2), encoding="utf-8")
            renderer.print_info(f"Exported {len(messages)} messages to: {export_path}")
        except OSError as e:
            renderer.print_error(f"Export failed: {e}")
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

    # Create plugins directory
    plugins_dir = Path(".clyde_plugins")
    if not plugins_dir.exists():
        plugins_dir.mkdir(parents=True, exist_ok=True)
        (plugins_dir / "README.md").write_text(
            "# Clyde Plugins\n\n"
            "Drop Python files here that define classes inheriting from `BaseTool`.\n"
            "They'll be auto-discovered and loaded at startup.\n",
            encoding="utf-8",
        )
        print(f"Created {plugins_dir}/")


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

"""System prompt assembly — Step 4 of the agent loop.

Merges project instructions (CLYDE.md), tool definitions,
workspace context, and persistent memory into the system prompt.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from .models import ToolDefinition


def build_system_prompt(
    *,
    identity_file: str = "CLYDE.md",
    tool_definitions: list[ToolDefinition] | None = None,
    memory_entries: list[str] | None = None,
    cwd: Path | None = None,
    extra_context: str = "",
) -> str:
    """Assemble the full system prompt from all sources.

    This is the Step 4 equivalent — everything the model needs
    to know before generating a response.
    """
    cwd = cwd or Path.cwd()
    sections: list[str] = []

    # --- Identity / project instructions ---
    identity_text = _load_identity(identity_file, cwd)
    if identity_text:
        sections.append(identity_text)

    # --- Environment context ---
    sections.append(_build_environment_context(cwd))

    # --- Tool definitions ---
    if tool_definitions:
        sections.append(_build_tool_section(tool_definitions))

    # --- Memory ---
    if memory_entries:
        sections.append(_build_memory_section(memory_entries))

    # --- Extra context ---
    if extra_context:
        sections.append(f"# Additional Context\n\n{extra_context}")

    return "\n\n---\n\n".join(sections)


def _load_identity(filename: str, cwd: Path) -> str:
    """Load CLYDE.md or similar identity file."""
    # Check cwd first, then walk up to find it
    for directory in [cwd, *cwd.parents]:
        candidate = directory / filename
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                break
        # Don't walk above home or root
        if directory == Path.home() or directory == directory.parent:
            break
    return ""


def _build_environment_context(cwd: Path) -> str:
    """Build workspace and environment context."""
    # Git info
    git_branch = ""
    git_root = cwd
    if (cwd / ".git").exists() or _find_git_root(cwd):
        git_root = _find_git_root(cwd) or cwd
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=str(git_root), timeout=5,
            )
            git_branch = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    lines = [
        "# Environment",
        "",
        f"- Working directory: {cwd}",
        f"- Platform: {platform.system()} {platform.release()}",
        f"- Python: {platform.python_version()}",
        f"- Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
    ]
    if git_branch:
        lines.append(f"- Git branch: {git_branch}")
        lines.append(f"- Git root: {git_root}")

    return "\n".join(lines)


def _find_git_root(path: Path) -> Path | None:
    """Walk up to find .git directory."""
    for directory in [path, *path.parents]:
        if (directory / ".git").exists():
            return directory
        if directory == directory.parent:
            break
    return None


def _build_tool_section(tools: list[ToolDefinition]) -> str:
    """Build the tool definitions section."""
    lines = [
        "# Available Tools",
        "",
        f"You have {len(tools)} tools available. Use them to accomplish tasks.",
        "",
    ]
    by_category: dict[str, list[ToolDefinition]] = {}
    for t in tools:
        by_category.setdefault(t.category, []).append(t)

    for category, tool_list in sorted(by_category.items()):
        lines.append(f"## {category.title()}")
        for t in tool_list:
            params = ", ".join(
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in t.parameters
            )
            lines.append(f"- **{t.name}**({params}): {t.description}")
        lines.append("")

    return "\n".join(lines)


def _build_memory_section(entries: list[str]) -> str:
    """Build the memory section from persistent memory entries."""
    lines = [
        "# Memory",
        "",
        "Relevant information from previous sessions:",
        "",
    ]
    for entry in entries:
        lines.append(f"- {entry}")
    return "\n".join(lines)

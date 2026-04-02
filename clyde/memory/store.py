"""Persistent memory store — survives across sessions.

Stores key-value memories as individual markdown files with
YAML frontmatter, indexed by a MEMORY.md file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryEntry:
    name: str
    description: str
    memory_type: str  # user, feedback, project, reference
    content: str
    filename: str


class MemoryStore:
    """File-based persistent memory system."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save(self, entry: MemoryEntry) -> Path:
        """Save a memory entry to disk."""
        path = self.memory_dir / entry.filename
        text = (
            f"---\n"
            f"name: {entry.name}\n"
            f"description: {entry.description}\n"
            f"type: {entry.memory_type}\n"
            f"---\n\n"
            f"{entry.content}\n"
        )
        path.write_text(text, encoding="utf-8")
        self._update_index()
        return path

    def load(self, filename: str) -> MemoryEntry | None:
        """Load a single memory entry."""
        path = self.memory_dir / filename
        if not path.exists():
            return None
        return self._parse_file(path)

    def load_all(self) -> list[MemoryEntry]:
        """Load all memory entries."""
        entries = []
        for path in self.memory_dir.glob("*.md"):
            if path.name == "MEMORY.md":
                continue
            entry = self._parse_file(path)
            if entry:
                entries.append(entry)
        return entries

    def get_context_entries(self) -> list[str]:
        """Get memory entries formatted for the system prompt."""
        entries = self.load_all()
        return [f"[{e.memory_type}] {e.name}: {e.content[:300]}" for e in entries]

    def delete(self, filename: str) -> bool:
        """Delete a memory entry."""
        path = self.memory_dir / filename
        if path.exists():
            path.unlink()
            self._update_index()
            return True
        return False

    def search(self, query: str) -> list[MemoryEntry]:
        """Search memories by keyword."""
        query_lower = query.lower()
        results = []
        for entry in self.load_all():
            if (
                query_lower in entry.name.lower()
                or query_lower in entry.content.lower()
                or query_lower in entry.description.lower()
            ):
                results.append(entry)
        return results

    def _parse_file(self, path: Path) -> MemoryEntry | None:
        """Parse a memory file with YAML frontmatter."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        # Parse frontmatter
        match = re.match(r"^---\n(.+?)\n---\n\n?(.*)", text, re.DOTALL)
        if not match:
            return None

        frontmatter, content = match.groups()
        meta: dict[str, str] = {}
        for line in frontmatter.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        return MemoryEntry(
            name=meta.get("name", path.stem),
            description=meta.get("description", ""),
            memory_type=meta.get("type", "general"),
            content=content.strip(),
            filename=path.name,
        )

    def _update_index(self) -> None:
        """Rebuild the MEMORY.md index file."""
        entries = self.load_all()
        lines = ["# Clyde Memory Index\n"]
        for entry in sorted(entries, key=lambda e: e.memory_type):
            lines.append(f"- [{entry.name}]({entry.filename}) — {entry.description[:100]}")

        index_path = self.memory_dir / "MEMORY.md"
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

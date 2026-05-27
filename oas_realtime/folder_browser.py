from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    path: Path


def list_child_directories(path: Path) -> list[DirectoryEntry]:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return []

    entries: list[DirectoryEntry] = []
    for child in path.iterdir():
        try:
            if child.is_dir():
                entries.append(DirectoryEntry(name=child.name, path=child))
        except OSError:
            continue
    return sorted(entries, key=lambda entry: entry.name.lower())


def parent_directory(path: Path) -> Path:
    path = Path(path)
    parent = path.parent
    return path if parent == path else parent

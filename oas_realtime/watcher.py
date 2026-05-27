from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time


TIME_TOKEN = re.compile(r"__\s*(\d+(?:\.\d+)?)\s*__")


@dataclass(frozen=True)
class AbsorbanceFile:
    path: Path
    time_point: float


@dataclass(frozen=True)
class FileSignature:
    size: int
    modified_at: float
    observed_at: float


def extract_time_token(name: str) -> float | None:
    match = TIME_TOKEN.search(name)
    if not match:
        return None
    return float(match.group(1))


def is_absorbance_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".txt"
        and "absorbance" in path.name.lower()
        and extract_time_token(path.name) is not None
    )


def scan_absorbance_files(folder: Path) -> list[AbsorbanceFile]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []

    files: list[AbsorbanceFile] = []
    for path in folder.iterdir():
        if not is_absorbance_file(path):
            continue
        time_point = extract_time_token(path.name)
        if time_point is not None:
            files.append(AbsorbanceFile(path=path, time_point=time_point))
    return sorted(files, key=lambda item: (item.time_point, item.path.name))


def get_file_signature(path: Path, now: float | None = None) -> FileSignature:
    stat = Path(path).stat()
    return FileSignature(
        size=stat.st_size,
        modified_at=stat.st_mtime,
        observed_at=time.time() if now is None else now,
    )


def is_file_stable(
    path: Path,
    previous: FileSignature | None,
    stable_seconds: float,
    now: float | None = None,
) -> tuple[bool, FileSignature]:
    current = get_file_signature(path, now=now)
    if previous is None:
        return False, current

    unchanged = current.size == previous.size and current.modified_at == previous.modified_at
    old_enough = current.observed_at - previous.observed_at >= stable_seconds
    return unchanged and old_enough, current

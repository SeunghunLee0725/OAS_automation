from __future__ import annotations

from pathlib import Path
from typing import Callable


def choose_directory(
    initial_dir: Path,
    title: str,
    root_factory: Callable[[], object] | None = None,
    askdirectory: Callable[..., str] | None = None,
) -> Path | None:
    if root_factory is None or askdirectory is None:
        import tkinter as tk
        from tkinter import filedialog

        root_factory = tk.Tk
        askdirectory = filedialog.askdirectory

    root = root_factory()
    try:
        if hasattr(root, "withdraw"):
            root.withdraw()
        if hasattr(root, "attributes"):
            root.attributes("-topmost", True)
        selected = askdirectory(
            parent=root,
            title=title,
            initialdir=str(initial_dir),
            mustexist=True,
        )
    finally:
        if hasattr(root, "destroy"):
            root.destroy()

    return Path(selected) if selected else None

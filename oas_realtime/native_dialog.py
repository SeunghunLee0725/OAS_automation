from __future__ import annotations

import base64
from pathlib import Path
import subprocess
import sys
from typing import Callable


def choose_directory_powershell(
    initial_dir: Path,
    title: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path | None:
    initial = str(Path(initial_dir))
    script = f"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = @'
{title}
'@
$dialog.SelectedPath = @'
{initial}
'@
$dialog.ShowNewFolderButton = $false
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.SelectedPath
}}
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    completed = runner(
        ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "PowerShell folder picker failed")
    selected = completed.stdout.strip()
    return Path(selected) if selected else None


def choose_directory(
    initial_dir: Path,
    title: str,
    root_factory: Callable[[], object] | None = None,
    askdirectory: Callable[..., str] | None = None,
) -> Path | None:
    if root_factory is None or askdirectory is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ModuleNotFoundError:
            if sys.platform.startswith("win"):
                return choose_directory_powershell(initial_dir=initial_dir, title=title)
            raise

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

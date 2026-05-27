from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RESULTS_FILENAME = "oas_realtime_results.csv"
FAILED_FILENAME = "oas_realtime_failed_files.csv"


@dataclass
class RealtimeSession:
    folder: Path
    results_df: pd.DataFrame = field(init=False)
    failed_df: pd.DataFrame = field(init=False)
    processed_filenames: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.results_path = self.folder / RESULTS_FILENAME
        self.failed_path = self.folder / FAILED_FILENAME
        self.load()

    def load(self) -> None:
        self.results_df = self._read_csv(self.results_path)
        self.failed_df = self._read_csv(self.failed_path)
        if "filename" in self.results_df.columns:
            self.processed_filenames = set(self.results_df["filename"].dropna().astype(str))
        else:
            self.processed_filenames = set()

    def append_result(self, row: dict[str, Any]) -> None:
        row_df = pd.DataFrame([row])
        self.results_df = pd.concat([self.results_df, row_df], ignore_index=True)
        self.processed_filenames.add(str(row.get("filename", "")))
        self._write_csv(self.results_df, self.results_path)

    def append_failure(self, filename: str, error: str) -> None:
        row = {
            "filename": filename,
            "error": error,
            "failed_at": datetime.now().isoformat(timespec="seconds"),
        }
        row_df = pd.DataFrame([row])
        self.failed_df = pd.concat([self.failed_df, row_df], ignore_index=True)
        self._write_csv(self.failed_df, self.failed_path)

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    def _write_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

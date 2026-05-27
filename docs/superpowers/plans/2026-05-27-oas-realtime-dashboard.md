# OAS Realtime Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit dashboard that watches a selected OAS experiment folder and shows fitted active species concentration time series in real time.

**Architecture:** Keep `nnls_v8_17.py` as the scientific fitting engine. Add a small `oas_realtime` package for filename discovery, file-stability checks, non-interactive single-file analysis, CSV persistence, and Streamlit UI. Deploy the package plus `cross_section/` to the remote Windows PC.

**Tech Stack:** Python, NumPy, pandas, SciPy, Matplotlib, Plotly, Streamlit, pytest.

---

## File Structure

- Create `oas_realtime/__init__.py`
  - Package marker and version.
- Create `oas_realtime/watcher.py`
  - Absorbance filename matching, time extraction, scanning, file-stability checks.
- Create `oas_realtime/analysis.py`
  - Non-interactive wrapper around `BatchPlasmaAnalyzer`.
  - Returns serializable result rows and latest spectrum diagnostic data.
- Create `oas_realtime/session.py`
  - Loads/saves `oas_realtime_results.csv` and `oas_realtime_failed_files.csv`.
  - Tracks processed and failed files.
- Create `oas_realtime/app.py`
  - Streamlit dashboard focused on concentration time series.
- Create `requirements.txt`
  - Runtime dependencies.
- Create `run_oas_dashboard.bat`
  - Windows launcher.
- Create tests under `tests/`
  - Unit tests for watcher, session persistence, and analysis row mapping.

## Task 1: Filename Discovery And Stability

**Files:**
- Create: `oas_realtime/__init__.py`
- Create: `oas_realtime/watcher.py`
- Test: `tests/test_watcher.py`

- [ ] **Step 1: Write failing watcher tests**

Test these behaviors:

- `extract_time_token("1_Absorbance__15__10-41-41-606.txt") == 15.0`
- non-absorbance `.txt` files are ignored.
- `scan_absorbance_files(folder)` returns matching files sorted by time.
- `is_file_stable(path, stable_seconds=1.0)` returns true only when size and mtime are unchanged for the required duration.

- [ ] **Step 2: Run watcher tests and verify they fail**

Run:

```bash
pytest tests/test_watcher.py -v
```

- [ ] **Step 3: Implement watcher**

Implement:

- `TIME_TOKEN = re.compile(r"__\s*(\d+(?:\.\d+)?)\s*__")`
- `extract_time_token(name: str) -> float | None`
- `is_absorbance_file(path: Path) -> bool`
- `scan_absorbance_files(folder: Path) -> list[AbsorbanceFile]`
- `get_file_signature(path: Path) -> FileSignature`
- `is_file_stable(path: Path, previous: FileSignature | None, stable_seconds: float) -> tuple[bool, FileSignature]`

- [ ] **Step 4: Run watcher tests and verify they pass**

Run:

```bash
pytest tests/test_watcher.py -v
```

## Task 2: Realtime CSV Session Persistence

**Files:**
- Create: `oas_realtime/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write failing session tests**

Test these behaviors:

- New session starts with empty results and processed set.
- Appending a result writes `oas_realtime_results.csv`.
- Reloading the same folder restores processed filenames.
- Failed files are written to `oas_realtime_failed_files.csv`.

- [ ] **Step 2: Run session tests and verify they fail**

Run:

```bash
pytest tests/test_session.py -v
```

- [ ] **Step 3: Implement session persistence**

Implement `RealtimeSession` with:

- `folder`
- `results_path`
- `failed_path`
- `results_df`
- `failed_df`
- `processed_filenames`
- `append_result(row: dict)`
- `append_failure(filename: str, error: str)`
- `load()`

- [ ] **Step 4: Run session tests and verify they pass**

Run:

```bash
pytest tests/test_session.py -v
```

## Task 3: Non-Interactive Analysis Wrapper

**Files:**
- Create: `oas_realtime/analysis.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Write failing analysis tests with a fake analyzer**

Avoid slow NNLS in unit tests by injecting a fake analyzer object. Test:

- concentration arrays map to species columns.
- result rows include filename, time, r2, RH, temperature, saturation count, and analyzed timestamp.
- latest spectrum data can be converted into a Plotly-friendly dataframe.

- [ ] **Step 2: Run analysis tests and verify they fail**

Run:

```bash
pytest tests/test_analysis.py -v
```

- [ ] **Step 3: Implement analysis wrapper**

Implement:

- `AnalysisSettings` dataclass.
- `build_analyzer(settings: AnalysisSettings) -> BatchPlasmaAnalyzer`
- `analyze_file(analyzer, file_path: Path, time_point: float, settings: AnalysisSettings) -> AnalysisOutput`
- `result_to_row(output: AnalysisOutput, settings: AnalysisSettings) -> dict`
- `spectrum_dataframe(output: AnalysisOutput) -> pd.DataFrame`

The wrapper must call `analyzer.load_cross_sections()` once before live monitoring begins.

- [ ] **Step 4: Run analysis tests and verify they pass**

Run:

```bash
pytest tests/test_analysis.py -v
```

## Task 4: Streamlit Dashboard

**Files:**
- Create: `oas_realtime/app.py`
- Create: `requirements.txt`

- [ ] **Step 1: Build the dashboard UI**

Implement:

- folder path input
- recent folder shortcut storage in Streamlit session state
- start/stop controls
- RH, temperature, saturation threshold, scan interval, file stability seconds controls
- species multiselect
- linear/log y-axis toggle
- main concentration time-series Plotly chart
- latest metrics
- collapsible latest measured-vs-fitted diagnostic chart
- failed file table

- [ ] **Step 2: Implement bounded polling**

Each auto-refresh should:

- scan matching files
- skip processed files
- check stability
- analyze at most a small configured batch, default `1`
- update CSV and dashboard state
- return control to Streamlit

Do not implement an infinite blocking loop.

- [ ] **Step 3: Add requirements**

Include:

```text
numpy
pandas
scipy
matplotlib
plotly
streamlit
pytest
```

## Task 5: Launcher And Remote Deployment

**Files:**
- Create: `run_oas_dashboard.bat`

- [ ] **Step 1: Create Windows launcher**

The launcher should:

- `cd /d` into the deployed app directory.
- run `python -m streamlit run oas_realtime\app.py`.
- pause on failure so the user can read errors.

- [ ] **Step 2: Copy files to remote Windows PC**

Use SSH/SCP to create a deployment folder, for example:

`C:\Users\user\Desktop\OAS_Realtime_Dashboard`

Copy:

- `oas_realtime/`
- `nnls_v8_17.py`
- `cross_section/`
- `requirements.txt`
- `run_oas_dashboard.bat`

- [ ] **Step 3: Install dependencies remotely**

Run:

```powershell
py -m pip install -r C:\Users\user\Desktop\OAS_Realtime_Dashboard\requirements.txt
```

Fallback to `python -m pip` if `py` is unavailable.

## Task 6: Verification

**Files:**
- Existing and created files.

- [ ] **Step 1: Run all local tests**

Run:

```bash
pytest -v
```

- [ ] **Step 2: Syntax-check application files**

Run:

```bash
python -m compileall oas_realtime nnls_v8_17.py
```

- [ ] **Step 3: Smoke-test Streamlit import**

Run:

```bash
python - <<'PY'
from oas_realtime.analysis import AnalysisSettings
from oas_realtime.watcher import extract_time_token
print(AnalysisSettings())
print(extract_time_token("1_Absorbance__0__10-30-37-184.txt"))
PY
```

- [ ] **Step 4: Remote smoke test**

Run a short remote command to verify files exist and dependencies import:

```powershell
cd C:\Users\user\Desktop\OAS_Realtime_Dashboard
python -c "import streamlit, plotly, scipy, pandas; print('ok')"
```

- [ ] **Step 5: Report exact run command**

Tell the user to run on the remote Windows PC:

```bat
C:\Users\user\Desktop\OAS_Realtime_Dashboard\run_oas_dashboard.bat
```

Then open the local Streamlit URL shown in the terminal.

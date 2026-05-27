# OAS Realtime Dashboard Design

Date: 2026-05-27

## Goal

Build a realtime in-situ optical absorption spectroscopy post-processing program for the remote Windows PC at `user@100.122.157.24`.

The user selects one experiment folder under `C:\Users\user\Desktop\최주연` or another local Windows path. The program watches for newly generated absorbance `.txt` files, runs NNLS fitting as each file becomes complete, and shows the fitted active species concentration time series during the experiment.

The primary output is the time-series concentration trend for:

- `O3`
- `NO`
- `NO2`
- `NO3`
- `N2O4`
- `N2O5`
- `HONO`
- `HONO2`

## Existing Context

The current workspace contains:

- `nnls_v8_17.py`: latest NNLS UV-Vis absorbance analysis script.
- `cross_section/`: local cross-section folder with the required files:
  - `O3_ordered_cross_section.txt`
  - `NO_ordered_cross_section.txt`
  - `NO2_ordered_cross_section.txt`
  - `NO3_ordered_cross_section.txt`
  - `N2O4_ordered_cross_section.txt`
  - `N2O5_ordered_cross_section.txt`
  - `HONO_ordered_cross_section.txt`
  - `HONO2_ordered_cross_section.txt`

Remote observations:

- The remote host is a Windows SSH server.
- Target data root exists at `C:\Users\user\Desktop\최주연`.
- Example data folder: `C:\Users\user\Desktop\최주연\20251226`.
- Example absorbance filename: `1_Absorbance__0__10-30-37-184.txt`.
- Data format contains a text header followed by `>>>>>Begin Spectral Data<<<<<` and two-column wavelength/absorbance rows.
- The remote target root contains many `.txt` files, so the program must filter by absorbance filename patterns rather than analyze every text file.
- The remote PC did not show a usable `cross_section` folder under `Desktop\최주연`; deployment should copy the local `cross_section` folder with the program.

## Recommended Approach

Use a Streamlit dashboard running locally on the remote Windows PC.

Reasons:

- It is Python-based and can reuse the existing NNLS analysis code.
- It provides a browser UI with minimal frontend code.
- It supports live Plotly charts, controls, logs, and table output.
- It is simpler to deploy than a separate FastAPI/frontend application.

## Architecture

Create a small application around the existing NNLS core:

- `analysis_core`
  - Wraps or imports `BatchPlasmaAnalyzer` from `nnls_v8_17.py`.
  - Provides a non-interactive API for analyzing one absorbance file.
  - Loads cross-section files from a configurable local `cross_section` path.

- `file_watcher`
  - Scans the selected folder for files matching the absorbance pattern.
  - Recognizes filenames containing `Absorbance` and a `__<integer>__` time token.
  - Sorts files by extracted time.
  - Detects newly created matching files while the dashboard is running.

- `stability_check`
  - Prevents reading a file while the spectrometer software is still writing it.
  - Treats a file as ready only after its size and modified time remain stable for a configurable wait period.

- `realtime_session`
  - Tracks processed files and failed files.
  - Holds the in-memory result table for the current dashboard run.
  - Persists each completed result to CSV immediately.
  - Restores previous results from CSV on dashboard restart.

- `dashboard`
  - Streamlit UI for entering a folder path, starting/stopping monitoring, showing status, and displaying live concentration time series.
  - Uses path text input plus recent-folder shortcuts. A native Windows folder picker is out of scope for the first Streamlit version because the dashboard runs in a browser.

- `deployment`
  - Copies the dashboard code and `cross_section` folder to the remote Windows PC.
  - Provides a Windows launch script for starting Streamlit.

The first version monitors one experiment folder at a time. Multi-folder monitoring is out of scope, but the internal modules should not prevent later expansion.

## Data Flow

1. The user starts the program on the remote Windows PC.
2. Streamlit opens in the browser.
3. The user enters or selects an experiment folder path.
4. The program validates:
   - Folder exists.
   - Cross-section files exist.
   - Existing result CSV can be read or created.
5. When monitoring starts:
   - Existing matching absorbance files are queued in time order.
   - Already processed files from prior CSV/log state are skipped.
   - New matching files are detected on each scan interval.
   - Each Streamlit refresh processes at most a small bounded batch of ready files, then yields back to the UI.
6. For each ready file:
   - Wait for file stability.
   - Run NNLS fitting.
   - Extract concentrations, time point, R-squared, and diagnostic fields.
   - Append the result to memory.
   - Write the result row to `oas_realtime_results.csv`.
   - Refresh the dashboard charts.
7. If a file fails:
   - Record filename, error message, and timestamp.
   - Continue processing later files.

## UI Design

The dashboard focuses on active species concentration time series.

Primary view:

- Large Plotly concentration time-series chart.
- Species visibility controls for `O3`, `NO`, `NO2`, `NO3`, `N2O4`, `N2O5`, `HONO`, `HONO2`.
- Linear/log y-axis toggle.
- Latest time point and latest concentration values.

Controls:

- Experiment folder path input.
- Recent folder shortcuts stored locally by the dashboard.
- `Start monitoring` button.
- `Stop monitoring` button.
- `Refresh once` button for manual scan.
- RH relative humidity setting.
- Fixed temperature setting.
- Saturation threshold setting.
- Scan interval setting.
- File stability wait setting.
- Optional per-file PNG save toggle.

Status and diagnostics:

- Current state: idle, scanning, waiting for stable file, analyzing, stopped, error.
- Number of processed files.
- Number of queued files.
- Number of failed files.
- Latest filename.
- Latest fitting R-squared.
- Latest processing time.
- Collapsible latest measured-vs-fitted spectrum plot for diagnostics.
- Failed file log.

The latest measured-vs-fitted spectrum is secondary. The concentration time-series chart must remain the main first-screen output.

## Analysis Settings

Use the current `nnls_v8_17.py` defaults unless exposed in the UI:

- `path_length = 5.0`
- `correct_overfit = True`
- `correct_n2o4_equilibrium = True`
- `noise_threshold = 1e-4`
- `no3_peak_constraint = True`
- `no3_peak_range = (615, 635)`
- `no3_damping_factor = 0.95`
- `no3_min_threshold = 0.01`
- `no3_max_iterations = 150`
- `no3_window_nm = 3.0`
- `uv_start = 200.0`
- `n2o5_scale = 1.0`
- `apply_n2o5_stoich_cap = False`
- `no2_vis_constraint = True`
- `no2_vis_range = (350, 450)`
- `no2_vis_window_nm = 10.0`
- `hono_no2_factor = 0.01`

First version supports fixed temperature only. Temperature CSV profiles are excluded from the realtime UI because changing time-dependent temperature inputs during live monitoring can make interpretation harder. This can be added later after the core realtime flow is stable.

## Output Files

Write outputs inside the monitored experiment folder:

- `oas_realtime_results.csv`
  - One row per successfully analyzed file.
  - Includes filename, time, species concentrations, R-squared, temperature, RH, saturation exclusion count, and processing timestamp.

- `oas_realtime_failed_files.csv`
  - One row per failed file.
  - Includes filename, error message, and failure timestamp.

- Optional per-file PNG folder:
  - Enabled only if the user turns on diagnostic PNG saving.
  - Disabled by default for realtime runs to reduce latency and disk output.

## Reliability Rules

- Monitoring must not run as a blocking infinite loop inside one Streamlit request. The dashboard should poll on timed refreshes or a small background-safe state machine so the page stays responsive.
- Only files matching the absorbance filename rule are processed.
- Files are analyzed once unless the user explicitly clears previous results.
- A file must be stable before analysis:
  - Size unchanged.
  - Modified time unchanged.
  - Stability duration reached.
- Failures do not stop monitoring.
- Results are flushed to CSV after every file.
- Restarting the dashboard restores existing concentration time-series data from CSV.
- Missing cross-section files are treated as startup-blocking errors.
- Invalid folder paths are shown in the dashboard and do not start monitoring.

## Deployment

Deploy to the remote Windows PC over SSH/SCP:

- Copy application files.
- Copy `cross_section/`.
- Install required Python packages if needed:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `plotly`
  - `streamlit`
- Provide a Windows launcher script, for example `run_oas_dashboard.bat`, that starts:

```bat
streamlit run app.py
```

The program should run on the remote PC and open locally accessible browser UI on that PC.

## Testing Plan

Local functional tests:

- Verify cross-section folder validation.
- Verify filename time extraction from examples like `1_Absorbance__0__10-30-37-184.txt`.
- Verify non-matching `.txt` files are ignored.
- Verify one sample absorbance file can be analyzed non-interactively.

Realtime behavior tests:

- Start dashboard against a test folder with a few existing absorbance files.
- Add files one at a time and verify the chart updates.
- Simulate a partially written file by copying it slowly or changing its size; verify analysis waits until stable.
- Restart dashboard and verify `oas_realtime_results.csv` restores prior series.
- Verify failed files are logged and do not stop monitoring.

Remote Windows tests:

- Copy the application and `cross_section` to the remote PC.
- Install dependencies.
- Start Streamlit through the launcher.
- Monitor a folder under `C:\Users\user\Desktop\최주연`.
- Confirm that the main visible output is the active species concentration time-series chart.

## Out Of Scope For First Version

- Monitoring multiple experiment folders at the same time.
- Multiple remote users.
- Full web app with account/login.
- Temperature CSV profile support.
- Editing or deleting past result rows from the UI.
- Re-fitting all historical files automatically when analysis parameters change.

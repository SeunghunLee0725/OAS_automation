from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from oas_realtime.analysis import (
    DEFAULT_SPECIES,
    AnalysisOutput,
    AnalysisSettings,
    analyze_file,
    build_analyzer,
    result_to_row,
    spectrum_dataframe,
)
from oas_realtime.session import RealtimeSession
from oas_realtime.watcher import FileSignature, is_file_stable, scan_absorbance_files


APP_TITLE = "OAS Realtime Active Species Dashboard"
DEFAULT_DEPLOY_ROOT = APP_ROOT
DEFAULT_CROSS_SECTION = DEFAULT_DEPLOY_ROOT / "cross_section"


@st.cache_resource(show_spinner=False)
def cached_analyzer(settings: AnalysisSettings):
    return build_analyzer(settings)


def initialize_state() -> None:
    st.session_state.setdefault("monitoring", False)
    st.session_state.setdefault("latest_output", None)
    st.session_state.setdefault("file_signatures", {})
    st.session_state.setdefault("recent_folders", [])
    st.session_state.setdefault("status", "idle")


def remember_folder(folder: str) -> None:
    if not folder:
        return
    recent = [item for item in st.session_state.recent_folders if item != folder]
    st.session_state.recent_folders = [folder, *recent][:5]


def validate_cross_sections(cross_section_path: Path) -> list[str]:
    missing = []
    for species in DEFAULT_SPECIES:
        path = cross_section_path / f"{species}_ordered_cross_section.txt"
        if not path.exists():
            missing.append(path.name)
    return missing


def concentration_long_df(results_df: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=["time", "species", "concentration"])
    available = [sp for sp in species if sp in results_df.columns]
    if not available:
        return pd.DataFrame(columns=["time", "species", "concentration"])
    return results_df.melt(
        id_vars=["time"],
        value_vars=available,
        var_name="species",
        value_name="concentration",
    ).sort_values(["time", "species"])


def process_ready_files(
    folder: Path,
    session: RealtimeSession,
    analyzer,
    settings: AnalysisSettings,
    stable_seconds: float,
    max_files_per_refresh: int,
) -> tuple[int, AnalysisOutput | None]:
    processed = 0
    latest_output = None
    signatures: dict[str, FileSignature] = st.session_state.file_signatures

    for item in scan_absorbance_files(folder):
        if item.path.name in session.processed_filenames:
            continue
        if processed >= max_files_per_refresh:
            break

        previous = signatures.get(str(item.path))
        stable, current = is_file_stable(item.path, previous, stable_seconds)
        signatures[str(item.path)] = current
        if not stable:
            st.session_state.status = f"waiting for stable file: {item.path.name}"
            continue

        try:
            st.session_state.status = f"analyzing: {item.path.name}"
            output = analyze_file(analyzer, item.path, item.time_point, settings)
            session.append_result(result_to_row(output, settings))
            latest_output = output
            processed += 1
        except Exception as exc:  # keep monitoring after a bad file
            session.append_failure(item.path.name, f"{exc}\n{traceback.format_exc(limit=3)}")
            session.processed_filenames.add(item.path.name)
            processed += 1

    st.session_state.file_signatures = signatures
    if processed:
        st.session_state.status = f"processed {processed} file(s)"
    return processed, latest_output


def render_controls() -> tuple[Path, Path, AnalysisSettings, float, float, int, list[str], bool]:
    default_folder = r"C:\Users\user\Desktop\최주연"
    folder_text = st.text_input("Experiment folder", value=st.session_state.get("folder_text", default_folder))
    st.session_state.folder_text = folder_text

    if st.session_state.recent_folders:
        selected_recent = st.selectbox("Recent folders", [""] + st.session_state.recent_folders)
        if selected_recent and st.button("Use recent folder"):
            st.session_state.folder_text = selected_recent
            st.rerun()

    cross_section_text = st.text_input("Cross-section folder", value=str(DEFAULT_CROSS_SECTION))
    cross_section_path = Path(cross_section_text)

    col1, col2, col3 = st.columns(3)
    with col1:
        rh_percent = st.number_input("RH (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        scan_interval = st.number_input("Scan interval (s)", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
    with col2:
        temperature_k = st.number_input("Temperature (K)", min_value=200.0, max_value=400.0, value=295.0, step=1.0)
        stable_seconds = st.number_input("File stable wait (s)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    with col3:
        saturation_threshold = st.number_input("Saturation threshold", min_value=0.1, max_value=10.0, value=2.5, step=0.05)
        max_files = st.number_input("Max files per refresh", min_value=1, max_value=20, value=1, step=1)

    species = st.multiselect("Species", DEFAULT_SPECIES, default=DEFAULT_SPECIES)
    log_y = st.toggle("Log y-axis", value=False)

    settings = AnalysisSettings(
        cross_section_path=cross_section_path,
        rh_percent=rh_percent,
        temperature_k=temperature_k,
        saturation_threshold=saturation_threshold,
    )
    return Path(folder_text), cross_section_path, settings, scan_interval, stable_seconds, int(max_files), species, log_y


def render_chart(results_df: pd.DataFrame, species: list[str], log_y: bool) -> None:
    long_df = concentration_long_df(results_df, species)
    if long_df.empty:
        st.info("No analyzed concentration data yet.")
        return

    fig = px.line(
        long_df,
        x="time",
        y="concentration",
        color="species",
        markers=True,
        log_y=log_y,
        labels={"time": "Time (s)", "concentration": "Concentration (molecules/cm^3)"},
    )
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_latest_spectrum(output: AnalysisOutput | None) -> None:
    if output is None:
        return
    with st.expander("Latest measured vs fitted spectrum", expanded=False):
        df = spectrum_dataframe(output)
        if df.empty:
            st.info("No spectrum data available for the latest file.")
            return
        fig = px.line(df, x="wavelength", y=["measured", "fitted"])
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    initialize_state()
    st.title(APP_TITLE)

    with st.sidebar:
        folder, cross_section_path, settings, scan_interval, stable_seconds, max_files, species, log_y = render_controls()
        start = st.button("Start monitoring", type="primary", use_container_width=True)
        stop = st.button("Stop monitoring", use_container_width=True)
        refresh_once = st.button("Refresh once", use_container_width=True)

    if start:
        st.session_state.monitoring = True
        remember_folder(str(folder))
    if stop:
        st.session_state.monitoring = False
        st.session_state.status = "stopped"

    st.caption(f"Status: {st.session_state.status}")

    if not folder.exists() or not folder.is_dir():
        st.error(f"Experiment folder does not exist: {folder}")
        return

    missing = validate_cross_sections(cross_section_path)
    if missing:
        st.error("Missing cross-section files: " + ", ".join(missing))
        return

    session = RealtimeSession(folder)

    if st.session_state.monitoring or refresh_once:
        analyzer = cached_analyzer(settings)
        _, latest_output = process_ready_files(
            folder=folder,
            session=session,
            analyzer=analyzer,
            settings=settings,
            stable_seconds=stable_seconds,
            max_files_per_refresh=max_files,
        )
        if latest_output is not None:
            st.session_state.latest_output = latest_output
        session.load()

    metrics = st.columns(4)
    metrics[0].metric("Processed files", len(session.processed_filenames))
    metrics[1].metric("Failed files", len(session.failed_df))
    if not session.results_df.empty and "time" in session.results_df.columns:
        metrics[2].metric("Latest time (s)", f"{session.results_df['time'].max():.0f}")
    if not session.results_df.empty and "r2" in session.results_df.columns:
        latest_r2 = session.results_df["r2"].dropna().iloc[-1] if not session.results_df["r2"].dropna().empty else None
        metrics[3].metric("Latest R²", f"{latest_r2:.4f}" if latest_r2 is not None else "-")

    render_chart(session.results_df, species, log_y)

    if not session.results_df.empty:
        with st.expander("Latest concentration table", expanded=True):
            display_cols = ["time", "filename", *[sp for sp in species if sp in session.results_df.columns], "r2"]
            st.dataframe(session.results_df[display_cols].tail(20), use_container_width=True)

    render_latest_spectrum(st.session_state.latest_output)

    if not session.failed_df.empty:
        with st.expander("Failed files", expanded=False):
            st.dataframe(session.failed_df.tail(50), use_container_width=True)

    if st.session_state.monitoring:
        time.sleep(scan_interval)
        st.rerun()


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nnls_v8_17 import BatchPlasmaAnalyzer


DEFAULT_SPECIES = ["O3", "NO", "NO2", "NO3", "N2O4", "N2O5", "HONO", "HONO2"]


@dataclass(frozen=True)
class AnalysisSettings:
    cross_section_path: Path = Path("cross_section")
    path_length: float = 5.0
    rh_percent: float = 0.0
    temperature_k: float = 295.0
    saturation_threshold: float = 2.5
    correct_overfit: bool = True
    correct_n2o4_equilibrium: bool = True
    noise_threshold: float = 1e-4
    no3_peak_constraint: bool = True
    no3_peak_range: tuple[float, float] = (615, 635)
    no3_damping_factor: float = 0.95
    no3_min_threshold: float = 0.01
    no3_max_iterations: int = 150
    no3_window_nm: float = 3.0
    uv_start: float = 200.0
    n2o5_scale: float = 1.0
    apply_n2o5_stoich_cap: bool = False
    no2_vis_constraint: bool = True
    no2_vis_range: tuple[float, float] = (350, 450)
    no2_vis_window_nm: float = 10.0
    hono_no2_factor: float = 0.01


@dataclass(frozen=True)
class AnalysisOutput:
    file_path: Path
    time_point: float
    result: dict[str, Any]
    species: list[str]


def build_analyzer(settings: AnalysisSettings) -> BatchPlasmaAnalyzer:
    analyzer = BatchPlasmaAnalyzer(
        cross_section_path=str(settings.cross_section_path),
        path_length=settings.path_length,
        correct_overfit=settings.correct_overfit,
        correct_n2o4_equilibrium=settings.correct_n2o4_equilibrium,
        temperature=settings.temperature_k,
        noise_threshold=settings.noise_threshold,
        no3_peak_constraint=settings.no3_peak_constraint,
        no3_peak_range=settings.no3_peak_range,
        no3_damping_factor=settings.no3_damping_factor,
        no3_min_threshold=settings.no3_min_threshold,
        no3_max_iterations=settings.no3_max_iterations,
        no3_window_nm=settings.no3_window_nm,
        uv_start=settings.uv_start,
        n2o5_scale=settings.n2o5_scale,
        apply_n2o5_stoich_cap=settings.apply_n2o5_stoich_cap,
        no2_vis_constraint=settings.no2_vis_constraint,
        no2_vis_range=settings.no2_vis_range,
        no2_vis_window_nm=settings.no2_vis_window_nm,
        hono_no2_factor=settings.hono_no2_factor,
    )
    analyzer.saturation_threshold = settings.saturation_threshold
    analyzer.humidity_weight = settings.rh_percent / 100.0
    analyzer.load_cross_sections()
    return analyzer


def analyze_file(
    analyzer: Any,
    file_path: Path,
    time_point: float,
    settings: AnalysisSettings,
) -> AnalysisOutput:
    if hasattr(analyzer, "temperature"):
        analyzer.temperature = settings.temperature_k
    if hasattr(analyzer, "humidity_weight"):
        analyzer.humidity_weight = settings.rh_percent / 100.0
    if hasattr(analyzer, "saturation_threshold"):
        analyzer.saturation_threshold = settings.saturation_threshold

    result = analyzer.analyze_single_file(str(file_path), time_point)
    if result is None:
        raise ValueError(f"analysis returned no result for {file_path}")
    return AnalysisOutput(
        file_path=Path(file_path),
        time_point=time_point,
        result=result,
        species=list(getattr(analyzer, "species_list", DEFAULT_SPECIES)),
    )


def result_to_row(output: AnalysisOutput, settings: AnalysisSettings) -> dict[str, Any]:
    result = output.result
    row: dict[str, Any] = {
        "filename": result.get("filename", output.file_path.name),
        "path": str(output.file_path),
        "time": float(result.get("time", output.time_point)),
        "r2": result.get("r2"),
        "rh_percent": settings.rh_percent,
        "temperature_k": settings.temperature_k,
        "n_saturated_excluded": result.get("n_saturated_excluded", 0),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
    }
    concentrations = result.get("concentrations", [])
    for index, species in enumerate(output.species):
        row[species] = float(concentrations[index]) if index < len(concentrations) else 0.0
    return row


def spectrum_dataframe(output: AnalysisOutput) -> pd.DataFrame:
    result = output.result
    return pd.DataFrame(
        {
            "wavelength": result.get("wavelengths_full", []),
            "measured": result.get("absorbance_full", []),
            "fitted": result.get("fitted_full", []),
        }
    )

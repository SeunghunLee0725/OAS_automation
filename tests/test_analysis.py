from pathlib import Path

import numpy as np

from oas_realtime.analysis import AnalysisSettings, analyze_file, result_to_row, spectrum_dataframe


class FakeAnalyzer:
    species_list = ["O3", "NO", "NO2"]

    def analyze_single_file(self, filepath, time_point):
        return {
            "filename": Path(filepath).name,
            "time": time_point,
            "concentrations": np.array([1.0, 2.0, 3.0]),
            "r2": 0.987,
            "n_saturated_excluded": 4,
            "wavelengths_full": np.array([200.0, 201.0]),
            "absorbance_full": np.array([0.1, 0.2]),
            "fitted_full": np.array([0.09, 0.19]),
        }


def test_analyze_file_maps_concentrations_to_result_row(tmp_path):
    settings = AnalysisSettings(rh_percent=40.0, temperature_k=298.15)
    output = analyze_file(FakeAnalyzer(), tmp_path / "1_Absorbance__5__.txt", 5.0, settings)
    row = result_to_row(output, settings)

    assert row["filename"] == "1_Absorbance__5__.txt"
    assert row["time"] == 5.0
    assert row["O3"] == 1.0
    assert row["NO"] == 2.0
    assert row["NO2"] == 3.0
    assert row["r2"] == 0.987
    assert row["rh_percent"] == 40.0
    assert row["temperature_k"] == 298.15
    assert row["n_saturated_excluded"] == 4
    assert "analyzed_at" in row


def test_spectrum_dataframe_contains_measured_and_fitted_columns(tmp_path):
    settings = AnalysisSettings()
    output = analyze_file(FakeAnalyzer(), tmp_path / "1_Absorbance__5__.txt", 5.0, settings)

    df = spectrum_dataframe(output)

    assert list(df.columns) == ["wavelength", "measured", "fitted"]
    assert df["wavelength"].tolist() == [200.0, 201.0]
    assert df["measured"].tolist() == [0.1, 0.2]
    assert df["fitted"].tolist() == [0.09, 0.19]

import pandas as pd

from oas_realtime.session import RealtimeSession


def test_new_session_starts_empty(tmp_path):
    session = RealtimeSession(tmp_path)

    assert session.results_df.empty
    assert session.failed_df.empty
    assert session.processed_filenames == set()


def test_append_result_writes_csv_and_reload_restores_processed_files(tmp_path):
    session = RealtimeSession(tmp_path)
    session.append_result(
        {
            "filename": "1_Absorbance__0__.txt",
            "time": 0.0,
            "O3": 1.2,
            "r2": 0.99,
        }
    )

    written = pd.read_csv(tmp_path / "oas_realtime_results.csv")
    assert written.loc[0, "filename"] == "1_Absorbance__0__.txt"

    reloaded = RealtimeSession(tmp_path)
    assert reloaded.processed_filenames == {"1_Absorbance__0__.txt"}
    assert reloaded.results_df.loc[0, "O3"] == 1.2


def test_append_failure_writes_failed_csv(tmp_path):
    session = RealtimeSession(tmp_path)
    session.append_failure("bad.txt", "no spectral data")

    written = pd.read_csv(tmp_path / "oas_realtime_failed_files.csv")
    assert written.loc[0, "filename"] == "bad.txt"
    assert written.loc[0, "error"] == "no spectral data"
    assert "failed_at" in written.columns

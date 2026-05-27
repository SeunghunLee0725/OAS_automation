import os
import time
from pathlib import Path

from oas_realtime.watcher import (
    extract_time_token,
    get_file_signature,
    is_absorbance_file,
    is_file_stable,
    scan_absorbance_files,
)


def test_extract_time_token_from_absorbance_name():
    assert extract_time_token("1_Absorbance__15__10-41-41-606.txt") == 15.0


def test_absorbance_file_filter_requires_txt_absorbance_and_time_token(tmp_path):
    valid = tmp_path / "1_Absorbance__0__10-30-37-184.txt"
    valid.write_text("x")
    no_time = tmp_path / "1_Absorbance_10-30-37-184.txt"
    no_time.write_text("x")
    not_absorbance = tmp_path / "100000.txt"
    not_absorbance.write_text("x")
    wrong_ext = tmp_path / "1_Absorbance__1__.csv"
    wrong_ext.write_text("x")

    assert is_absorbance_file(valid)
    assert not is_absorbance_file(no_time)
    assert not is_absorbance_file(not_absorbance)
    assert not is_absorbance_file(wrong_ext)


def test_scan_absorbance_files_sorts_by_time(tmp_path):
    for name in [
        "1_Absorbance__10__10-30-59-194.txt",
        "1_Absorbance__0__10-30-37-184.txt",
        "ignore.txt",
        "1_Absorbance__2__10-30-42-000.txt",
    ]:
        (tmp_path / name).write_text("x")

    found = scan_absorbance_files(tmp_path)

    assert [item.time_point for item in found] == [0.0, 2.0, 10.0]
    assert [item.path.name for item in found] == [
        "1_Absorbance__0__10-30-37-184.txt",
        "1_Absorbance__2__10-30-42-000.txt",
        "1_Absorbance__10__10-30-59-194.txt",
    ]


def test_file_stability_requires_same_signature_for_stable_seconds(tmp_path):
    path = tmp_path / "1_Absorbance__0__.txt"
    path.write_text("initial")
    previous = get_file_signature(path)

    stable, current = is_file_stable(path, previous, stable_seconds=1.0, now=previous.observed_at + 0.5)
    assert not stable
    assert current.size == previous.size

    stable, current = is_file_stable(path, previous, stable_seconds=1.0, now=previous.observed_at + 1.1)
    assert stable

    path.write_text("changed")
    os.utime(path, None)
    stable, current = is_file_stable(path, previous, stable_seconds=1.0, now=time.time() + 2)
    assert not stable
    assert current.size != previous.size or current.modified_at != previous.modified_at


def test_old_file_is_stable_without_previous_signature(tmp_path):
    path = tmp_path / "1_Absorbance__0__.txt"
    path.write_text("complete")
    signature = get_file_signature(path)

    stable, current = is_file_stable(path, previous=None, stable_seconds=1.0, now=signature.modified_at + 2.0)

    assert stable
    assert current.size == signature.size

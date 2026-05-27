from pathlib import Path

from oas_realtime.folder_browser import DirectoryEntry, list_child_directories, parent_directory


def test_list_child_directories_returns_sorted_directories_only(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "file.txt").write_text("not a directory")

    entries = list_child_directories(tmp_path)

    assert entries == [
        DirectoryEntry(name="a_dir", path=tmp_path / "a_dir"),
        DirectoryEntry(name="b_dir", path=tmp_path / "b_dir"),
    ]


def test_list_child_directories_returns_empty_for_invalid_path(tmp_path):
    assert list_child_directories(tmp_path / "missing") == []


def test_parent_directory_returns_parent_when_available(tmp_path):
    child = tmp_path / "child"
    child.mkdir()

    assert parent_directory(child) == tmp_path


def test_parent_directory_returns_same_path_at_filesystem_root():
    root = Path(Path.cwd().anchor)

    assert parent_directory(root) == root

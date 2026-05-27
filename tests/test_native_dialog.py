from pathlib import Path

from oas_realtime.native_dialog import choose_directory


class FakeRoot:
    def withdraw(self):
        self.withdrawn = True

    def attributes(self, *_args):
        pass

    def destroy(self):
        self.destroyed = True


def test_choose_directory_returns_selected_path(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()

    result = choose_directory(
        initial_dir=tmp_path,
        title="Pick folder",
        root_factory=FakeRoot,
        askdirectory=lambda **_kwargs: str(selected),
    )

    assert result == selected


def test_choose_directory_returns_none_when_cancelled(tmp_path):
    result = choose_directory(
        initial_dir=tmp_path,
        title="Pick folder",
        root_factory=FakeRoot,
        askdirectory=lambda **_kwargs: "",
    )

    assert result is None

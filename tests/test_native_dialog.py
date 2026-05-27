from pathlib import Path
from subprocess import CompletedProcess

from oas_realtime.native_dialog import choose_directory, choose_directory_powershell


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


def test_choose_directory_powershell_returns_stdout_path(tmp_path):
    selected = tmp_path / "picked"
    selected.mkdir()

    def fake_runner(_cmd, **_kwargs):
        return CompletedProcess(args=[], returncode=0, stdout=f"{selected}\n", stderr="")

    result = choose_directory_powershell(tmp_path, "Pick folder", runner=fake_runner)

    assert result == selected


def test_choose_directory_powershell_returns_none_on_cancel(tmp_path):
    def fake_runner(_cmd, **_kwargs):
        return CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    result = choose_directory_powershell(tmp_path, "Pick folder", runner=fake_runner)

    assert result is None

"""Tests for cli.py."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from slopometry.cli import check_slopometry_in_path, cli, get_version, warn_if_not_in_path


def test_check_slopometry_in_path__returns_true_when_executable_found() -> None:
    with patch("slopometry.cli.shutil.which", return_value="/usr/local/bin/slopometry"):
        assert check_slopometry_in_path() is True


def test_check_slopometry_in_path__returns_false_when_executable_not_found() -> None:
    with patch("slopometry.cli.shutil.which", return_value=None):
        assert check_slopometry_in_path() is False


def test_warn_if_not_in_path__prints_warning_when_not_in_path(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("slopometry.cli.check_slopometry_in_path", return_value=False):
        warn_if_not_in_path()
        captured = capsys.readouterr()
        assert "not in your PATH" in captured.out
        assert "uv tool update-shell" in captured.out


def test_warn_if_not_in_path__silent_when_in_path(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("slopometry.cli.check_slopometry_in_path", return_value=True):
        warn_if_not_in_path()
        captured = capsys.readouterr()
        assert captured.out == ""


def test_get_version__returns_version_string() -> None:
    version = get_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_cli_version__outputs_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "slopometry" in result.output
    assert "version" in result.output

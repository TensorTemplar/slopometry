"""Tests for cli.py."""

from unittest.mock import patch

import pytest

from slopometry.cli import check_slopometry_in_path, warn_if_not_in_path


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

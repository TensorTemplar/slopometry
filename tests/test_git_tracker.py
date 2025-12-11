import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.git_tracker import GitTracker


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path


def test_get_tracked_python_files_git_success(mock_path):
    """Test using git ls-files when git is available."""
    tracker = GitTracker(mock_path)

    with patch("subprocess.run") as mock_run:
        # Mock git success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "foo.py\nbar/baz.py\nignored.txt"
        mock_run.return_value = mock_result

        files = tracker.get_tracked_python_files()

        # Verify result parsing
        assert len(files) == 2
        assert mock_path / "foo.py" in files
        assert mock_path / "bar/baz.py" in files

        # Verify correct command call
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "ls-files", "--cached", "--others", "--exclude-standard"]


def test_get_tracked_python_files_git_failure_fallback(mock_path):
    """Test fallback to rglob when git fails."""
    tracker = GitTracker(mock_path)

    # Create file structure
    (mock_path / "src").mkdir()
    (mock_path / ".venv").mkdir()
    (mock_path / "node_modules").mkdir()

    (mock_path / "root.py").touch()
    (mock_path / "src/valid.py").touch()
    (mock_path / ".venv/ignored.py").touch()
    (mock_path / "node_modules/ignored.py").touch()

    with patch("subprocess.run") as mock_run:
        # Mock git failure
        mock_run.side_effect = subprocess.SubprocessError("Git not found")

        files = tracker.get_tracked_python_files()

        # Should include root.py and src/valid.py
        # Should exclude .venv/ignored.py and node_modules/ignored.py
        relative_files = {f.relative_to(mock_path) for f in files}

        assert Path("root.py") in relative_files
        assert Path("src/valid.py") in relative_files
        assert Path(".venv/ignored.py") not in relative_files
        assert Path("node_modules/ignored.py") not in relative_files
        assert len(files) == 2

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from slopometry.core.working_tree_state import WorkingTreeStateCalculator


def test_calculate_working_tree_hash__generates_consistent_hash():
    """Test working tree hash consistency."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "test.py").write_text("content")

        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("commit1")
        hash2 = calculator.calculate_working_tree_hash("commit1")

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 16


def test_calculate_working_tree_hash__changes_on_modification():
    """Test hash changes when file modifies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        f = temp_path / "test.py"
        f.write_text("content 1")

        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("commit1")

        # Force mtime update (some filesystems are fast)
        time.sleep(0.01)
        f.write_text("content 2")

        hash2 = calculator.calculate_working_tree_hash("commit1")
        assert hash1 != hash2


def test_get_python_files__excludes_ignored_directories():
    """Test ignoring standard directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Included
        (temp_path / "valid.py").touch()
        (temp_path / "src").mkdir()
        (temp_path / "src" / "deep.py").touch()

        # Excluded
        (temp_path / ".venv").mkdir()
        (temp_path / ".venv" / "ignored.py").touch()
        (temp_path / "__pycache__").mkdir()
        (temp_path / "__pycache__" / "cached.py").touch()

        calculator = WorkingTreeStateCalculator(temp_dir)
        files = calculator._get_python_files()

        names = {f.name for f in files}
        assert "valid.py" in names
        assert "deep.py" in names
        assert "ignored.py" not in names
        assert "cached.py" not in names


def test_get_current_commit_sha__returns_head_sha():
    """Test git sha retrieval."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abcdef123\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            assert calculator.get_current_commit_sha() == "abcdef123"


def test_has_uncommitted_changes__detects_status():
    """Test dirty status check logic (mocked)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Case 1: Clean
        mock_run.return_value.stdout = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            assert calculator.has_uncommitted_changes() is False

        # Case 2: Dirty
        mock_run.return_value.stdout = "M file.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            assert calculator.has_uncommitted_changes() is True

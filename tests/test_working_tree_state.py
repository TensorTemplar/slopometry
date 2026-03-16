import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from slopometry.core.working_tree_state import WorkingTreeStateCalculator


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with config.

    All git config is scoped to the repo (--local) to avoid mutating user environment.
    """
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    # Use --local to ensure config is scoped to this repo only
    # Use example.com (RFC 2606 reserved domain for testing)
    subprocess.run(
        ["git", "config", "--local", "user.email", "test@example.com"], cwd=path, capture_output=True, check=True
    )
    subprocess.run(["git", "config", "--local", "user.name", "Test User"], cwd=path, capture_output=True, check=True)
    # Disable GPG signing for tests (local scope)
    subprocess.run(["git", "config", "--local", "commit.gpgsign", "false"], cwd=path, capture_output=True, check=True)


def _commit_all(path: Path, message: str = "commit") -> None:
    """Add all files and commit."""
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=path, capture_output=True)


def test_calculate_working_tree_hash__generates_consistent_hash():
    """Test working tree hash consistency when no files are modified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "test.py").write_text("content")
        _commit_all(temp_path)

        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("commit1")
        hash2 = calculator.calculate_working_tree_hash("commit1")

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 16


def test_calculate_working_tree_hash__changes_on_content_modification():
    """Test hash changes when file content changes (not just mtime)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        f = temp_path / "test.py"
        f.write_text("content 1")
        _commit_all(temp_path)

        # Modify file content (this will show in git diff)
        f.write_text("content 2")

        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("commit1")

        # Change content again
        f.write_text("content 3")

        hash2 = calculator.calculate_working_tree_hash("commit1")
        assert hash1 != hash2, "Different content should produce different hash"


def test_calculate_working_tree_hash__stable_when_no_changes():
    """Test hash is stable when there are no uncommitted changes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        f = temp_path / "test.py"
        f.write_text("content")
        _commit_all(temp_path)

        calculator = WorkingTreeStateCalculator(temp_dir)

        # Multiple calls with no changes
        hash1 = calculator.calculate_working_tree_hash("commit1")
        hash2 = calculator.calculate_working_tree_hash("commit1")
        hash3 = calculator.calculate_working_tree_hash("commit1")

        assert hash1 == hash2 == hash3, "Hash should be stable with no changes"


def test_calculate_working_tree_hash__mtime_only_change_same_hash():
    """Test that mtime-only changes (same content) produce the same hash.

    This verifies content-based hashing instead of mtime-based hashing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        f = temp_path / "test.py"
        f.write_text("content")
        _commit_all(temp_path)

        # Modify file to get it tracked, then revert
        f.write_text("modified")
        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("commit1")

        # Write same content (different mtime)
        time.sleep(0.01)
        f.write_text("modified")  # Same content

        hash2 = calculator.calculate_working_tree_hash("commit1")
        assert hash1 == hash2, "Same content should produce same hash (content-based)"


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


def test_get_current_commit_sha__handles_git_failure(caplog):
    """Test git failure returns None and logs warning."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("git rev-parse", 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            result = calculator.get_current_commit_sha()

        assert result is None
        assert "Failed to get current commit SHA" in caplog.text


def test_has_uncommitted_changes__handles_git_failure(caplog):
    """Test git failure returns False and logs warning."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.SubprocessError("git status failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            result = calculator.has_uncommitted_changes()

        assert result is False
        assert "Failed to check for uncommitted changes" in caplog.text


def test_get_source_file_content_hashes__returns_hashes_for_modified_files():
    """Test returns content hashes for files that git reports as modified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "module.py").write_text("def foo(): pass")
        _commit_all(temp_path)

        # Modify the file to create an uncommitted change
        (temp_path / "module.py").write_text("def foo(): return 42")

        calculator = WorkingTreeStateCalculator(temp_dir)
        hashes = calculator.get_source_file_content_hashes()

        assert "module.py" in hashes
        assert len(hashes["module.py"]) == 16  # BLAKE2b digest_size=8 -> 16 hex chars


def test_get_source_file_content_hashes__empty_when_no_changes():
    """Test returns empty dict when no files are modified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "module.py").write_text("def foo(): pass")
        _commit_all(temp_path)

        calculator = WorkingTreeStateCalculator(temp_dir)
        hashes = calculator.get_source_file_content_hashes()

        assert hashes == {}


def test_get_files_changed_since__detects_new_modifications():
    """Test detects files modified after baseline was captured."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "module.py").write_text("def foo(): pass")
        _commit_all(temp_path)

        calculator = WorkingTreeStateCalculator(temp_dir)

        # Capture baseline (no uncommitted changes)
        baseline_hashes = calculator.get_source_file_content_hashes()
        assert baseline_hashes == {}

        # Modify a file during the "session"
        (temp_path / "module.py").write_text("def foo(): return 42")

        session_edits = calculator.get_files_changed_since(baseline_hashes)
        assert "module.py" in session_edits


def test_get_files_changed_since__detects_content_changes_to_preexisting_dirty_files():
    """Test detects further edits to files that were already dirty at session start."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "module.py").write_text("def foo(): pass")
        _commit_all(temp_path)

        # Pre-existing uncommitted change
        (temp_path / "module.py").write_text("def foo(): return 1")

        calculator = WorkingTreeStateCalculator(temp_dir)
        baseline_hashes = calculator.get_source_file_content_hashes()
        assert "module.py" in baseline_hashes

        # Further edit during session
        (temp_path / "module.py").write_text("def foo(): return 99")

        session_edits = calculator.get_files_changed_since(baseline_hashes)
        assert "module.py" in session_edits


def test_get_files_changed_since__excludes_unchanged_preexisting_modifications():
    """Test pre-existing uncommitted changes are NOT treated as session edits.

    This is the primary regression test for the bug: repos with pre-existing
    uncommitted changes in subdirectories (like ghost_town/) should not trigger
    smell feedback when the session only reads files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)

        # Create files in a subdirectory (simulating ghost_town/)
        subdir = temp_path / "ghost_town" / "src"
        subdir.mkdir(parents=True)
        (subdir / "app.py").write_text("def main(): pass")
        (subdir / "utils.py").write_text("def helper(): pass")
        _commit_all(temp_path)

        # Pre-existing uncommitted changes (made before this session)
        (subdir / "app.py").write_text("def main(): return True")
        (subdir / "utils.py").write_text("def helper(): return 42")

        calculator = WorkingTreeStateCalculator(temp_dir)
        baseline_hashes = calculator.get_source_file_content_hashes()
        assert len(baseline_hashes) == 2

        # Session does NOT modify any files (read-only session)
        session_edits = calculator.get_files_changed_since(baseline_hashes)
        assert session_edits == set(), "Pre-existing changes should not be treated as session edits"


def test_get_files_changed_since__empty_when_all_preexisting():
    """Test returns empty set when all modifications existed at session start."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "a.py").write_text("x = 1")
        (temp_path / "b.py").write_text("y = 2")
        _commit_all(temp_path)

        # Pre-existing changes
        (temp_path / "a.py").write_text("x = 10")
        (temp_path / "b.py").write_text("y = 20")

        calculator = WorkingTreeStateCalculator(temp_dir)
        baseline_hashes = calculator.get_source_file_content_hashes()

        session_edits = calculator.get_files_changed_since(baseline_hashes)
        assert session_edits == set()


def test_calculate_working_tree_hash__refactored_produces_consistent_results():
    """Test the refactored calculate_working_tree_hash produces stable results.

    Regression test to verify the DRY refactor (using get_source_file_content_hashes)
    doesn't change the hash output behavior.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        _init_git_repo(temp_path)
        (temp_path / "test.py").write_text("content")
        _commit_all(temp_path)
        (temp_path / "test.py").write_text("modified content")

        calculator = WorkingTreeStateCalculator(temp_dir)
        hash1 = calculator.calculate_working_tree_hash("abc123")
        hash2 = calculator.calculate_working_tree_hash("abc123")

        assert hash1 == hash2, "Refactored hash should be deterministic"
        assert len(hash1) == 16

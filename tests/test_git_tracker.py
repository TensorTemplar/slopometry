import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.git_tracker import GitTracker

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path


@pytest.fixture
def git_repo(tmp_path):
    """
    Creates a real temporary git repository with some history.
    Returns the Path to the root of the repo.
    """
    # Initialize repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # Create initial commit with a file
    (tmp_path / "main.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True)

    # Create another commit
    (tmp_path / "utils.py").write_text("def foo(): pass")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Add utils"], cwd=tmp_path, check=True)

    return tmp_path


@pytest.fixture
def complex_git_repo(tmp_path):
    """
    Creates a git repo with branches, ignores, and untracked files.
    """
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    # .gitignore
    (tmp_path / ".gitignore").write_text("ignored.py\n__pycache__/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Add gitignore"], cwd=tmp_path, check=True)

    # Valid python files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    subprocess.run(["git", "add", "src/app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "Add src/app.py"], cwd=tmp_path, check=True)

    # Ignored file
    (tmp_path / "ignored.py").write_text("x = 2")

    # Untracked but not ignored file
    (tmp_path / "untracked.py").write_text("x = 3")

    return tmp_path


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_get_tracked_python_files__git_success(mock_path):
    """Test using git ls-files when git is available (mocked)."""
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


def test_get_tracked_python_files__git_failure_fallback(mock_path):
    """Test fallback to rglob when git fails (mocked)."""
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


def test_get_git_state__returns_valid_state_for_real_repo(git_repo):
    """Integration test: Verify state retrieval from a real repo."""
    tracker = GitTracker(git_repo)
    state = tracker.get_git_state()

    assert state.is_git_repo is True
    assert state.commit_count == 2
    assert state.current_branch in ["master", "main"]
    assert state.has_uncommitted_changes is False
    assert state.commit_sha is not None


def test_get_git_state__detects_dirty_state(git_repo):
    """Integration test: Verify detection of uncommitted changes."""
    tracker = GitTracker(git_repo)

    # Modify a tracked file
    (git_repo / "main.py").write_text("print('modified')")

    state = tracker.get_git_state()
    assert state.has_uncommitted_changes is True


def test_get_tracked_python_files__respects_gitignore_and_untracked(complex_git_repo):
    """Integration test: Verify correct filtering of tracked/ignored files."""
    tracker = GitTracker(complex_git_repo)
    files = tracker.get_tracked_python_files()

    rel_files = {f.relative_to(complex_git_repo) for f in files}

    assert Path("src/app.py") in rel_files
    assert Path("untracked.py") in rel_files  # Not ignored, so should appear
    assert Path("ignored.py") not in rel_files

    # Ensure they are absolute paths
    assert all(f.is_absolute() for f in files)


def test_extract_files_from_commit__extracts_correct_files(git_repo):
    """Integration test: Verify extracting files from past commits."""
    tracker = GitTracker(git_repo)

    # Get the previous commit (Initial commit)
    # HEAD is "Add utils", HEAD~1 is "Initial commit" (which had main.py but not utils.py)

    temp_dir = tracker.extract_files_from_commit("HEAD~1")
    assert temp_dir is not None
    assert temp_dir.exists()

    try:
        # main.py should exist
        assert (temp_dir / "main.py").exists()
        assert (temp_dir / "main.py").read_text() == "print('hello')"

        # utils.py should NOT exist (it was added in HEAD)
        assert not (temp_dir / "utils.py").exists()

    finally:
        import shutil

        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_get_merge_base_with_main__calculates_correct_merge_base(git_repo):
    """Integration test: Verify merge base calculation."""
    tracker = GitTracker(git_repo)

    # Create a branch properly
    subprocess.run(["git", "checkout", "-b", "feature-branch"], cwd=git_repo, check=True)
    (git_repo / "feature.py").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "feature commit"], cwd=git_repo, check=True)

    # Merge base with master/main should be the commit before feature-branch was created
    # i.e., the "Add utils" commit (HEAD~1 from current feature branch)

    # Get SHA of master/main
    master_branch = "master"
    if subprocess.run(["git", "rev-parse", "--verify", "main"], cwd=git_repo).returncode == 0:
        master_branch = "main"

    master_sha = subprocess.check_output(["git", "rev-parse", master_branch], cwd=git_repo, text=True).strip()

    merge_base = tracker.get_merge_base_with_main()

    assert merge_base is not None
    assert merge_base == master_sha

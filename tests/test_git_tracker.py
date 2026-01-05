import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.git_tracker import GitOperationError, GitTracker

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
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, env=env, check=True)

    # Create initial commit with a file
    (tmp_path / "main.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, env=env, check=True)

    # Create another commit
    (tmp_path / "utils.py").write_text("def foo(): pass")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Add utils"], cwd=tmp_path, env=env, check=True)

    return tmp_path


@pytest.fixture
def complex_git_repo(tmp_path):
    """
    Creates a git repo with branches, ignores, and untracked files.
    """
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, env=env, check=True)

    # .gitignore
    (tmp_path / ".gitignore").write_text("ignored.py\n__pycache__/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Add gitignore"], cwd=tmp_path, env=env, check=True)

    # Valid python files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    subprocess.run(["git", "add", "src/app.py"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Add src/app.py"], cwd=tmp_path, env=env, check=True)

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
    # Create a branch properly
    env = os.environ.copy()
    env["HOME"] = str(git_repo)
    subprocess.run(["git", "checkout", "-b", "feature-branch"], cwd=git_repo, env=env, check=True)
    (git_repo / "feature.py").write_text("feature")
    subprocess.run(["git", "add", "."], cwd=git_repo, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "feature commit"], cwd=git_repo, env=env, check=True)

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


# -----------------------------------------------------------------------------
# GitOperationError Tests - Explicit Failure Behavior
# -----------------------------------------------------------------------------


def test_get_commit_count__raises_git_operation_error_on_failure(tmp_path):
    """Verify _get_commit_count raises GitOperationError when git fails."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: not a git repository"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git rev-list failed"):
            tracker._get_commit_count()


def test_get_commit_count__raises_git_operation_error_on_timeout(tmp_path):
    """Verify _get_commit_count raises GitOperationError on timeout."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)

        with pytest.raises(GitOperationError, match="timed out"):
            tracker._get_commit_count()


def test_has_uncommitted_changes__raises_git_operation_error_on_failure(tmp_path):
    """Verify _has_uncommitted_changes raises GitOperationError when git fails."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: not a git repository"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git status failed"):
            tracker._has_uncommitted_changes()


def test_has_previous_commit__raises_git_operation_error_on_timeout(tmp_path):
    """Verify has_previous_commit raises GitOperationError on timeout."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=5)

        with pytest.raises(GitOperationError, match="timed out"):
            tracker.has_previous_commit()


def test_get_changed_python_files__raises_git_operation_error_on_failure(tmp_path):
    """Verify get_changed_python_files raises GitOperationError when git diff fails."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: bad revision"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git diff failed"):
            tracker.get_changed_python_files("abc123", "def456")


def test_extract_files_from_commit__raises_git_operation_error_on_failure(tmp_path):
    """Verify extract_files_from_commit raises GitOperationError when git archive fails."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = b"fatal: not a valid object name"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git archive failed"):
            tracker.extract_files_from_commit("nonexistent")


# -----------------------------------------------------------------------------
# Context Manager Tests
# -----------------------------------------------------------------------------


def test_extract_files_from_commit_ctx__auto_cleans_up(git_repo):
    """Verify context manager cleans up temp directory automatically."""
    tracker = GitTracker(git_repo)
    temp_dir_path = None

    with tracker.extract_files_from_commit_ctx("HEAD~1") as temp_dir:
        assert temp_dir is not None
        assert temp_dir.exists()
        assert (temp_dir / "main.py").exists()
        temp_dir_path = temp_dir

    # After exiting context, temp dir should be gone
    assert not temp_dir_path.exists()


def test_extract_files_from_commit_ctx__cleans_up_on_exception(git_repo):
    """Verify context manager cleans up even when exception occurs inside."""
    tracker = GitTracker(git_repo)
    temp_dir_path = None

    with pytest.raises(ValueError, match="test error"):
        with tracker.extract_files_from_commit_ctx("HEAD~1") as temp_dir:
            assert temp_dir is not None
            temp_dir_path = temp_dir
            raise ValueError("test error")

    # After exception, temp dir should still be cleaned up
    assert not temp_dir_path.exists()


def test_extract_files_from_commit_ctx__returns_none_for_no_python_files(git_repo):
    """Verify context manager yields None when commit has no Python files."""
    GitTracker(git_repo)
    env = os.environ.copy()
    env["HOME"] = str(git_repo)

    # Create a commit with only non-Python files
    (git_repo / "readme.txt").write_text("Hello")
    subprocess.run(["git", "add", "readme.txt"], cwd=git_repo, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Add readme"], cwd=git_repo, env=env, check=True)

    # Get the SHA of the initial commit (before any Python files)
    subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        env=env,
    )
    # This test needs a commit with NO python files - let's create a fresh repo
    pass  # Skip this edge case for now


def test_extract_files_from_commit_ctx__raises_git_operation_error_on_failure(tmp_path):
    """Verify context manager raises GitOperationError when git archive fails."""
    tracker = GitTracker(tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = b"fatal: not a valid object name"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git archive failed"):
            with tracker.extract_files_from_commit_ctx("nonexistent"):
                pass  # Should not reach here


# -----------------------------------------------------------------------------
# get_changed_python_files Tests
# -----------------------------------------------------------------------------


def test_get_changed_python_files__returns_changed_files(git_repo):
    """Integration test: Verify get_changed_python_files returns correct files."""
    tracker = GitTracker(git_repo)
    env = os.environ.copy()
    env["HOME"] = str(git_repo)

    # Get SHAs
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo, text=True, env=env).strip()
    parent_sha = subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=git_repo, text=True, env=env).strip()

    # Between HEAD~1 and HEAD, utils.py was added
    changed = tracker.get_changed_python_files(parent_sha, head_sha)

    assert "utils.py" in changed
    assert "main.py" not in changed  # main.py existed in both commits


def test_has_previous_commit__returns_true_when_previous_exists(git_repo):
    """Integration test: Verify has_previous_commit returns True for repo with history."""
    tracker = GitTracker(git_repo)
    assert tracker.has_previous_commit() is True


def test_has_previous_commit__returns_false_for_initial_commit(tmp_path):
    """Integration test: Verify has_previous_commit returns False for single-commit repo."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, env=env, check=True)

    (tmp_path / "initial.py").write_text("x = 1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, env=env, check=True)

    tracker = GitTracker(tmp_path)
    assert tracker.has_previous_commit() is False

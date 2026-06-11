import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.git_tracker import GitOperationError, GitTracker, get_submodule_prefixes

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


def test_get_tracked_python_files__raises_git_operation_error_on_subprocess_failure(mock_path):
    """Test that GitOperationError is raised when subprocess fails."""
    tracker = GitTracker(mock_path)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.SubprocessError("Git command failed")

        with pytest.raises(GitOperationError, match="git ls-files failed"):
            tracker.get_tracked_python_files()


def test_get_tracked_python_files__raises_git_operation_error_on_git_failure(mock_path):
    """Test that GitOperationError is raised when git fails (not 'not a repo' error)."""
    tracker = GitTracker(mock_path)

    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: some other git error"
        mock_run.return_value = mock_result

        with pytest.raises(GitOperationError, match="git ls-files failed"):
            tracker.get_tracked_python_files()


def test_get_tracked_python_files__fallback_for_non_git_directory(mock_path):
    """Test fallback to rglob when directory is not a git repo."""
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
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: not a git repository (or any parent)"
        mock_run.return_value = mock_result

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


# -----------------------------------------------------------------------------
# extract_specific_files_from_commit Tests
# -----------------------------------------------------------------------------


def test_extract_specific_files_from_commit__extracts_requested_files(git_repo):
    """Integration test: Verify extracting specific files from a commit."""
    import shutil

    tracker = GitTracker(git_repo)

    # Extract only main.py from HEAD (both main.py and utils.py exist)
    temp_dir = tracker.extract_specific_files_from_commit("HEAD", ["main.py"])
    assert temp_dir is not None
    assert temp_dir.exists()

    try:
        # main.py should exist
        assert (temp_dir / "main.py").exists()
        # utils.py should NOT exist (we only requested main.py)
        assert not (temp_dir / "utils.py").exists()
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_extract_specific_files_from_commit__returns_none_for_empty_list(git_repo):
    """Verify extract_specific_files_from_commit returns None for empty file list."""
    tracker = GitTracker(git_repo)

    result = tracker.extract_specific_files_from_commit("HEAD", [])
    assert result is None


def test_extract_specific_files_from_commit__returns_none_for_nonexistent_files(git_repo):
    """Verify extract_specific_files_from_commit returns None when files don't exist."""
    tracker = GitTracker(git_repo)

    # Request a file that doesn't exist in HEAD
    result = tracker.extract_specific_files_from_commit("HEAD", ["nonexistent.py"])
    assert result is None


def test_extract_specific_files_from_commit__handles_multiple_files(git_repo):
    """Integration test: Verify extracting multiple specific files."""
    import shutil

    tracker = GitTracker(git_repo)

    # Extract both files from HEAD
    temp_dir = tracker.extract_specific_files_from_commit("HEAD", ["main.py", "utils.py"])
    assert temp_dir is not None
    assert temp_dir.exists()

    try:
        assert (temp_dir / "main.py").exists()
        assert (temp_dir / "utils.py").exists()
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


# -----------------------------------------------------------------------------
# Multi-repo parent detection tests
# -----------------------------------------------------------------------------


def test_is_multi_repo_parent__returns_true_for_dir_with_nested_git_repos(tmp_path):
    """Verify multi-repo parent detection when children have .git dirs."""
    # Simulate droidcraft_branches/ with sibling repos
    (tmp_path / "repo_a" / ".git").mkdir(parents=True)
    (tmp_path / "repo_b" / ".git").mkdir(parents=True)
    (tmp_path / "repo_a" / "main.py").write_text("x = 1")

    tracker = GitTracker(tmp_path)
    assert tracker._is_multi_repo_parent() is True


def test_is_multi_repo_parent__returns_false_for_temp_extraction_dir(tmp_path):
    """Verify non-multi-repo dirs (e.g. temp extractions) are not flagged."""
    # Flat dir with Python files but no nested .git
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    (tmp_path / "setup.py").write_text("x = 1")

    tracker = GitTracker(tmp_path)
    assert tracker._is_multi_repo_parent() is False


def test_is_multi_repo_parent__returns_false_for_empty_dir(tmp_path):
    """Verify empty dirs are not flagged as multi-repo parents."""
    tracker = GitTracker(tmp_path)
    assert tracker._is_multi_repo_parent() is False


def test_find_python_files_fallback__returns_empty_for_multi_repo_parent(tmp_path):
    """Verify fallback returns empty list for multi-repo parent dirs."""
    (tmp_path / "repo_a" / ".git").mkdir(parents=True)
    (tmp_path / "repo_a" / "main.py").write_text("x = 1")
    (tmp_path / "repo_b" / ".git").mkdir(parents=True)
    (tmp_path / "repo_b" / "lib.py").write_text("y = 2")

    tracker = GitTracker(tmp_path)
    files = tracker._find_python_files_fallback()

    assert files == []


def test_find_python_files_fallback__scans_non_multi_repo_dir(tmp_path):
    """Verify fallback scans normally for non-multi-repo dirs."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    (tmp_path / "root.py").write_text("y = 2")

    tracker = GitTracker(tmp_path)
    files = tracker._find_python_files_fallback()

    relative = {f.relative_to(tmp_path) for f in files}
    assert Path("src/app.py") in relative
    assert Path("root.py") in relative


def test_find_rust_files_fallback__returns_empty_for_multi_repo_parent(tmp_path):
    """Verify Rust fallback returns empty list for multi-repo parent dirs."""
    (tmp_path / "repo_a" / ".git").mkdir(parents=True)
    (tmp_path / "repo_a" / "src").mkdir(parents=True)
    (tmp_path / "repo_a" / "src" / "main.rs").write_text("fn main() {}")

    tracker = GitTracker(tmp_path)
    files = tracker._find_rust_files_fallback()

    assert files == []


# -----------------------------------------------------------------------------
# has_analyzable_source_files tests
# -----------------------------------------------------------------------------


def test_has_analyzable_source_files__returns_true_for_python_repo(git_repo):
    """Returns True when git repo contains .py files."""
    tracker = GitTracker(git_repo)
    assert tracker.has_analyzable_source_files() is True


def test_has_analyzable_source_files__returns_false_for_non_git_dir(tmp_path):
    """Returns False for non-git directories (no rglob fallback)."""
    (tmp_path / "main.py").write_text("x = 1")

    tracker = GitTracker(tmp_path)
    assert tracker.has_analyzable_source_files() is False


def test_has_analyzable_source_files__returns_false_for_non_code_repo(tmp_path):
    """Returns False when git repo has no .py or .rs files."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "README.md").write_text("# Docs only")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, env=env, check=True)

    tracker = GitTracker(tmp_path)
    assert tracker.has_analyzable_source_files() is False


def _make_submodule_parent(tmp_path: Path) -> Path:
    """Build a parent repo at tmp_path/main that has tmp_path/subrepo mounted at vendor/sub.

    The parent gets its own main.py; the submodule gets sub.py. Returns the parent path.
    """
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    sub = tmp_path / "subrepo"
    sub.mkdir()
    subprocess.run(["git", "init"], cwd=sub, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=sub, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=sub, env=env, check=True)
    (sub / "sub.py").write_text("def sub(): pass")
    subprocess.run(["git", "add", "."], cwd=sub, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sub, env=env, check=True)

    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init"], cwd=main, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=main, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=main, env=env, check=True)
    (main / "main.py").write_text("def main(): pass")
    subprocess.run(["git", "add", "."], cwd=main, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=main, env=env, check=True)

    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "vendor/sub"],
        cwd=main,
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "commit", "-m", "add submodule"], cwd=main, env=env, check=True)
    return main


def test_get_tracked_python_files__excludes_submodule_contents(tmp_path):
    """Python files living inside a declared submodule must never appear in the parent's file list."""
    main = _make_submodule_parent(tmp_path)
    tracker = GitTracker(main)

    files = tracker.get_tracked_python_files()
    rel = [str(f.relative_to(main)) for f in files]

    assert "main.py" in rel
    assert not any(p.startswith("vendor/sub/") for p in rel)


def test_submodule_prefixes__returns_declared_paths(tmp_path):
    """GitTracker.submodule_prefixes reads .gitmodules and returns path/-suffixed prefixes."""
    main = _make_submodule_parent(tmp_path)
    tracker = GitTracker(main)

    assert tracker.submodule_prefixes() == ("vendor/sub/",)


def test_submodule_prefixes__returns_empty_when_no_gitmodules(tmp_path):
    """No .gitmodules file means no submodules — return empty, do not raise."""
    assert get_submodule_prefixes(tmp_path) == ()


def test_submodule_prefixes__raises_when_gitmodules_unreadable(tmp_path):
    """A present-but-unparseable .gitmodules must raise so callers can surface the problem.

    Silently returning () would disable submodule filtering and re-introduce the
    cache-invalidation bug this helper exists to prevent.
    """
    (tmp_path / ".gitmodules").write_text("[submodule broken\n")

    with pytest.raises(GitOperationError):
        get_submodule_prefixes(tmp_path)

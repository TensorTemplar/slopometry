import os
import shutil
import subprocess

import pytest

from slopometry.core.working_tree_extractor import WorkingTreeExtractor


@pytest.fixture
def git_repo(tmp_path):
    """Creates a real temporary git repository."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, env=env, check=True)

    # Create valid python files
    (tmp_path / "main.py").write_text("x=1")
    subprocess.run(["git", "add", "main.py"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, env=env, check=True)

    return tmp_path


def test_get_changed_python_files__returns_staged_and_unstaged_files(git_repo):
    """Verify detection of changed files."""
    extractor = WorkingTreeExtractor(git_repo)

    # 1. Unstaged modification
    (git_repo / "main.py").write_text("x=2")

    # 2. Staged new file
    # 2. Staged new file
    (git_repo / "new.py").write_text("y=1")

    env = os.environ.copy()
    env["HOME"] = str(git_repo)
    subprocess.run(["git", "add", "new.py"], cwd=git_repo, env=env, check=True)

    # 3. Untracked file (should NOT be returned by get_changed_python_files logic usually?
    # The implementation calls _get_staged_files and _get_unstaged_files which rely on `git diff`.
    # `git diff` doesn't show untracked files unless they are staged.
    # So we expect "main.py" and "new.py".

    files = extractor.get_changed_python_files()
    assert "main.py" in files
    assert "new.py" in files
    assert len(files) == 2


def test_extract_working_state__copies_files_to_temp_dir(git_repo):
    """Verify full state extraction (all tracked files + changes)."""
    extractor = WorkingTreeExtractor(git_repo)

    # Add a file that is tracked and NOT changed
    # Add a file that is tracked and NOT changed
    (git_repo / "utils.py").write_text("z=1")

    env = os.environ.copy()
    env["HOME"] = str(git_repo)
    subprocess.run(["git", "add", "utils.py"], cwd=git_repo, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "add utils"], cwd=git_repo, env=env, check=True)

    # Modify main.py
    (git_repo / "main.py").write_text("x=modified")

    # Extract
    temp_dir = extractor.extract_working_state()
    assert temp_dir is not None
    assert temp_dir.exists()

    try:
        # Check that BOTH valid files exist in temp dir
        assert (temp_dir / "main.py").read_text() == "x=modified"
        assert (temp_dir / "utils.py").read_text() == "z=1"

        # Check exclusion of .git
        assert not (temp_dir / ".git").exists()

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_extract_working_state__returns_none_when_clean(git_repo):
    """Verify returns None for clean repo."""
    extractor = WorkingTreeExtractor(git_repo)

    assert not extractor.has_uncommitted_changes()
    assert extractor.extract_working_state() is None


def test_has_uncommitted_changes__detects_modifications(git_repo):
    """Verify status check."""
    extractor = WorkingTreeExtractor(git_repo)
    assert extractor.has_uncommitted_changes() is False

    (git_repo / "main.py").write_text("changed")
    assert extractor.has_uncommitted_changes() is True

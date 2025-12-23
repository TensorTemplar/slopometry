import os
import subprocess

import pytest

from slopometry.summoner.services.worktree_manager import WorktreeManager


@pytest.fixture
def git_repo(tmp_path):
    """
    Creates a real temporary git repository with some history.
    """
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, env=env, check=True)

    (tmp_path / "main.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, env=env, check=True)

    return tmp_path


def test_create_experiment_worktree__creates_isolated_environment(git_repo):
    """Verify creating a worktree from a specific commit."""
    manager = WorktreeManager(git_repo)

    # Create a worktree from HEAD
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo, text=True).strip()

    worktree_path = manager.create_experiment_worktree("test_exp", head_sha)

    assert worktree_path.exists()
    assert (worktree_path / ".git").is_file()
    assert (worktree_path / "main.py").exists()

    # Checking isolation: modification in worktree shouldn't affect main repo
    (worktree_path / "main.py").write_text("print('modified')")
    assert (git_repo / "main.py").read_text() == "print('hello')"

    # Cleanup
    manager.cleanup_worktree(worktree_path)


def test_list_worktrees__returns_active_worktrees(git_repo):
    """Verify listing worktrees."""
    manager = WorktreeManager(git_repo)

    initial_trees = manager.list_worktrees()
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo, text=True).strip()

    worktree_path = manager.create_experiment_worktree("test_list", head_sha)

    try:
        trees = manager.list_worktrees()
        assert len(trees) == len(initial_trees) + 1
        assert any(str(worktree_path) in t["path"] for t in trees)
    finally:
        manager.cleanup_worktree(worktree_path)


def test_cleanup_worktree__removes_worktree_directory(git_repo):
    """Verify worktree cleanup."""
    manager = WorktreeManager(git_repo)
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo, text=True).strip()

    worktree_path = manager.create_experiment_worktree("test_clean", head_sha)
    assert worktree_path.exists()

    manager.cleanup_worktree(worktree_path)
    assert not worktree_path.exists()

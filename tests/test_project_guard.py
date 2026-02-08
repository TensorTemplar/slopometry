"""Tests for project_guard.py."""

import subprocess
from pathlib import Path

import pytest

from slopometry.core.project_guard import (
    MultiProjectError,
    detect_multi_project_directory,
    guard_single_project,
)


def _init_git_repo(path: Path) -> None:
    """Initialize a real git repo with an initial commit."""
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "commit.gpgsign", "false"],
        capture_output=True,
        check=True,
    )
    # Need at least one commit for gitignore to work
    (path / ".gitkeep").write_text("")
    subprocess.run(
        ["git", "-C", str(path), "add", ".gitkeep"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True,
        check=True,
    )


class TestDetectMultiProjectDirectory:
    """Tests for detect_multi_project_directory."""

    def test_detect_multi_project_directory__single_project(self, tmp_path: Path) -> None:
        """Single git repo returns one project."""
        (tmp_path / ".git").mkdir()
        projects = detect_multi_project_directory(tmp_path)
        assert projects == ["."]

    def test_detect_multi_project_directory__multiple_projects(self, tmp_path: Path) -> None:
        """Multiple git repos in subdirs returns all."""
        proj1 = tmp_path / "project1"
        proj2 = tmp_path / "project2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / ".git").mkdir()
        (proj2 / ".git").mkdir()

        projects = detect_multi_project_directory(tmp_path)
        assert sorted(projects) == ["project1", "project2"]

    def test_detect_multi_project_directory__respects_max_depth(self, tmp_path: Path) -> None:
        """Projects beyond max_depth are not found."""
        deep = tmp_path / "level1" / "level2" / "level3"
        deep.mkdir(parents=True)
        (deep / ".git").mkdir()

        projects_shallow = detect_multi_project_directory(tmp_path, max_depth=2)
        assert projects_shallow == []

        projects_deep = detect_multi_project_directory(tmp_path, max_depth=3)
        assert projects_deep == ["level1/level2/level3"]

    def test_detect_multi_project_directory__ignores_hidden_dirs(self, tmp_path: Path) -> None:
        """Hidden directories are not searched."""
        hidden = tmp_path / ".hidden_project"
        hidden.mkdir()
        (hidden / ".git").mkdir()

        projects = detect_multi_project_directory(tmp_path)
        assert projects == []

    def test_detect_multi_project_directory__skips_submodules(self, tmp_path: Path) -> None:
        """Git submodules are not counted as separate projects."""
        (tmp_path / ".git").mkdir()
        submodule = tmp_path / "vendor" / "lib"
        submodule.mkdir(parents=True)
        (submodule / ".git").write_text("gitdir: ../../.git/modules/vendor/lib")

        projects = detect_multi_project_directory(tmp_path)
        assert projects == ["."]

    def test_detect_multi_project_directory__nested_single_project(self, tmp_path: Path) -> None:
        """Nested dirs with single git repo."""
        proj = tmp_path / "code" / "myproject"
        proj.mkdir(parents=True)
        (proj / ".git").mkdir()

        projects = detect_multi_project_directory(tmp_path)
        assert projects == ["code/myproject"]

    def test_detect_multi_project_directory__no_git_repos(self, tmp_path: Path) -> None:
        """No git repos returns empty list."""
        (tmp_path / "somefile.txt").write_text("hello")
        projects = detect_multi_project_directory(tmp_path)
        assert projects == []

    def test_detect_multi_project_directory__skips_gitignored_subdirs(self, tmp_path: Path) -> None:
        """Gitignored subdirectories with .git are excluded from detection."""
        # Create a git repo at tmp_path with gitignore rules
        _init_git_repo(tmp_path)
        (tmp_path / ".gitignore").write_text("playground/\nreferences/\n")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", ".gitignore"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "add gitignore"],
            capture_output=True,
            check=True,
        )

        # Create a workspace subdir (no .git) — this is the scan root
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Gitignored subdirs with their own .git repos
        for name in ["playground/sam3", "references/ag-ui", "references/pydantic-ai"]:
            subdir = workspace / name
            subdir.mkdir(parents=True)
            (subdir / ".git").mkdir()

        # All subdirs are gitignored, so nothing should be detected
        projects = detect_multi_project_directory(workspace)
        assert projects == []

    def test_detect_multi_project_directory__counts_non_ignored_subdirs(self, tmp_path: Path) -> None:
        """Non-ignored subdirs with .git are counted even when ignored ones exist."""
        _init_git_repo(tmp_path)
        (tmp_path / ".gitignore").write_text("vendor/\n")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", ".gitignore"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "add gitignore"],
            capture_output=True,
            check=True,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Ignored subdir with .git
        vendor = workspace / "vendor" / "lib"
        vendor.mkdir(parents=True)
        (vendor / ".git").mkdir()

        # Non-ignored subdir with .git — should be detected
        real_project = workspace / "services" / "api"
        real_project.mkdir(parents=True)
        (real_project / ".git").mkdir()

        projects = detect_multi_project_directory(workspace)
        assert projects == ["services/api"]

    def test_detect_multi_project_directory__no_git_root_scans_all(self, tmp_path: Path) -> None:
        """When not inside a git repo, all subdirs are scanned (no filtering)."""
        # No git init on tmp_path — not a git repo
        proj1 = tmp_path / "proj1"
        proj2 = tmp_path / "proj2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / ".git").mkdir()
        (proj2 / ".git").mkdir()

        projects = detect_multi_project_directory(tmp_path)
        assert sorted(projects) == ["proj1", "proj2"]


class TestGuardSingleProject:
    """Tests for guard_single_project."""

    def test_guard_single_project__passes_for_single_project(self, tmp_path: Path) -> None:
        """Single project doesn't raise."""
        (tmp_path / ".git").mkdir()
        guard_single_project(tmp_path)

    def test_guard_single_project__raises_for_multiple_projects(self, tmp_path: Path) -> None:
        """Multiple projects raises MultiProjectError."""
        proj1 = tmp_path / "project1"
        proj2 = tmp_path / "project2"
        proj1.mkdir()
        proj2.mkdir()
        (proj1 / ".git").mkdir()
        (proj2 / ".git").mkdir()

        with pytest.raises(MultiProjectError) as exc_info:
            guard_single_project(tmp_path)

        assert exc_info.value.project_count == 2
        assert "project1" in exc_info.value.projects
        assert "project2" in exc_info.value.projects

    def test_guard_single_project__raises_for_no_git_repo(self, tmp_path: Path) -> None:
        """No git repo raises MultiProjectError with count 0."""
        with pytest.raises(MultiProjectError) as exc_info:
            guard_single_project(tmp_path)

        assert exc_info.value.project_count == 0


class TestMultiProjectError:
    """Tests for MultiProjectError."""

    def test_multi_project_error__message_format(self) -> None:
        """Error message includes count and project names."""
        error = MultiProjectError(3, ["proj1", "proj2", "proj3"])
        assert "3 git repositories" in str(error)
        assert "proj1" in str(error)

    def test_multi_project_error__truncates_long_lists(self) -> None:
        """Long project lists are truncated in message."""
        projects = [f"proj{i}" for i in range(10)]
        error = MultiProjectError(10, projects)
        assert "..." in str(error)
        assert "proj5" not in str(error)

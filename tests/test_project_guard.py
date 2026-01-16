"""Tests for project_guard.py."""

from pathlib import Path

import pytest

from slopometry.core.project_guard import (
    MultiProjectError,
    detect_multi_project_directory,
    guard_single_project,
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

"""Project identification logic."""

import subprocess
from pathlib import Path

import toml

from slopometry.core.models import Project, ProjectSource


class ProjectTracker:
    """Determines the project based on git, pyproject.toml, or working directory."""

    def __init__(self, working_dir: Path | None = None):
        self.working_dir = working_dir or Path.cwd()

    def get_project(self) -> Project | None:
        """Get project identifier.

        Resolution order:
        1. Git remote URL
        2. pyproject.toml standard project name
        """
        if git_project := self._get_project_from_git():
            return git_project

        if pyproject_project := self._get_project_from_pyproject():
            return pyproject_project

        return None

    def _get_project_from_git(self) -> Project | None:
        """Get project from git remote origin URL."""
        try:
            check_git = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if check_git.returncode != 0 or not check_git.stdout.strip() == "true":
                return None

            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Project(name=result.stdout.strip(), source=ProjectSource.GIT)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass

        return None

    def _get_project_from_pyproject(self) -> Project | None:
        """Get project name from pyproject.toml."""
        pyproject_path = self.working_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            return None

        try:
            data = toml.load(pyproject_path)
            # Look for the standard PEP 621 project name
            project_name = data.get("project", {}).get("name")
            if project_name and isinstance(project_name, str):
                return Project(name=project_name, source=ProjectSource.PYPROJECT)
        except (toml.TomlDecodeError, OSError, KeyError, TypeError):
            # Invalid TOML, file access error, or unexpected structure
            pass

        return None

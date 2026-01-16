"""Guard against running analysis in directories with multiple projects."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiProjectError(Exception):
    """Raised when analysis is attempted in a directory with multiple projects."""

    def __init__(self, project_count: int, projects: list[str]):
        self.project_count = project_count
        self.projects = projects
        super().__init__(
            f"Directory contains {project_count} git repositories. "
            f"Slopometry analyzes single projects. Run from within a specific project directory.\n"
            f"Found: {', '.join(projects[:5])}{'...' if len(projects) > 5 else ''}"
        )


def _is_git_submodule(path: Path, root: Path) -> bool:
    """Check if a .git path is a submodule (file pointing to parent's .git/modules)."""
    git_path = path / ".git"
    if not git_path.exists():
        return False
    if git_path.is_file():
        return True
    if path == root:
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "config", "--file", ".gitmodules", "--get-regexp", "path"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            rel_path = str(path.relative_to(root))
            for line in result.stdout.strip().split("\n"):
                if line and rel_path in line:
                    return True
    except Exception as e:
        logger.debug(f"Failed to check .gitmodules for submodule detection: {e}")
    return False


def detect_multi_project_directory(
    root: Path,
    max_depth: int = 2,
    max_projects: int = 1,
) -> list[str]:
    """Detect if directory contains multiple git repositories.

    Args:
        root: Directory to check
        max_depth: Maximum directory depth to search (default 2)
        max_projects: Maximum allowed projects before raising error (default 1)

    Returns:
        List of project paths found (relative to root)
    """
    root = root.resolve()
    projects: list[str] = []

    def scan_dir(path: Path, depth: int) -> None:
        if depth > max_depth:
            return

        git_dir = path / ".git"
        if git_dir.exists():
            if not _is_git_submodule(path, root):
                rel_path = str(path.relative_to(root)) if path != root else "."
                projects.append(rel_path)
            return

        try:
            for child in path.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    scan_dir(child, depth + 1)
        except PermissionError as e:
            logger.debug(f"Permission denied scanning directory {path}: {e}")

    scan_dir(root, 0)
    return projects


def guard_single_project(root: Path, max_depth: int = 2) -> None:
    """Raise MultiProjectError if directory contains multiple projects.

    Args:
        root: Directory to check
        max_depth: Maximum directory depth to search

    Raises:
        MultiProjectError: If multiple git repositories found
    """
    projects = detect_multi_project_directory(root, max_depth=max_depth, max_projects=1)

    if len(projects) > 1:
        raise MultiProjectError(len(projects), projects)

    if len(projects) == 0:
        git_dir = root / ".git"
        if not git_dir.exists():
            raise MultiProjectError(0, [])

"""Guard against running analysis in directories with multiple projects."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCKED_DIRECTORIES = {
    Path.home(),
    Path("/"),
    Path("/usr"),
    Path("/opt"),
    Path("/var"),
    Path("/tmp"),
}


class UnsafeDirectoryError(Exception):
    """Raised when analysis is attempted in a blocked directory like home."""

    def __init__(self, directory: Path, reason: str):
        self.directory = directory
        self.reason = reason
        super().__init__(f"Refusing to analyze {directory}: {reason}")


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


def _get_git_root(path: Path) -> Path | None:
    """Find the enclosing git repo root for a path.

    Returns None if the path is not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception as e:
        logger.debug(f"Failed to find git root for {path}: {e}")
    return None


def _is_gitignored_batch(paths: list[Path], git_root: Path) -> set[Path]:
    """Batch-check which paths are gitignored.

    Uses `git check-ignore` to determine which of the given paths are
    covered by gitignore rules. Returns the set of ignored paths.
    Fails open: if the command fails, returns an empty set.
    """
    if not paths:
        return set()
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "check-ignore", *[str(p) for p in paths]],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # git check-ignore exits 0 if any path is ignored, 1 if none are ignored
        if result.returncode in (0, 1):
            ignored = set()
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line:
                    ignored.add(Path(line).resolve())
            return ignored
    except Exception as e:
        logger.debug(f"Failed to check gitignore for paths: {e}")
    return set()


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
    git_root = _get_git_root(root)

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
            children = [child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")]
        except PermissionError as e:
            logger.debug(f"Permission denied scanning directory {path}: {e}")
            return

        if git_root and children:
            ignored = _is_gitignored_batch(children, git_root)
            children = [c for c in children if c.resolve() not in ignored]

        for child in children:
            scan_dir(child, depth + 1)

    scan_dir(root, 0)
    return projects


def _is_blocked_directory(path: Path) -> str | None:
    """Check if path is a blocked directory.

    Returns:
        Reason string if blocked, None if allowed
    """
    resolved = path.resolve()

    for blocked in BLOCKED_DIRECTORIES:
        try:
            if resolved == blocked.resolve():
                return f"'{resolved}' is a system/home directory"
        except (OSError, ValueError):
            continue

    xdg_data = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    xdg_config = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")

    sensitive_paths = [
        Path(xdg_data),
        Path(xdg_cache),
        Path(xdg_config),
        Path.home() / ".local",
    ]

    for sensitive in sensitive_paths:
        try:
            if resolved == sensitive.resolve():
                return f"'{resolved}' is a user data/cache directory"
        except (OSError, ValueError):
            continue

    return None


def guard_single_project(root: Path, max_depth: int = 2) -> None:
    """Raise error if directory is unsafe or contains multiple projects.

    Args:
        root: Directory to check
        max_depth: Maximum directory depth to search

    Raises:
        UnsafeDirectoryError: If directory is blocked (home, /, etc.)
        MultiProjectError: If multiple git repositories found
    """
    if reason := _is_blocked_directory(root):
        raise UnsafeDirectoryError(root, reason)

    projects = detect_multi_project_directory(root, max_depth=max_depth, max_projects=1)

    if len(projects) > 1:
        raise MultiProjectError(len(projects), projects)

    if len(projects) == 0:
        git_dir = root / ".git"
        if not git_dir.exists():
            raise MultiProjectError(0, [])

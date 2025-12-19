"""Working tree state calculation for intelligent caching."""

import hashlib
import subprocess
from pathlib import Path

from slopometry.core.git_tracker import GitTracker


class WorkingTreeStateCalculator:
    """Calculates unique identifiers for working tree states including uncommitted changes."""

    def __init__(self, working_directory: Path | str):
        """Initialize calculator for a specific directory.

        Args:
            working_directory: Directory to analyze for working tree state
        """
        self.working_directory = Path(working_directory).resolve()

    def calculate_working_tree_hash(self, commit_sha: str) -> str:
        """Calculate a hash representing the current working tree state.

        Args:
            commit_sha: Current git commit SHA as base

        Returns:
            Unique hash representing current working tree state
        """
        python_files = self._get_python_files()

        hash_components = [commit_sha]

        for py_file in sorted(python_files):
            try:
                mtime = py_file.stat().st_mtime
                rel_path = py_file.relative_to(self.working_directory)
                hash_components.append(f"{rel_path}:{mtime}")
            except (OSError, ValueError):
                continue

        hash_components.append(f"file_count:{len(python_files)}")

        combined = "|".join(hash_components)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in the working directory that would be analyzed.

        Returns:
            List of Python file paths
        """
        tracker = GitTracker(self.working_directory)
        return tracker.get_tracked_python_files()

    def get_current_commit_sha(self) -> str | None:
        """Get current git commit SHA for the working directory.

        Returns:
            Current commit SHA or None if not a git repository
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass
        return None

    def has_uncommitted_changes(self) -> bool:
        """Check if the working directory has uncommitted changes.

        Returns:
            True if there are uncommitted changes
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return bool(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass
        return False

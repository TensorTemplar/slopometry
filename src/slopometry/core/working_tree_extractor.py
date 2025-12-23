"""Working tree extraction for complexity analysis."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from slopometry.core.git_tracker import GitTracker


class WorkingTreeExtractor:
    """Extracts working tree state (uncommitted changes) for complexity analysis."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def get_changed_python_files(self) -> list[str]:
        """Get list of changed Python files (staged and unstaged).

        Returns:
            List of Python file paths that have uncommitted changes
        """
        files: set[str] = set()
        files.update(self._get_staged_files())
        files.update(self._get_unstaged_files())
        return sorted([f for f in files if f.endswith(".py")])

    def _get_staged_files(self) -> list[str]:
        """Get list of staged files (added, copied, modified, renamed)."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        all_files = result.stdout.strip().split("\n")
        return [f for f in all_files if f]

    def _get_unstaged_files(self) -> list[str]:
        """Get list of unstaged modified files in tracked files."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        all_files = result.stdout.strip().split("\n")
        return [f for f in all_files if f]

    def has_uncommitted_changes(self) -> bool:
        """Check if there are any uncommitted changes (staged or unstaged)."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def extract_working_state(self) -> Path | None:
        """Extract current working tree state to temp directory.

        Simply copies current Python files - no stashing needed since we want
        the full working state including both staged and unstaged changes.

        Returns the path to a temporary directory containing all Python files
        from the current working tree.

        Returns None if there are no uncommitted changes.
        """
        if not self.has_uncommitted_changes():
            return None

        temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_working_"))

        try:
            self._copy_python_files_to_temp(temp_dir)
            return temp_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def _copy_python_files_to_temp(self, temp_dir: Path) -> None:
        """Copy all Python files from the working directory to temp dir."""
        tracker = GitTracker(self.repo_path)
        tracked_files = tracker.get_tracked_python_files()

        for py_file in tracked_files:
            if not py_file.exists():
                continue

            try:
                relative_path = py_file.relative_to(self.repo_path)
            except ValueError:
                continue

            dest_path = temp_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dest_path)

    def get_head_commit_sha(self) -> str | None:
        """Get the current HEAD commit SHA."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

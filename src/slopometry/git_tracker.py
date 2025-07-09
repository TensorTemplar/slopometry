"""Git state tracking for Claude Code sessions."""

import subprocess
import tempfile
from pathlib import Path

from .models import GitState


class GitTracker:
    """Tracks git repository state and commit counts."""

    def __init__(self, working_dir: Path | None = None):
        self.working_dir = working_dir or Path.cwd()

    def get_git_state(self) -> GitState:
        """Get current git repository state."""
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return GitState(is_git_repo=False)

            # Get commit count
            commit_count = self._get_commit_count()

            # Get current branch
            current_branch = self._get_current_branch()

            # Check for uncommitted changes
            has_uncommitted_changes = self._has_uncommitted_changes()

            return GitState(
                is_git_repo=True,
                commit_count=commit_count,
                current_branch=current_branch,
                has_uncommitted_changes=has_uncommitted_changes,
            )

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return GitState(is_git_repo=False)

    def _get_commit_count(self) -> int:
        """Get total number of commits in the repository."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return int(result.stdout.strip())

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, OSError):
            pass

        return 0

    def _get_current_branch(self) -> str | None:
        """Get current branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                branch = result.stdout.strip()
                return branch if branch else None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return None

    def _has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return bool(result.stdout.strip())

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return False

    def calculate_commits_made(self, initial_state: GitState, final_state: GitState) -> int:
        """Calculate commits made between two git states."""
        if not initial_state.is_git_repo or not final_state.is_git_repo:
            return 0

        return max(0, final_state.commit_count - initial_state.commit_count)

    def get_python_files_from_commit(self, commit_ref: str = "HEAD~1") -> list[str]:
        """Get list of Python files that existed in a specific commit.

        Args:
            commit_ref: Git commit reference (default: HEAD~1 for previous commit)

        Returns:
            List of relative paths to Python files
        """
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit_ref],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                all_files = result.stdout.strip().split("\n")
                # Filter for Python files
                python_files = [f for f in all_files if f.endswith(".py")]
                return python_files

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return []

    def extract_files_from_commit(self, commit_ref: str = "HEAD~1") -> Path | None:
        """Extract Python files from a specific commit to a temporary directory.

        Args:
            commit_ref: Git commit reference (default: HEAD~1 for previous commit)

        Returns:
            Path to temporary directory containing extracted files, or None if failed
        """
        try:
            # Get list of Python files in the commit
            python_files = self.get_python_files_from_commit(commit_ref)
            if not python_files:
                return None

            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_baseline_"))

            # Extract each Python file
            for file_path in python_files:
                try:
                    # Get file content from commit
                    result = subprocess.run(
                        ["git", "show", f"{commit_ref}:{file_path}"],
                        cwd=self.working_dir,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if result.returncode == 0:
                        # Create directory structure in temp dir
                        target_file = temp_dir / file_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)

                        # Write file content
                        target_file.write_text(result.stdout)

                except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
                    # Skip files that can't be extracted
                    continue

            return temp_dir

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return None

    def has_previous_commit(self) -> bool:
        """Check if there's a previous commit to compare against."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD~1"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

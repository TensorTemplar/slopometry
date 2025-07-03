"""Git state tracking for Claude Code sessions."""

import subprocess
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

"""Git state tracking for Claude Code sessions."""

import subprocess
import tarfile
import tempfile
from pathlib import Path

from slopometry.core.models import GitState


class GitTracker:
    """Tracks git repository state and commit counts."""

    def __init__(self, working_dir: Path | None = None):
        self.working_dir = working_dir or Path.cwd()

    def get_git_state(self) -> GitState:
        """Get current git repository state."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return GitState(is_git_repo=False)

            commit_count = self._get_commit_count()

            current_branch = self._get_current_branch()

            has_uncommitted_changes = self._has_uncommitted_changes()

            commit_sha = self._get_current_commit_sha()

            return GitState(
                is_git_repo=True,
                commit_count=commit_count,
                current_branch=current_branch,
                has_uncommitted_changes=has_uncommitted_changes,
                commit_sha=commit_sha,
            )

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return GitState(is_git_repo=False)

    def _get_commit_count(self) -> int:
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

    def _get_current_commit_sha(self) -> str | None:
        """Get current git commit SHA.

        Returns:
            Current commit SHA or None if failed
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return None

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
                python_files = [f for f in all_files if f.endswith(".py")]
                return python_files

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return []

    def get_tracked_python_files(self) -> list[Path]:
        """Get list of Python files tracked by git or not ignored (if untracked).

        Uses git ls-files if available, otherwise falls back to rglob with exclusion.

        Returns:
            List of Path objects for Python files
        """

        try:
            cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            files = []
            for line in result.stdout.splitlines():
                if line.endswith(".py"):
                    files.append(self.working_dir / line)
            return files

        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        files = []
        ignored_dirs = {
            ".venv",
            "venv",
            "env",
            ".env",
            ".git",
            ".idea",
            ".vscode",
            "__pycache__",
            "node_modules",
            "site-packages",
            "dist",
            "build",
        }

        for file_path in self.working_dir.rglob("*.py"):
            parts = file_path.relative_to(self.working_dir).parts
            if any(part in ignored_dirs for part in parts):
                continue
            files.append(file_path)

        return files

    def extract_files_from_commit(self, commit_ref: str = "HEAD~1") -> Path | None:
        """Extract Python files from a specific commit to a temporary directory.

        Uses git archive for efficient extraction of entire tree.

        Args:
            commit_ref: Git commit reference (default: HEAD~1 for previous commit)

        Returns:
            Path to temporary directory containing extracted files, or None if failed
        """
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_baseline_"))

            # Use git archive to extract entire tree at once
            result = subprocess.run(
                ["git", "archive", "--format=tar", commit_ref],
                cwd=self.working_dir,
                capture_output=True,
                timeout=60,
            )

            if result.returncode != 0:
                return None

            from io import BytesIO

            tar_data = BytesIO(result.stdout)
            with tarfile.open(fileobj=tar_data, mode="r") as tar:
                python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
                if not python_members:
                    return None

                tar.extractall(path=temp_dir, members=python_members)

            return temp_dir

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, tarfile.TarError):
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

    def get_merge_base_with_main(self) -> str | None:
        """Get the merge-base commit where current branch diverged from main/master.

        Returns:
            Commit SHA of the merge-base, or None if not found
        """
        try:
            main_branch = None
            for branch_name in ["main", "master", "origin/main", "origin/master"]:
                result = subprocess.run(
                    ["git", "rev-parse", "--verify", branch_name],
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    main_branch = branch_name
                    break

            if not main_branch:
                return None

            result = subprocess.run(
                ["git", "merge-base", "HEAD", main_branch],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            pass

        return None

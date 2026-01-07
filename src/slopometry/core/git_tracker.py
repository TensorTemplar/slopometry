"""Git state tracking for Claude Code sessions."""

import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from slopometry.core.models import GitState


class GitOperationError(Exception):
    """Raised when a git operation fails unexpectedly.

    This exception indicates that a git command failed in a context where
    failure should not be silently ignored. Callers should catch this and
    either propagate it or provide meaningful error handling.
    """

    pass


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

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, GitOperationError):
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

            raise GitOperationError(f"git rev-list failed: {result.stderr.strip()}")

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git rev-list timed out: {e}") from e
        except ValueError as e:
            raise GitOperationError(f"Invalid commit count output: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"git rev-list failed: {e}") from e

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

            raise GitOperationError(f"git status failed: {result.stderr.strip()}")

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git status timed out: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"git status failed: {e}") from e

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
        """Extract Python files and coverage.xml from a specific commit to a temporary directory.

        Uses git archive for efficient extraction of entire tree.

        Args:
            commit_ref: Git commit reference (default: HEAD~1 for previous commit)

        Returns:
            Path to temporary directory containing extracted files, or None if no Python files

        Raises:
            GitOperationError: If git archive fails or tar extraction fails
        """
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_baseline_"))

            result = subprocess.run(
                ["git", "archive", "--format=tar", commit_ref],
                cwd=self.working_dir,
                capture_output=True,
                timeout=60,
            )

            if result.returncode != 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise GitOperationError(f"git archive failed for {commit_ref}: {result.stderr.decode().strip()}")

            from io import BytesIO

            tar_data = BytesIO(result.stdout)
            with tarfile.open(fileobj=tar_data, mode="r") as tar:
                python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
                coverage_members = [m for m in tar.getmembers() if m.name == "coverage.xml"]

                members_to_extract = python_members + coverage_members
                if not python_members:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None

                tar.extractall(path=temp_dir, members=members_to_extract, filter="data")

            return temp_dir

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git archive timed out for {commit_ref}: {e}") from e
        except tarfile.TarError as e:
            raise GitOperationError(f"Failed to extract tar for {commit_ref}: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"git archive failed for {commit_ref}: {e}") from e

    @contextmanager
    def extract_files_from_commit_ctx(self, commit_ref: str = "HEAD~1") -> Iterator[Path | None]:
        """Extract Python files from a commit to a temporary directory with auto-cleanup.

        This is the preferred method over extract_files_from_commit as it ensures
        the temporary directory is automatically cleaned up when the context exits.

        Args:
            commit_ref: Git commit reference (default: HEAD~1 for previous commit)

        Yields:
            Path to temporary directory containing extracted files, or None if no Python files

        Raises:
            GitOperationError: If git archive fails or tar extraction fails
        """
        with tempfile.TemporaryDirectory(prefix="slopometry_baseline_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            try:
                result = subprocess.run(
                    ["git", "archive", "--format=tar", commit_ref],
                    cwd=self.working_dir,
                    capture_output=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    raise GitOperationError(f"git archive failed for {commit_ref}: {result.stderr.decode().strip()}")

                from io import BytesIO

                tar_data = BytesIO(result.stdout)
                with tarfile.open(fileobj=tar_data, mode="r") as tar:
                    python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
                    coverage_members = [m for m in tar.getmembers() if m.name == "coverage.xml"]

                    members_to_extract = python_members + coverage_members
                    if not python_members:
                        yield None
                        return

                    tar.extractall(path=temp_dir, members=members_to_extract, filter="data")

                yield temp_dir

            except subprocess.TimeoutExpired as e:
                raise GitOperationError(f"git archive timed out for {commit_ref}: {e}") from e
            except tarfile.TarError as e:
                raise GitOperationError(f"Failed to extract tar for {commit_ref}: {e}") from e
            except (subprocess.SubprocessError, OSError) as e:
                raise GitOperationError(f"git archive failed for {commit_ref}: {e}") from e

    def get_changed_python_files(self, parent_sha: str, child_sha: str) -> list[str]:
        """Get list of Python files that changed between two commits.

        Args:
            parent_sha: Parent commit SHA
            child_sha: Child commit SHA

        Returns:
            List of changed Python file paths (relative to repo root)

        Raises:
            GitOperationError: If git diff fails
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", parent_sha, child_sha, "--", "*.py"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise GitOperationError(f"git diff failed for {parent_sha}..{child_sha}: {result.stderr.strip()}")

            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git diff timed out for {parent_sha}..{child_sha}: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"git diff failed for {parent_sha}..{child_sha}: {e}") from e

    def extract_specific_files_from_commit(self, commit_ref: str, file_paths: list[str]) -> Path | None:
        """Extract specific files from a commit to a temporary directory.

        Args:
            commit_ref: Git commit reference
            file_paths: List of file paths to extract

        Returns:
            Path to temporary directory containing extracted files, or None if no files to extract

        Raises:
            GitOperationError: If extraction fails completely
        """
        if not file_paths:
            return None

        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_delta_"))
            failed_files: list[str] = []

            for file_path in file_paths:
                try:
                    result = subprocess.run(
                        ["git", "show", f"{commit_ref}:{file_path}"],
                        cwd=self.working_dir,
                        capture_output=True,
                        timeout=10,
                    )

                    if result.returncode == 0:
                        dest_path = temp_dir / file_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        dest_path.write_bytes(result.stdout)
                    else:
                        failed_files.append(file_path)
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    failed_files.append(file_path)

            # Don't error on files that don't exist in this commit
            # (e.g., newly added files when extracting from parent commit)
            if not any(temp_dir.rglob("*.py")):
                if failed_files and len(failed_files) == len(file_paths):
                    raise GitOperationError(
                        f"Failed to extract any files from {commit_ref}. "
                        f"These files may not exist in this commit. Failed: {failed_files}"
                    )
                return None

            return temp_dir

        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"Failed to extract files from {commit_ref}: {e}") from e

    def has_previous_commit(self) -> bool:
        """Check if there's a previous commit to compare against.

        Returns:
            True if HEAD~1 exists, False if this is the first commit

        Raises:
            GitOperationError: If git command fails unexpectedly
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD~1"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git rev-parse timed out: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise GitOperationError(f"git rev-parse failed: {e}") from e

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

"""Git state tracking for Claude Code sessions."""

import logging
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from slopometry.core.models.hook import GitState

logger = logging.getLogger(__name__)


class GitOperationError(Exception):
    """Raised when a git operation fails unexpectedly.

    This exception indicates that a git command failed in a context where
    failure should not be silently ignored. Callers should catch this and
    either propagate it or provide meaningful error handling.
    """

    pass


def get_submodule_prefixes(working_dir: Path) -> tuple[str, ...]:
    """Get submodule paths (relative to working_dir) as a tuple of prefixes.

    Reads .gitmodules in the repository root to find declared submodule paths.
    Returned strings end with a forward slash so callers can use str.startswith
    to test whether an arbitrary relative path lives inside a submodule.

    Returns an empty tuple only when no .gitmodules file exists. If .gitmodules
    is present but cannot be parsed, raises GitOperationError — silently
    returning () would disable submodule filtering and re-introduce the exact
    cache-invalidation bug this helper exists to prevent.
    """
    gitmodules = working_dir / ".gitmodules"
    if not gitmodules.is_file():
        return ()
    try:
        result = subprocess.run(
            ["git", "config", "--file", str(gitmodules), "--get-regexp", r"^submodule\..*\.path$"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as e:
        raise GitOperationError(f"git config timed out reading {gitmodules}: {e}") from e
    except (subprocess.SubprocessError, OSError) as e:
        raise GitOperationError(f"git config failed reading {gitmodules}: {e}") from e

    # returncode == 1 is git config's standard "no matching keys" result — a
    # .gitmodules with no submodule.*.path entries is malformed but not fatal;
    # treat it as no submodules. Any other non-zero return is a real failure.
    if result.returncode == 1:
        return ()
    if result.returncode != 0:
        raise GitOperationError(
            f"git config failed reading {gitmodules} (exit {result.returncode}): {result.stderr.strip()}"
        )

    prefixes: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[1]:
            prefix = parts[1].rstrip("/") + "/"
            prefixes.append(prefix)
    return tuple(prefixes)


def path_in_submodule(rel_path: str, submodule_prefixes: tuple[str, ...]) -> bool:
    """Return True if rel_path lives inside any declared submodule."""
    normalized = rel_path.replace("\\", "/").lstrip("./")
    return any(normalized.startswith(prefix) for prefix in submodule_prefixes)


class GitTracker:
    """Tracks git repository state and commit counts."""

    def __init__(self, working_dir: Path | None = None):
        self.working_dir = working_dir or Path.cwd()
        self._submodule_prefixes: tuple[str, ...] | None = None

    def submodule_prefixes(self) -> tuple[str, ...]:
        """Cached accessor for declared submodule paths as relative prefixes."""
        if self._submodule_prefixes is None:
            self._submodule_prefixes = get_submodule_prefixes(self.working_dir)
        return self._submodule_prefixes

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

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to get current commit SHA: {e}")

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

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to get Python files from commit {commit_ref}: {e}")

        return []

    def get_tracked_python_files(self) -> list[Path]:
        """Get list of Python files tracked by git or not ignored (if untracked).

        For non-git directories (e.g., temp extraction dirs), falls back to
        finding all Python files while excluding common virtual env directories.

        Returns:
            List of Path objects for Python files

        Raises:
            GitOperationError: If inside a git repo but git command fails
        """
        try:
            cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                submodule_prefixes = self.submodule_prefixes()
                files = []
                for line in result.stdout.splitlines():
                    if line.endswith(".py") and not path_in_submodule(line, submodule_prefixes):
                        files.append(self.working_dir / line)
                return files

            # Check if this is a "not a git repo" error vs actual failure
            stderr = result.stderr.strip().lower()
            if "not a git repository" in stderr:
                # Not a git repo - fall back to finding Python files directly
                return self._find_python_files_fallback()

            # Actual git failure in a git repo
            raise GitOperationError(f"git ls-files failed: {result.stderr.strip()}")

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git ls-files timed out: {e}") from e
        except FileNotFoundError as e:
            raise GitOperationError(f"git not found - is git installed? {e}") from e
        except subprocess.SubprocessError as e:
            raise GitOperationError(f"git ls-files failed: {e}") from e

    def has_analyzable_source_files(self) -> bool:
        """Check if the working directory contains any Python or Rust source files.

        Uses git ls-files for git repos (fast, respects .gitignore). For non-git
        directories, returns False to avoid scanning massive directory trees.

        Returns:
            True if at least one .py or .rs file is found via git ls-files.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False

            submodule_prefixes = self.submodule_prefixes()
            for line in result.stdout.splitlines():
                if not (line.endswith(".py") or line.endswith(".rs")):
                    continue
                if path_in_submodule(line, submodule_prefixes):
                    continue
                return True
            return False

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    def _is_multi_repo_parent(self) -> bool:
        """Check if working_dir is a parent directory containing multiple git repos.

        Detects directories like droidcraft_branches/ that contain sibling repos
        (IsaacSim/.git, moelite/.git, etc.) to avoid catastrophic rglob scans.

        Returns:
            True if any immediate child directory contains a .git directory.
        """
        return any(self.working_dir.glob("*/.git"))

    def _find_python_files_fallback(self) -> list[Path]:
        """Find Python files without git (for non-git directories like temp extractions).

        Detects multi-repo parent directories (e.g. droidcraft_branches/) by checking
        for nested .git dirs in immediate children. If found, returns empty list to avoid
        scanning hundreds of thousands of files across sibling repos.

        Only uses rglob for genuine non-git directories like temp extraction dirs.
        """
        if self._is_multi_repo_parent():
            return []

        ignored_dirs = {
            ".venv",
            "venv",
            "env",
            ".env",
            ".git",
            "__pycache__",
            "node_modules",
            "site-packages",
            "dist",
            "build",
        }

        submodule_prefixes = self.submodule_prefixes()
        files = []
        for file_path in self.working_dir.rglob("*.py"):
            rel = file_path.relative_to(self.working_dir)
            if any(part in ignored_dirs for part in rel.parts):
                continue
            if path_in_submodule(rel.as_posix(), submodule_prefixes):
                continue
            files.append(file_path)

        return files

    def get_tracked_rust_files(self) -> list[Path]:
        """Get list of Rust files tracked by git or not ignored (if untracked).

        For non-git directories, falls back to finding all Rust files while
        excluding common build directories.

        Returns:
            List of Path objects for Rust files

        Raises:
            GitOperationError: If inside a git repo but git command fails
        """
        try:
            cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                submodule_prefixes = self.submodule_prefixes()
                files = []
                for line in result.stdout.splitlines():
                    if line.endswith(".rs") and not path_in_submodule(line, submodule_prefixes):
                        files.append(self.working_dir / line)
                return files

            # Check if this is a "not a git repo" error vs actual failure
            stderr = result.stderr.strip().lower()
            if "not a git repository" in stderr:
                # Not a git repo - fall back to finding Rust files directly
                return self._find_rust_files_fallback()

            # Actual git failure in a git repo
            raise GitOperationError(f"git ls-files failed: {result.stderr.strip()}")

        except subprocess.TimeoutExpired as e:
            raise GitOperationError(f"git ls-files timed out: {e}") from e
        except FileNotFoundError as e:
            raise GitOperationError(f"git not found - is git installed? {e}") from e
        except subprocess.SubprocessError as e:
            raise GitOperationError(f"git ls-files failed: {e}") from e

    def _find_rust_files_fallback(self) -> list[Path]:
        """Find Rust files without git (for non-git directories).

        Detects multi-repo parent directories and returns empty list to avoid
        scanning sibling repos. See _find_python_files_fallback for details.
        """
        if self._is_multi_repo_parent():
            return []

        ignored_dirs = {
            "target",  # Cargo build output
            ".cargo",  # Cargo cache
            ".git",
            "node_modules",
        }

        submodule_prefixes = self.submodule_prefixes()
        files = []
        for file_path in self.working_dir.rglob("*.rs"):
            rel = file_path.relative_to(self.working_dir)
            if any(part in ignored_dirs for part in rel.parts):
                continue
            if path_in_submodule(rel.as_posix(), submodule_prefixes):
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

        Uses git archive with pathspec for batch extraction (single subprocess call)
        instead of per-file git show calls.

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

        temp_dir: Path | None = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="slopometry_delta_"))

            # Use git archive with pathspec to extract only the specified files
            # This is O(1) subprocess calls instead of O(n) with git show
            result = subprocess.run(
                ["git", "archive", "--format=tar", commit_ref, "--"] + file_paths,
                cwd=self.working_dir,
                capture_output=True,
                timeout=60,
            )

            if result.returncode != 0:
                # git archive fails if none of the files exist in this commit
                # This is normal for newly added files when extracting from parent
                stderr = result.stderr.decode().strip()
                if "pathspec" in stderr.lower() or "not in" in stderr.lower():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None
                raise GitOperationError(f"git archive failed for {commit_ref}: {stderr}")

            tar_data = BytesIO(result.stdout)
            try:
                with tarfile.open(fileobj=tar_data, mode="r") as tar:
                    python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
                    if not python_members:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return None
                    tar.extractall(path=temp_dir, members=python_members, filter="data")
            except tarfile.TarError as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise GitOperationError(f"Failed to extract tar for {commit_ref}: {e}") from e

            if not any(temp_dir.rglob("*.py")):
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            return temp_dir

        except subprocess.TimeoutExpired as e:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise GitOperationError(f"git archive timed out for {commit_ref}: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
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

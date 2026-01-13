"""Working tree state calculation for intelligent caching."""

import hashlib
import subprocess
from pathlib import Path

from slopometry.core.git_tracker import GitTracker
from slopometry.core.language_config import (
    get_combined_git_patterns,
    should_ignore_path,
)
from slopometry.core.models import ProjectLanguage


class WorkingTreeStateCalculator:
    """Calculates unique identifiers for working tree states including uncommitted changes."""

    def __init__(
        self,
        working_directory: Path | str,
        languages: list[ProjectLanguage] | None = None,
    ):
        """Initialize calculator for a specific directory.

        Args:
            working_directory: Directory to analyze for working tree state
            languages: Languages to consider, or None for all supported languages
        """
        self.working_directory = Path(working_directory).resolve()
        self.languages = languages

    def calculate_working_tree_hash(self, commit_sha: str) -> str:
        """Calculate a hash representing the current working tree state.

        Uses two-tier detection:
        1. Get list of potentially modified source files from git (fast)
        2. For each file, use content hash to verify actual changes

        Uses BLAKE2b for hashing - fast on both arm64 and amd64, built into Python.

        Args:
            commit_sha: Current git commit SHA as base

        Returns:
            Unique hash representing current working tree state
        """
        modified_source_files = self._get_modified_source_files_from_git()

        hash_components = [commit_sha]

        for source_file in sorted(modified_source_files):
            try:
                # Use content hash, not mtime - filters out touch/checkout false positives
                # BLAKE2b with digest_size=8 gives 16 hex chars, fast on arm64/amd64
                content_hash = hashlib.blake2b(source_file.read_bytes(), digest_size=8).hexdigest()
                rel_path = source_file.relative_to(self.working_directory)
                hash_components.append(f"{rel_path}:{content_hash}")
            except (OSError, ValueError):
                continue

        hash_components.append(f"file_count:{len(modified_source_files)}")

        combined = "|".join(hash_components)
        return hashlib.blake2b(combined.encode("utf-8"), digest_size=8).hexdigest()

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in the working directory that would be analyzed.

        Returns:
            List of Python file paths
        """
        tracker = GitTracker(self.working_directory)
        return tracker.get_tracked_python_files()

    def _get_modified_source_files_from_git(self) -> list[Path]:
        """Get source files with uncommitted changes using git diff (fast first-pass).

        Uses git diff to get modified source files (staged + unstaged) for
        configured languages. Filters out files in ignored directories
        (build artifacts, caches, etc.).

        This is tier 1 of the two-tier change detection - fast but may include
        files that only have mtime changes (which tier 2 content hash filters out).

        Returns:
            List of Path objects for source files that git reports as modified
        """
        git_patterns = get_combined_git_patterns(self.languages)
        files: set[Path] = set()

        for pattern in git_patterns:
            try:
                # Unstaged changes
                result1 = subprocess.run(
                    ["git", "diff", "--name-only", "--", pattern],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # Staged changes
                result2 = subprocess.run(
                    ["git", "diff", "--cached", "--name-only", "--", pattern],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                for line in result1.stdout.splitlines() + result2.stdout.splitlines():
                    if line.strip():
                        rel_path = line.strip()
                        # Filter out files in ignored directories (build artifacts, caches)
                        if not should_ignore_path(rel_path, self.languages):
                            files.add(self.working_directory / rel_path)
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
                continue

        return list(files)

    def _get_modified_python_files_from_git(self) -> list[Path]:
        """Get Python files with uncommitted changes.

        Convenience wrapper for Python-specific detection.
        Delegates to _get_modified_source_files_from_git with Python language.

        Returns:
            List of Path objects for Python files that git reports as modified
        """
        # Use stored languages or default to Python-only for backwards compatibility
        return self._get_modified_source_files_from_git()

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

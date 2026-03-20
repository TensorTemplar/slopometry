"""Working tree state calculation for intelligent caching."""

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

from slopometry.core.git_tracker import GitTracker
from slopometry.core.language_config import (
    get_combined_git_patterns,
    is_source_file,
    should_ignore_path,
)
from slopometry.core.models.hook import ProjectLanguage


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

    def get_source_file_content_hashes(self) -> dict[str, str]:
        """Get per-file content hashes for all modified source files.

        Returns a mapping of relative path to BLAKE2b content hash for each
        source file that git reports as modified (staged + unstaged).
        Filters by language config and ignore patterns.

        Returns:
            Dict mapping relative path strings to 16-char hex BLAKE2b hashes
        """
        modified_source_files = self._get_modified_source_files_from_git()
        result: dict[str, str] = {}

        for source_file in sorted(modified_source_files):
            try:
                content_hash = hashlib.blake2b(source_file.read_bytes(), digest_size=8).hexdigest()
                rel_path = str(source_file.relative_to(self.working_directory))
                result[rel_path] = content_hash
            except (OSError, ValueError):
                continue

        return result

    def get_files_changed_since(self, previous_hashes: dict[str, str]) -> set[str]:
        """Compute source files that changed since a previous state.

        Compares current modified source files against a previous set of
        content hashes (e.g., from the last time the hook fired). A file
        is considered changed if it:
        - Is currently modified AND was not modified previously (new dirty file)
        - Is currently modified AND has a different content hash than previously

        Args:
            previous_hashes: File content hashes from the previous state

        Returns:
            Set of relative path strings for files that changed
        """
        current_hashes = self.get_source_file_content_hashes()
        changed: set[str] = set()

        for rel_path, current_hash in current_hashes.items():
            previous_hash = previous_hashes.get(rel_path)
            if previous_hash is None or previous_hash != current_hash:
                changed.add(rel_path)

        return changed

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
        file_hashes = self.get_source_file_content_hashes()

        hash_components = [commit_sha]
        for rel_path in sorted(file_hashes):
            hash_components.append(f"{rel_path}:{file_hashes[rel_path]}")
        hash_components.append(f"file_count:{len(file_hashes)}")

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
                        # Filter out non-source files and ignored directories
                        if is_source_file(rel_path, self.languages) and not should_ignore_path(
                            rel_path, self.languages
                        ):
                            files.add(self.working_directory / rel_path)
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
                continue

        return list(files)

    def get_modified_source_file_paths(self) -> set[str]:
        """Get relative paths of modified source files, filtered by ignore patterns.

        Combines git diff detection with should_ignore_path filtering to return
        only legitimate source file changes. This is the public API for callers
        that need relative-path strings (e.g., for smell scoping and cache keys).

        Returns:
            Set of relative path strings for modified source files
        """
        absolute_paths = self._get_modified_source_files_from_git()
        return {str(p.relative_to(self.working_directory)) for p in absolute_paths}

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
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to get current commit SHA: {e}")
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
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to check for uncommitted changes: {e}")
        return False

"""Working tree state calculation for intelligent caching."""

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

from slopometry.core.git_tracker import GitTracker, get_submodule_prefixes, path_in_submodule
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

    def _list_source_files(self, ls_files_flags: list[str]) -> list[Path]:
        """Run ``git ls-files <flags>`` and return the in-scope source files it lists.

        Applies the same extension / ``should_ignore_path`` / submodule filtering as
        ``_get_modified_source_files_from_git`` so build artifacts (``build/``,
        ``dist/``, ``*.egg-info``) and submodule contents are excluded. ``git
        ls-files`` never descends into submodules (a submodule is a single gitlink
        entry, not a source path), so submodule source cannot leak in.

        Args:
            ls_files_flags: Flags appended to ``git ls-files`` (e.g. ``--cached``).

        Returns:
            List of absolute Path objects for in-scope source files.
        """
        submodule_prefixes = get_submodule_prefixes(self.working_directory)
        try:
            result = subprocess.run(
                ["git", "ls-files", *ls_files_flags],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return []
        if result.returncode != 0:
            return []

        files: list[Path] = []
        for line in result.stdout.splitlines():
            rel_path = line.strip()
            if not rel_path:
                continue
            if path_in_submodule(rel_path, submodule_prefixes):
                continue
            if is_source_file(rel_path, self.languages) and not should_ignore_path(rel_path, self.languages):
                files.append(self.working_directory / rel_path)
        return files

    def get_all_source_files(self) -> list[Path]:
        """List every non-ignored source file in the working tree (tracked + untracked).

        Uses ``git ls-files --cached --others --exclude-standard`` so the set is a
        function of the current working tree, independent of HEAD/commit state, and
        includes new untracked ``.py``/``.rs`` files while honoring ``.gitignore``.

        Returns:
            List of absolute Path objects for in-scope source files.
        """
        return self._list_source_files(["--cached", "--others", "--exclude-standard"])

    def get_untracked_source_files(self) -> list[Path]:
        """List untracked, non-ignored source files in the working tree.

        Uses ``git ls-files --others --exclude-standard`` (untracked only). The
        fast-path uses this so that creating a new source file is treated as a
        source change even though ``git diff`` (tracked-only) cannot see it.

        Returns:
            List of absolute Path objects for untracked in-scope source files.
        """
        return self._list_source_files(["--others", "--exclude-standard"])

    def calculate_source_content_key(self) -> str:
        """Commit-invariant cache key over the working-tree content of all source files.

        Hashes the current content of every in-scope ``.py``/``.rs`` file (tracked +
        untracked) and folds the ``(rel_path, blob_sha)`` pairs into a single digest.
        Blob SHAs are computed by ``git hash-object`` (C-level, fast on large repos);
        a pure-Python fallback computes the identical git blob SHA when the git call
        fails, so the key is independent of which path produced it.

        The key is a pure function of source *content*: it is stable across commits,
        branch switches, pulls, rebases, and non-source churn, and changes only when
        source bytes change (including the addition of a new untracked source file).

        Returns:
            BLAKE2b hex digest (16 chars) of the source content state.
        """
        rel_paths = sorted(str(f.relative_to(self.working_directory)) for f in self.get_all_source_files())
        if not rel_paths:
            return hashlib.blake2b(b"no-source", digest_size=8).hexdigest()

        blob_shas = self._hash_files_via_git(rel_paths)
        components = [f"{rel_path}:{blob_shas[rel_path]}" for rel_path in rel_paths if rel_path in blob_shas]
        combined = "|".join(components)
        return hashlib.blake2b(combined.encode("utf-8"), digest_size=8).hexdigest()

    def _hash_files_via_git(self, rel_paths: list[str]) -> dict[str, str]:
        """Map each relative path to its git blob SHA of current working-tree content.

        Prefers a single ``git hash-object --no-filters --stdin-paths`` call. Falls
        back to an equivalent pure-Python git-blob hash if git is unavailable or the
        output does not line up with the inputs.

        ``--no-filters`` is required for fallback parity: without it, git applies
        gitattributes / ``core.autocrlf`` clean filters (e.g. CRLF→LF) before hashing,
        producing a SHA that differs from the raw-bytes hash computed by
        ``_hash_files_in_python``. With ``--no-filters`` both paths hash the exact
        working-tree bytes, so the cache key is identical regardless of which ran.
        """
        try:
            result = subprocess.run(
                ["git", "hash-object", "--no-filters", "--stdin-paths"],
                input="\n".join(rel_paths) + "\n",
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                shas = result.stdout.split()
                if len(shas) == len(rel_paths):
                    return dict(zip(rel_paths, shas, strict=True))
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"git hash-object unavailable ({e}); using Python git-blob fallback")
        return self._hash_files_in_python(rel_paths)

    def _hash_files_in_python(self, rel_paths: list[str]) -> dict[str, str]:
        """Compute git blob SHAs in Python (``sha1("blob <len>\\0" + content)``).

        Produces byte-identical SHAs to ``git hash-object`` so the content key is
        the same regardless of whether the git fast path or this fallback ran.
        Files that cannot be read are skipped.
        """
        result: dict[str, str] = {}
        for rel_path in rel_paths:
            try:
                content = (self.working_directory / rel_path).read_bytes()
            except OSError as e:
                logger.debug(f"Skipping unreadable source file {rel_path}: {e}")
                continue
            header = f"blob {len(content)}\0".encode()
            result[rel_path] = hashlib.sha1(header + content).hexdigest()  # noqa: S324 - git blob id, not security
        return result

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
        (build artifacts, caches, etc.) and any paths inside declared
        submodules — --ignore-submodules=all on each git diff invocation
        keeps submodule gitlink diffs out of the cache key regardless of
        user git config (submodule.recurse, diff.submodule=log, etc.).

        This is tier 1 of the two-tier change detection - fast but may include
        files that only have mtime changes (which tier 2 content hash filters out).

        Returns:
            List of Path objects for source files that git reports as modified
        """
        git_patterns = get_combined_git_patterns(self.languages)
        files: set[Path] = set()
        submodule_prefixes = get_submodule_prefixes(self.working_directory)

        for pattern in git_patterns:
            try:
                result1 = subprocess.run(
                    ["git", "diff", "--name-only", "--ignore-submodules=all", "--", pattern],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                result2 = subprocess.run(
                    ["git", "diff", "--cached", "--name-only", "--ignore-submodules=all", "--", pattern],
                    cwd=self.working_directory,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                for line in result1.stdout.splitlines() + result2.stdout.splitlines():
                    if line.strip():
                        rel_path = line.strip()
                        if path_in_submodule(rel_path, submodule_prefixes):
                            continue
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

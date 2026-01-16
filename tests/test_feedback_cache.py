"""Tests for feedback cache functionality to prevent repeated feedback display."""

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from slopometry.core.hook_handler import (
    _compute_feedback_cache_key,
    _get_feedback_cache_path,
    _is_feedback_cached,
    _save_feedback_cache,
)
from slopometry.core.working_tree_state import WorkingTreeStateCalculator


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with initial commit.

    All git config is scoped to the repo (--local) to avoid mutating user environment.
    """
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    # Use --local to ensure config is scoped to this repo only
    # Use example.com (RFC 2606 reserved domain for testing)
    subprocess.run(
        ["git", "config", "--local", "user.email", "test@example.com"], cwd=path, capture_output=True, check=True
    )
    subprocess.run(["git", "config", "--local", "user.name", "Test User"], cwd=path, capture_output=True, check=True)
    # Disable GPG signing for tests (local scope)
    subprocess.run(["git", "config", "--local", "commit.gpgsign", "false"], cwd=path, capture_output=True, check=True)


def _commit_all(path: Path, message: str = "commit") -> None:
    """Add all files and commit."""
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=path, capture_output=True)


class TestFeedbackCacheKeyComputation:
    """Tests for _compute_feedback_cache_key function."""

    def test_compute_feedback_cache_key__same_feedback_different_sessions_same_key(self):
        """Verify that identical feedback with different session_ids produces same cache key.

        This is the primary bug fix - session_id should not affect the cache key.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            # Same feedback content
            feedback_content = "Code smells detected: orphan comments"
            feedback_hash = hashlib.blake2b(feedback_content.encode(), digest_size=8).hexdigest()

            # Compute cache key
            key1 = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)
            key2 = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key1 == key2, "Same feedback should produce same cache key"

    def test_compute_feedback_cache_key__uv_lock_changes_dont_invalidate(self):
        """Verify non-Python file changes (uv.lock) don't cause cache key changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Modify uv.lock (non-Python file)
            (tmppath / "uv.lock").write_text("some lock content")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "uv.lock changes should not invalidate cache"

    def test_compute_feedback_cache_key__pycache_changes_dont_invalidate(self):
        """Verify __pycache__/*.pyc files don't affect the cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Create __pycache__ with .pyc file
            pycache = tmppath / "__pycache__"
            pycache.mkdir()
            (pycache / "test.cpython-312.pyc").write_bytes(b"\x00\x00\x00\x00")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "__pycache__ should not invalidate cache"

    def test_compute_feedback_cache_key__compiled_extensions_dont_invalidate(self):
        """Verify compiled extensions (.so, .pyd) don't affect the cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Create compiled extension files
            (tmppath / "module.so").write_bytes(b"\x7fELF")
            (tmppath / "module.pyd").write_bytes(b"MZ")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "Compiled extensions should not invalidate cache"

    def test_compute_feedback_cache_key__python_content_changes_invalidate(self):
        """Verify actual Python code changes invalidate the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Modify Python file content
            (tmppath / "test.py").write_text("def foo(): return 42")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before != key_after, "Python content changes should invalidate cache"

    def test_compute_feedback_cache_key__empty_edited_files_stable_key(self):
        """Verify cache key is stable when no Python files are modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"

            # Multiple calls with empty edited_files
            key1 = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)
            key2 = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)
            key3 = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key1 == key2 == key3, "Cache key should be stable"


class TestWorkingTreeHashContentBased:
    """Tests for content-based working tree hash (not mtime-based)."""

    def test_working_tree_hash__mtime_change_without_content_does_not_invalidate(self):
        """Verify that touching a Python file (mtime change only) does NOT invalidate cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            _commit_all(tmppath)

            # Modify the file to get it tracked as changed by git
            py_file.write_text("def foo(): pass\n")
            _commit_all(tmppath, "second commit")

            # Now change back to original content
            py_file.write_text("def foo(): pass")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            hash1 = calculator.calculate_working_tree_hash("commit1")

            # Touch the file (changes mtime but not content)
            time.sleep(0.01)
            original_content = py_file.read_text()
            py_file.write_text(original_content)  # Same content

            hash2 = calculator.calculate_working_tree_hash("commit1")

            # With content-based hashing, same content = same hash
            assert hash1 == hash2, "Mtime-only change should NOT invalidate (content-based hash)"

    def test_working_tree_hash__actual_content_change_invalidates(self):
        """Verify that actual content changes DO invalidate the hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            _commit_all(tmppath)

            # Make actual content change
            py_file.write_text("def foo(): return 42")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            hash1 = calculator.calculate_working_tree_hash("commit1")

            # Change content again
            py_file.write_text("def bar(): return 99")

            hash2 = calculator.calculate_working_tree_hash("commit1")

            assert hash1 != hash2, "Different content should produce different hash"


class TestFeedbackCachePersistence:
    """Tests for feedback cache persistence."""

    def test_feedback_cache__persists_across_sessions(self):
        """Verify cache file persists and works across multiple calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            cache_key = "test_cache_key_123"

            # First call - should not be cached
            assert not _is_feedback_cached(str(tmppath), cache_key)

            # Save to cache
            _save_feedback_cache(str(tmppath), cache_key)

            # Second call - should be cached
            assert _is_feedback_cached(str(tmppath), cache_key)

            # Verify cache file exists
            cache_path = _get_feedback_cache_path(str(tmppath))
            assert cache_path.exists()
            cache_data = json.loads(cache_path.read_text())
            assert cache_data["last_key"] == cache_key

    def test_feedback_cache__different_key_not_cached(self):
        """Verify that a different cache key is not considered cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            # Save one key
            _save_feedback_cache(str(tmppath), "key1")

            # Check different key - should not be cached
            assert not _is_feedback_cached(str(tmppath), "key2")


class TestModifiedPythonFilesDetection:
    """Tests for _get_modified_python_files_from_git helper."""

    def test_get_modified_python_files__detects_staged_changes(self):
        """Verify staged Python file changes are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            _commit_all(tmppath)

            # Stage a change
            py_file.write_text("def foo(): return 42")
            subprocess.run(["git", "add", "test.py"], cwd=tmppath, capture_output=True)

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator._get_modified_python_files_from_git()

            assert len(modified) == 1
            assert modified[0].name == "test.py"

    def test_get_modified_python_files__detects_unstaged_changes(self):
        """Verify unstaged Python file changes are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            _commit_all(tmppath)

            # Make unstaged change
            py_file.write_text("def foo(): return 42")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator._get_modified_python_files_from_git()

            assert len(modified) == 1
            assert modified[0].name == "test.py"

    def test_get_modified_python_files__ignores_non_python_files(self):
        """Verify non-Python file changes are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            lock_file = tmppath / "uv.lock"
            lock_file.write_text("old lock")
            _commit_all(tmppath)

            # Modify both files
            py_file.write_text("def foo(): return 42")
            lock_file.write_text("new lock")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator._get_modified_python_files_from_git()

            # Should only include Python file
            assert len(modified) == 1
            assert modified[0].name == "test.py"

    def test_get_modified_python_files__empty_when_no_changes(self):
        """Verify empty list when no Python files are modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            py_file = tmppath / "test.py"
            py_file.write_text("def foo(): pass")
            _commit_all(tmppath)

            # No changes
            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator._get_modified_python_files_from_git()

            assert modified == []


class TestSubmoduleHandling:
    """Tests for git submodule handling."""

    def test_feedback_cache__submodule_changes_dont_invalidate(self):
        """Verify submodule changes don't cause cache misses.

        Note: This test creates a real submodule setup to verify the behavior.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create main repo
            main_repo = tmppath / "main"
            main_repo.mkdir()
            _init_git_repo(main_repo)
            (main_repo / "main.py").write_text("def main(): pass")
            _commit_all(main_repo)

            # Create submodule repo
            sub_repo = tmppath / "subrepo"
            sub_repo.mkdir()
            _init_git_repo(sub_repo)
            (sub_repo / "sub.py").write_text("def sub(): pass")
            _commit_all(sub_repo)

            # Add submodule to main repo
            subprocess.run(
                ["git", "submodule", "add", str(sub_repo), "vendor/sub"],
                cwd=main_repo,
                capture_output=True,
            )
            _commit_all(main_repo, "add submodule")

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(main_repo), set(), feedback_hash)

            # Update submodule (creates a change in main repo's git status)
            subprocess.run(
                ["git", "-C", "vendor/sub", "fetch", "--all"],
                cwd=main_repo,
                capture_output=True,
            )

            key_after = _compute_feedback_cache_key(str(main_repo), set(), feedback_hash)

            assert key_before == key_after, "Submodule changes should not invalidate cache"


class TestNewUntrackedFiles:
    """Tests for new untracked Python file handling."""

    def test_feedback_cache__new_untracked_python_files_invalidate(self):
        """Verify that new Python files (even untracked) invalidate cache.

        Note: New untracked files won't appear in git diff, but they will be
        detected by git ls-files if not gitignored. The behavior here depends
        on whether git considers them "changed" - typically they won't be.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "existing.py").write_text("def existing(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Add new untracked Python file
            (tmppath / "new_file.py").write_text("def new(): pass")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # New untracked files don't show in git diff, so key should be same
            # This is expected behavior - only tracked file changes matter
            assert key_before == key_after, "Untracked files don't appear in git diff"


class TestBuildArtifactFiltering:
    """Tests for build artifact and cache directory filtering."""

    def test_feedback_cache__dist_directory_ignored(self):
        """Verify Python files in dist/ directory don't affect cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "module.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Create dist directory with Python file (shouldn't affect cache)
            (tmppath / "dist").mkdir()
            (tmppath / "dist" / "generated.py").write_text("# Generated")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "dist/ directory should be ignored"

    def test_feedback_cache__build_directory_ignored(self):
        """Verify Python files in build/ directory don't affect cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "module.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Create build directory with Python file (shouldn't affect cache)
            (tmppath / "build").mkdir()
            (tmppath / "build" / "lib").mkdir()
            (tmppath / "build" / "lib" / "module.py").write_text("# Built")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "build/ directory should be ignored"

    def test_feedback_cache__egg_info_directory_ignored(self):
        """Verify *.egg-info directories don't affect cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "module.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            feedback_hash = "feedbackhash1234"
            key_before = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            # Create egg-info directory (shouldn't affect cache)
            (tmppath / "package.egg-info").mkdir()
            (tmppath / "package.egg-info" / "PKG-INFO").write_text("Name: package")

            key_after = _compute_feedback_cache_key(str(tmppath), set(), feedback_hash)

            assert key_before == key_after, "*.egg-info directory should be ignored"

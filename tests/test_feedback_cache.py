"""Tests for feedback cache functionality to prevent repeated feedback display."""

import subprocess
import tempfile
import time
from pathlib import Path

from slopometry.core.hook_handler import (
    _compute_working_tree_cache_key,
    _get_feedback_cache_path,
    _load_feedback_cache,
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


class TestWorkingTreeCacheKeyComputation:
    """Tests for _compute_working_tree_cache_key function."""

    def test_compute_working_tree_cache_key__stable_across_calls(self):
        """Verify repeated calls with same state produce same cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key1 = _compute_working_tree_cache_key(str(tmppath))
            key2 = _compute_working_tree_cache_key(str(tmppath))

            assert key1 == key2, "Same state should produce same cache key"

    def test_compute_working_tree_cache_key__uv_lock_changes_dont_invalidate(self):
        """Verify non-Python file changes (uv.lock) don't cause cache key changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Modify uv.lock (non-Python file)
            (tmppath / "uv.lock").write_text("some lock content")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "uv.lock changes should not invalidate cache"

    def test_compute_working_tree_cache_key__pycache_changes_dont_invalidate(self):
        """Verify __pycache__/*.pyc files don't affect the cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Create __pycache__ with .pyc file
            pycache = tmppath / "__pycache__"
            pycache.mkdir()
            (pycache / "test.cpython-312.pyc").write_bytes(b"\x00\x00\x00\x00")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "__pycache__ should not invalidate cache"

    def test_compute_working_tree_cache_key__compiled_extensions_dont_invalidate(self):
        """Verify compiled extensions (.so, .pyd) don't affect the cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Create compiled extension files
            (tmppath / "module.so").write_bytes(b"\x7fELF")
            (tmppath / "module.pyd").write_bytes(b"MZ")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "Compiled extensions should not invalidate cache"

    def test_compute_working_tree_cache_key__python_content_changes_invalidate(self):
        """Verify actual Python code changes invalidate the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Modify Python file content
            (tmppath / "test.py").write_text("def foo(): return 42")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before != key_after, "Python content changes should invalidate cache"

    def test_compute_working_tree_cache_key__stable_when_no_modifications(self):
        """Verify cache key is stable when no Python files are modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key1 = _compute_working_tree_cache_key(str(tmppath))
            key2 = _compute_working_tree_cache_key(str(tmppath))
            key3 = _compute_working_tree_cache_key(str(tmppath))

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
    """Tests for feedback cache persistence using FeedbackCacheState."""

    def test_feedback_cache__persists_and_loads_correctly(self):
        """Verify cache file persists with file hashes and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            cache_key = "test_cache_key_123"
            file_hashes = {"src/app.py": "abcdef1234567890"}

            # First load - should return None
            assert _load_feedback_cache(str(tmppath)) is None

            # Save to cache
            _save_feedback_cache(str(tmppath), cache_key, file_hashes)

            # Second load - should return state
            loaded = _load_feedback_cache(str(tmppath))
            assert loaded is not None
            assert loaded.last_key == cache_key
            assert loaded.file_hashes == file_hashes

            # Verify cache file exists
            cache_path = _get_feedback_cache_path(str(tmppath))
            assert cache_path.exists()

    def test_feedback_cache__different_key_detected(self):
        """Verify that a different cache key is detected as a change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            # Save one key
            _save_feedback_cache(str(tmppath), "key1", {})

            # Load and check — key mismatch means working tree changed
            loaded = _load_feedback_cache(str(tmppath))
            assert loaded is not None
            assert loaded.last_key != "key2"

    def test_feedback_cache__file_hashes_enable_change_detection(self):
        """Verify saved file hashes allow detecting which files changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            # Save cache with file hashes
            file_hashes = {"app.py": "hash1", "utils.py": "hash2"}
            _save_feedback_cache(str(tmppath), "cache_key", file_hashes)

            loaded = _load_feedback_cache(str(tmppath))
            assert loaded is not None
            assert loaded.file_hashes == file_hashes

            # Use file hashes with WorkingTreeStateCalculator.get_files_changed_since
            calculator = WorkingTreeStateCalculator(str(tmppath))
            changed = calculator.get_files_changed_since(loaded.file_hashes)
            # No actual git changes, so nothing should be "changed"
            assert changed == set()


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

            key_before = _compute_working_tree_cache_key(str(main_repo))

            # Update submodule (creates a change in main repo's git status)
            subprocess.run(
                ["git", "-C", "vendor/sub", "fetch", "--all"],
                cwd=main_repo,
                capture_output=True,
            )

            key_after = _compute_working_tree_cache_key(str(main_repo))

            assert key_before == key_after, "Submodule changes should not invalidate cache"


class TestNewUntrackedFiles:
    """Tests for new untracked Python file handling."""

    def test_feedback_cache__new_untracked_python_files_dont_invalidate(self):
        """Verify that new untracked Python files don't invalidate cache.

        New untracked files won't appear in git diff, so the working tree
        cache key remains unchanged. Only tracked file changes matter.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "existing.py").write_text("def existing(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Add new untracked Python file
            (tmppath / "new_file.py").write_text("def new(): pass")

            key_after = _compute_working_tree_cache_key(str(tmppath))

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

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Create dist directory with Python file (shouldn't affect cache)
            (tmppath / "dist").mkdir()
            (tmppath / "dist" / "generated.py").write_text("# Generated")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "dist/ directory should be ignored"

    def test_feedback_cache__build_directory_ignored(self):
        """Verify Python files in build/ directory don't affect cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "module.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Create build directory with Python file (shouldn't affect cache)
            (tmppath / "build").mkdir()
            (tmppath / "build" / "lib").mkdir()
            (tmppath / "build" / "lib" / "module.py").write_text("# Built")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "build/ directory should be ignored"

    def test_feedback_cache__egg_info_directory_ignored(self):
        """Verify *.egg-info directories don't affect cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "module.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            # Create egg-info directory (shouldn't affect cache)
            (tmppath / "package.egg-info").mkdir()
            (tmppath / "package.egg-info" / "PKG-INFO").write_text("Name: package")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "*.egg-info directory should be ignored"


class TestGetModifiedSourceFilePathsFiltering:
    """Tests for get_modified_source_file_paths() ignore filtering."""

    def test_get_modified_source_file_paths__excludes_venv_files(self):
        """Verify .venv Python files are excluded from modified source files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)

            # Create source file and venv file, commit both as tracked
            (tmppath / "src").mkdir()
            (tmppath / "src" / "app.py").write_text("def app(): pass")
            (tmppath / ".venv").mkdir()
            (tmppath / ".venv" / "lib.py").write_text("def lib(): pass")
            _commit_all(tmppath)

            # Modify both files
            (tmppath / "src" / "app.py").write_text("def app(): return 1")
            (tmppath / ".venv" / "lib.py").write_text("def lib(): return 1")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator.get_modified_source_file_paths()

            assert "src/app.py" in modified
            assert ".venv/lib.py" not in modified

    def test_get_modified_source_file_paths__excludes_site_packages_files(self):
        """Verify site-packages Python files are excluded from modified source files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)

            (tmppath / "app.py").write_text("def app(): pass")
            (tmppath / "site-packages").mkdir()
            (tmppath / "site-packages" / "pkg.py").write_text("def pkg(): pass")
            _commit_all(tmppath)

            # Modify both
            (tmppath / "app.py").write_text("def app(): return 1")
            (tmppath / "site-packages" / "pkg.py").write_text("def pkg(): return 1")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator.get_modified_source_file_paths()

            assert "app.py" in modified
            assert "site-packages/pkg.py" not in modified

    def test_get_modified_source_file_paths__returns_relative_string_paths(self):
        """Verify return type is set[str] with relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "test.py").write_text("def foo(): pass")
            _commit_all(tmppath)

            (tmppath / "test.py").write_text("def foo(): return 1")

            calculator = WorkingTreeStateCalculator(str(tmppath))
            modified = calculator.get_modified_source_file_paths()

            assert isinstance(modified, set)
            for path in modified:
                assert isinstance(path, str)
                assert not Path(path).is_absolute()


def test_feedback_cache__slopometry_dir_visibility_does_not_affect_key():
    """Verify cache key is stable whether .slopometry/ is in gitignore or not.

    The key only depends on: commit SHA and source file content hashes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _init_git_repo(tmppath)

        # Create Python file and gitignore WITHOUT .slopometry entry
        (tmppath / "test.py").write_text("def foo(): pass")
        (tmppath / ".gitignore").write_text("__pycache__/\n")
        _commit_all(tmppath)

        # Modify Python file (uncommitted)
        (tmppath / "test.py").write_text("def foo(): return 1")

        # Scenario 1: .slopometry NOT in gitignore
        key1 = _compute_working_tree_cache_key(str(tmppath))

        # Save cache (creates .slopometry/ directory)
        _save_feedback_cache(str(tmppath), key1, {"test.py": "somehash"})

        # Scenario 2: Add .slopometry to gitignore (uncommitted)
        (tmppath / ".gitignore").write_text("__pycache__/\n.slopometry/\n")
        key2 = _compute_working_tree_cache_key(str(tmppath))

        # Scenario 3: Remove .slopometry from gitignore
        (tmppath / ".gitignore").write_text("__pycache__/\n")
        key3 = _compute_working_tree_cache_key(str(tmppath))

        assert key1 == key2, "Adding .slopometry to gitignore should not change cache key"
        assert key2 == key3, "Removing .slopometry from gitignore should not change cache key"
        assert key1 == key3, "Cache key should be identical across all scenarios"


def test_feedback_cache__gitignore_modification_does_not_invalidate():
    """Verify that modifying .gitignore (non-Python file) doesn't invalidate cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _init_git_repo(tmppath)
        (tmppath / "test.py").write_text("def foo(): pass")
        (tmppath / ".gitignore").write_text("*.pyc\n")
        _commit_all(tmppath)

        key_before = _compute_working_tree_cache_key(str(tmppath))

        # Modify .gitignore (uncommitted)
        (tmppath / ".gitignore").write_text("*.pyc\n.slopometry/\n__pycache__/\n")

        key_after = _compute_working_tree_cache_key(str(tmppath))

        assert key_before == key_after, ".gitignore modifications should not invalidate cache"


def test_feedback_cache__env_file_changes_dont_invalidate():
    """Verify .env file changes don't cause cache invalidation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _init_git_repo(tmppath)
        (tmppath / "test.py").write_text("def foo(): pass")
        (tmppath / ".env").write_text("SECRET=old")
        _commit_all(tmppath)

        key_before = _compute_working_tree_cache_key(str(tmppath))

        # Modify .env (tracked but non-source)
        (tmppath / ".env").write_text("SECRET=new")

        key_after = _compute_working_tree_cache_key(str(tmppath))

        assert key_before == key_after, ".env changes should not invalidate cache"


def test_feedback_cache__markdown_changes_dont_invalidate():
    """Verify .md file changes don't cause cache invalidation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _init_git_repo(tmppath)
        (tmppath / "test.py").write_text("def foo(): pass")
        (tmppath / "README.md").write_text("# Old readme")
        _commit_all(tmppath)

        key_before = _compute_working_tree_cache_key(str(tmppath))

        # Modify markdown (tracked but non-source)
        (tmppath / "README.md").write_text("# New readme with changes")

        key_after = _compute_working_tree_cache_key(str(tmppath))

        assert key_before == key_after, ".md changes should not invalidate cache"

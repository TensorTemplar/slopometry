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


def _add_submodule(main_repo: Path, sub_repo: Path, rel_path: str) -> None:
    """Add sub_repo as a submodule of main_repo at rel_path.

    Uses protocol.file.allow=always to permit local-path submodule URLs
    (disabled by default on recent git).
    """
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(sub_repo),
            rel_path,
        ],
        cwd=main_repo,
        capture_output=True,
        check=True,
    )
    _commit_all(main_repo, "add submodule")


class TestSubmoduleHandling:
    """Tests that nothing inside a git submodule can invalidate the parent's cache.

    The stop-hook cache key must be stable against every kind of submodule
    state change. Submodule code belongs to the submodule's own repository
    and is not part of the parent project's source tree.
    """

    def _build_main_with_submodule(self, tmppath: Path) -> tuple[Path, Path]:
        """Set up a parent repo with one submodule containing a python file."""
        main_repo = tmppath / "main"
        main_repo.mkdir()
        _init_git_repo(main_repo)
        (main_repo / "main.py").write_text("def main(): pass")
        _commit_all(main_repo)

        sub_repo = tmppath / "subrepo"
        sub_repo.mkdir()
        _init_git_repo(sub_repo)
        (sub_repo / "sub.py").write_text("def sub(): pass")
        _commit_all(sub_repo)

        _add_submodule(main_repo, sub_repo, "vendor/sub")
        return main_repo, sub_repo

    def test_feedback_cache__edit_inside_submodule_does_not_invalidate(self):
        """A tracked .py edited inside a submodule must not invalidate the parent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_repo, _sub = self._build_main_with_submodule(Path(tmpdir))

            key_before = _compute_working_tree_cache_key(str(main_repo))

            (main_repo / "vendor" / "sub" / "sub.py").write_text("def sub(): return 99")

            key_after = _compute_working_tree_cache_key(str(main_repo))
            assert key_before == key_after

    def test_feedback_cache__submodule_head_move_does_not_invalidate(self):
        """A submodule HEAD pointer move (dirty gitlink) must not invalidate the parent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_repo, _sub = self._build_main_with_submodule(Path(tmpdir))
            sub_checkout = main_repo / "vendor" / "sub"
            # git submodule add creates a worktree whose config does not inherit the
            # source repo's local settings; set identity AND disable signing so the
            # commit below can be authored on machines with commit.gpgsign=true.
            for key, value in (
                ("user.email", "test@example.com"),
                ("user.name", "Test User"),
                ("commit.gpgsign", "false"),
            ):
                subprocess.run(
                    ["git", "config", "--local", key, value],
                    cwd=sub_checkout,
                    capture_output=True,
                    check=True,
                )

            key_before = _compute_working_tree_cache_key(str(main_repo))

            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "bump"],
                cwd=sub_checkout,
                capture_output=True,
                check=True,
            )

            key_after = _compute_working_tree_cache_key(str(main_repo))
            assert key_before == key_after

    def test_feedback_cache__submodule_recurse_config_does_not_invalidate(self):
        """submodule.recurse=true on the parent must not leak submodule diffs into cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_repo, _sub = self._build_main_with_submodule(Path(tmpdir))
            subprocess.run(
                ["git", "config", "--local", "submodule.recurse", "true"],
                cwd=main_repo,
                capture_output=True,
                check=True,
            )

            key_before = _compute_working_tree_cache_key(str(main_repo))
            (main_repo / "vendor" / "sub" / "sub.py").write_text("def sub(): return 1")
            key_after = _compute_working_tree_cache_key(str(main_repo))

            assert key_before == key_after

    def test_feedback_cache__parent_edit_still_invalidates_with_submodule(self):
        """Sanity: real parent-source edits must still invalidate when submodules exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_repo, _sub = self._build_main_with_submodule(Path(tmpdir))

            key_before = _compute_working_tree_cache_key(str(main_repo))
            (main_repo / "main.py").write_text("def main(): return 42")
            key_after = _compute_working_tree_cache_key(str(main_repo))

            assert key_before != key_after


class TestNewUntrackedFiles:
    """Tests for new untracked Python file handling."""

    def test_feedback_cache__new_untracked_python_files_invalidate(self):
        """Verify that a new untracked Python file DOES invalidate the cache.

        Creating a source file is a code change. The content key is built from
        `git ls-files --cached --others`, which includes untracked (non-ignored)
        source files, so the key changes and the hook fires.
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

            assert key_before != key_after, "A new untracked source file is a code change"

    def test_feedback_cache__new_untracked_non_source_file_does_not_invalidate(self):
        """A new untracked non-source file (docs/config) must not invalidate the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "existing.py").write_text("def existing(): pass")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))
            (tmppath / "NOTES.md").write_text("# scratch notes")
            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "Untracked non-source files are not code changes"


class TestCommitInvariance:
    """Tests that the firing key is a pure function of source *content*.

    Committing, switching branches, pulling, or otherwise moving HEAD does not
    change source bytes, so it must not change the key. This is the core fix for
    the hook firing on commits and other non-source git activity.
    """

    def test_compute_working_tree_cache_key__commit_of_identical_source_does_not_change_key(self):
        """Committing already-written code must not change the key (no re-fire)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "app.py").write_text("def app(): pass")
            _commit_all(tmppath)

            # Edit the file (uncommitted) — this is a real change
            (tmppath / "app.py").write_text("def app(): return 42")
            key_dirty = _compute_working_tree_cache_key(str(tmppath))

            # Commit that exact content — bytes are unchanged, only HEAD moves
            _commit_all(tmppath, "commit the edit")
            key_committed = _compute_working_tree_cache_key(str(tmppath))

            assert key_dirty == key_committed, "Committing identical source must not change the key"

    def test_compute_working_tree_cache_key__docs_only_commit_does_not_change_key(self):
        """A commit that touches only non-source files must not change the key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "app.py").write_text("def app(): pass")
            (tmppath / "README.md").write_text("# old")
            _commit_all(tmppath)

            key_before = _compute_working_tree_cache_key(str(tmppath))

            (tmppath / "README.md").write_text("# new and improved")
            _commit_all(tmppath, "docs only")

            key_after = _compute_working_tree_cache_key(str(tmppath))

            assert key_before == key_after, "A docs-only commit must not change the key"

    def test_compute_working_tree_cache_key__branch_switch_to_identical_source_does_not_change_key(self):
        """Switching to a branch with identical source content must not change the key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "app.py").write_text("def app(): pass")
            _commit_all(tmppath)

            key_main = _compute_working_tree_cache_key(str(tmppath))

            # Capture the default branch name (git init may produce main or master)
            original_branch = subprocess.run(
                ["git", "branch", "--show-current"], cwd=tmppath, capture_output=True, text=True, check=True
            ).stdout.strip()

            # Create a branch with a non-source-only difference, then switch back
            subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=tmppath, capture_output=True, check=True)
            (tmppath / "CHANGELOG.md").write_text("- nothing")
            _commit_all(tmppath, "docs on feature")

            subprocess.run(["git", "checkout", "-q", original_branch], cwd=tmppath, capture_output=True, check=True)
            key_back = _compute_working_tree_cache_key(str(tmppath))

            assert key_main == key_back, "Branch metadata changes must not change the key"

    def test_compute_working_tree_cache_key__committed_source_edit_then_revert_round_trips_key(self):
        """Editing+committing then reverting the source returns to the original key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _init_git_repo(tmppath)
            (tmppath / "app.py").write_text("def app(): pass")
            _commit_all(tmppath)
            key_original = _compute_working_tree_cache_key(str(tmppath))

            (tmppath / "app.py").write_text("def app(): return 1")
            _commit_all(tmppath, "change")
            key_changed = _compute_working_tree_cache_key(str(tmppath))

            (tmppath / "app.py").write_text("def app(): pass")
            key_reverted = _compute_working_tree_cache_key(str(tmppath))

            assert key_original != key_changed, "Real source change must change the key"
            assert key_original == key_reverted, "Reverting source content returns to the original key"


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

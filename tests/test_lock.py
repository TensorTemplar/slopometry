"""Tests for SlopometryLock."""

import os
import time
from unittest.mock import patch

from slopometry.core.lock import (
    STALE_LOCK_AGE_SECONDS,
    SlopometryLock,
    _project_lock_name,
    cleanup_stale_locks,
)


class TestSlopometryLock:
    def test_acquire_release__acquires_and_releases_lock(self, tmp_path):
        """Test basic acquire and release cycle."""
        lock = SlopometryLock(project_dir="/tmp/myproject", timeout=1.0)
        lock.lock_file_path = tmp_path / "test.lock"

        with lock.acquire() as acquired:
            assert acquired
            assert lock.lock_file_path.exists()

    def test_acquire__times_out_when_locked_by_another(self, tmp_path):
        """Test that second acquisition times out when locked."""
        lock1 = SlopometryLock(project_dir="/tmp/myproject", timeout=0.1)
        lock1.lock_file_path = tmp_path / "test.lock"

        lock2 = SlopometryLock(project_dir="/tmp/myproject", timeout=0.1)
        lock2.lock_file_path = tmp_path / "test.lock"

        with lock1.acquire() as acquired1:
            assert acquired1

            start = time.time()
            with lock2.acquire() as acquired2:
                assert not acquired2
            duration = time.time() - start
            assert duration >= 0.1

    def test_acquire__releases_lock_after_exception(self, tmp_path):
        """Test that lock is released even when exception occurs in context."""
        lock = SlopometryLock(project_dir="/tmp/myproject", timeout=1.0)
        lock.lock_file_path = tmp_path / "test.lock"

        try:
            with lock.acquire() as acquired:
                assert acquired
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released, so we can acquire it again
        lock2 = SlopometryLock(project_dir="/tmp/myproject", timeout=0.1)
        lock2.lock_file_path = tmp_path / "test.lock"
        with lock2.acquire() as acquired2:
            assert acquired2

    def test_different_projects__get_different_lock_files(self, tmp_path):
        """Different project directories produce different lock files."""
        lock_a = SlopometryLock(project_dir="/home/user/project-a")
        lock_b = SlopometryLock(project_dir="/home/user/project-b")

        assert lock_a.lock_file_path != lock_b.lock_file_path

    def test_same_project__gets_same_lock_file(self):
        """Same project directory always produces the same lock file."""
        lock1 = SlopometryLock(project_dir="/home/user/project-a")
        lock2 = SlopometryLock(project_dir="/home/user/project-a")

        assert lock1.lock_file_path == lock2.lock_file_path

    def test_no_project_dir__gets_global_lock(self):
        """When no project_dir is given, a global fallback lock is used."""
        lock = SlopometryLock(project_dir=None)
        assert lock.lock_file_path.name == "slopometry_global.lock"

    def test_different_projects__do_not_block_each_other(self, tmp_path):
        """Locks from different projects don't contend."""
        lock_a = SlopometryLock(project_dir="/tmp/project-a", timeout=0.1)
        lock_a.lock_file_path = tmp_path / "a.lock"

        lock_b = SlopometryLock(project_dir="/tmp/project-b", timeout=0.1)
        lock_b.lock_file_path = tmp_path / "b.lock"

        with lock_a.acquire() as acquired_a:
            assert acquired_a
            with lock_b.acquire() as acquired_b:
                assert acquired_b  # Should NOT be blocked by lock_a


class TestProjectLockName:
    def test_project_lock_name__deterministic(self):
        """Same path always gives the same lock name."""
        name1 = _project_lock_name("/home/user/code/myproject")
        name2 = _project_lock_name("/home/user/code/myproject")
        assert name1 == name2

    def test_project_lock_name__different_paths_different_names(self):
        """Different paths give different lock names."""
        name1 = _project_lock_name("/home/user/project-a")
        name2 = _project_lock_name("/home/user/project-b")
        assert name1 != name2

    def test_project_lock_name__format(self):
        """Lock name follows expected format."""
        name = _project_lock_name("/some/path")
        assert name.startswith("slopometry_")
        assert name.endswith(".lock")


class TestCleanupStaleLocks:
    def test_cleanup_stale_locks__removes_old_files(self, tmp_path):
        """Stale lock files older than threshold are removed."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()

        stale_lock = lock_dir / "slopometry_abc123.lock"
        stale_lock.touch()

        # Set mtime to 3 hours ago (exceeds 2-hour threshold)
        old_time = time.time() - (3 * 60 * 60)
        os.utime(stale_lock, (old_time, old_time))

        with patch("slopometry.core.lock._get_lock_directory", return_value=lock_dir):
            removed = cleanup_stale_locks()

        assert removed == 1
        assert not stale_lock.exists()

    def test_cleanup_stale_locks__preserves_fresh_files(self, tmp_path):
        """Fresh lock files are not removed."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()

        fresh_lock = lock_dir / "slopometry_abc123.lock"
        fresh_lock.touch()

        with patch("slopometry.core.lock._get_lock_directory", return_value=lock_dir):
            removed = cleanup_stale_locks()

        assert removed == 0
        assert fresh_lock.exists()

    def test_cleanup_stale_locks__ignores_non_slopometry_files(self, tmp_path):
        """Files not matching slopometry_*.lock pattern are ignored."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()

        other_file = lock_dir / "other.lock"
        other_file.touch()

        old_time = time.time() - (3 * 60 * 60)
        os.utime(other_file, (old_time, old_time))

        with patch("slopometry.core.lock._get_lock_directory", return_value=lock_dir):
            removed = cleanup_stale_locks()

        assert removed == 0
        assert other_file.exists()

    def test_stale_lock_age__is_two_hours(self):
        """Stale lock age threshold is 2 hours."""
        assert STALE_LOCK_AGE_SECONDS == 7200

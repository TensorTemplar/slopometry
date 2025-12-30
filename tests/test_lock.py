"""Tests for SlopometryLock."""

import time
from unittest.mock import patch

from slopometry.core.lock import SlopometryLock


class TestSlopometryLock:
    def test_acquire_release__acquires_and_releases_lock(self, tmp_path):
        """Test basic acquire and release cycle."""
        lock = SlopometryLock(timeout=1.0)
        lock.lock_file_path = tmp_path / "test.lock"

        with lock.acquire() as acquired:
            assert acquired
            assert lock.lock_file_path.exists()

    def test_acquire__times_out_when_locked_by_another(self, tmp_path):
        """Test that second acquisition times out when locked."""
        lock1 = SlopometryLock(timeout=0.1)
        lock1.lock_file_path = tmp_path / "test.lock"

        lock2 = SlopometryLock(timeout=0.1)
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
        lock = SlopometryLock(timeout=1.0)
        lock.lock_file_path = tmp_path / "test.lock"

        try:
            with lock.acquire() as acquired:
                assert acquired
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Lock should be released, so we can acquire it again
        lock2 = SlopometryLock(timeout=0.1)
        lock2.lock_file_path = tmp_path / "test.lock"
        with lock2.acquire() as acquired2:
            assert acquired2

    def test_get_lock_file_path__uses_db_parent_when_exists(self, tmp_path):
        """Test lock file path uses database parent directory when it exists."""
        db_path = tmp_path / "slopometry.db"

        with patch("slopometry.core.lock.settings") as mock_settings:
            mock_settings.resolved_database_path = db_path
            lock = SlopometryLock()
            assert lock.lock_file_path == tmp_path / "slopometry.lock"

    def test_get_lock_file_path__falls_back_to_tempdir(self):
        """Test lock file path falls back to tempdir when db path unavailable."""
        import tempfile
        from pathlib import Path

        with patch("slopometry.core.lock.settings") as mock_settings:
            mock_settings.resolved_database_path = None
            lock = SlopometryLock()
            assert lock.lock_file_path == Path(tempfile.gettempdir()) / "slopometry.lock"

    def test_get_lock_file_path__falls_back_on_exception(self):
        """Test lock file path falls back to tempdir on settings exception."""
        import tempfile
        from pathlib import Path

        with patch("slopometry.core.lock.settings") as mock_settings:
            mock_settings.resolved_database_path = property(
                fget=lambda self: (_ for _ in ()).throw(RuntimeError("Settings error"))
            )
            type(mock_settings).resolved_database_path = property(
                fget=lambda self: (_ for _ in ()).throw(RuntimeError("Settings error"))
            )
            lock = SlopometryLock()
            assert lock.lock_file_path == Path(tempfile.gettempdir()) / "slopometry.lock"

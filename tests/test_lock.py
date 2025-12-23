"""Tests for SlopometryLock."""

import time

from slopometry.core.lock import SlopometryLock


class TestSlopometryLock:
    def test_acquire_release(self, tmp_path):
        """Test basic acquire and release cycle."""
        lock = SlopometryLock(timeout=1.0)
        lock.lock_file_path = tmp_path / "test.lock"

        with lock.acquire() as acquired:
            assert acquired
            assert lock.lock_file_path.exists()

    def test_concurrent_access(self, tmp_path):
        """Test that second acquisition fails/waits when locked."""
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

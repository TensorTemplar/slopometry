"""File-based locking for Slopometry hook coordination."""

import fcntl
import logging
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


class SlopometryLock:
    """Acquires a file lock to prevent overlapping hook executions."""

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self.lock_file_path = self._get_lock_file_path()
        self._lock_file_fd = None

    def _get_lock_file_path(self) -> Path:
        """Get path for the lock file."""
        try:
            db_path = settings.resolved_database_path
            if db_path and db_path.parent.exists():
                return db_path.parent / "slopometry.lock"
        except Exception as e:
            logger.debug(f"Failed to get db path for lock file, using tempdir: {e}")

        return Path(tempfile.gettempdir()) / "slopometry.lock"

    @contextmanager
    def acquire(self) -> Generator[bool]:
        """Attempt to acquire the lock.

        Yields:
            True if lock acquired, False if timed out.
        """
        start_time = time.time()
        self._lock_file_fd = open(self.lock_file_path, "w")

        acquired = False
        try:
            while True:
                try:
                    fcntl.flock(self._lock_file_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if time.time() - start_time >= self.timeout:
                        break
                    time.sleep(0.1)

            yield acquired

        finally:
            if acquired:
                try:
                    fcntl.flock(self._lock_file_fd, fcntl.LOCK_UN)
                except OSError as e:
                    logger.debug(f"Failed to unlock file (may already be released): {e}")

            try:
                self._lock_file_fd.close()
            except OSError as e:
                logger.debug(f"Failed to close lock file (may already be closed): {e}")

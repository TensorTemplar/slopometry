"""File-based locking for Slopometry hook coordination.

Uses per-project lock files so that a stuck analysis in one project
does not block hooks in other projects. Stale lock files older than
STALE_LOCK_AGE_SECONDS are cleaned up automatically.
"""

import fcntl
import hashlib
import logging
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from slopometry.core.settings import settings

logger = logging.getLogger(__name__)

STALE_LOCK_AGE_SECONDS = 2 * 60 * 60


def _get_lock_directory() -> Path:
    """Get the directory where lock files are stored."""
    try:
        db_path = settings.resolved_database_path
        if db_path and db_path.parent.exists():
            lock_dir = db_path.parent / "locks"
            lock_dir.mkdir(exist_ok=True)
            return lock_dir
    except Exception as e:
        logger.debug(f"Failed to get db path for lock dir, using tempdir: {e}")

    lock_dir = Path(tempfile.gettempdir()) / "slopometry_locks"
    lock_dir.mkdir(exist_ok=True)
    return lock_dir


def _project_lock_name(project_dir: str) -> str:
    """Derive a stable lock file name from a project directory path.

    Uses a short blake2b hash to avoid filesystem issues with long paths.
    """
    digest = hashlib.blake2b(project_dir.encode(), digest_size=8).hexdigest()
    return f"slopometry_{digest}.lock"


def cleanup_stale_locks() -> int:
    """Remove lock files older than STALE_LOCK_AGE_SECONDS.

    Returns:
        Number of stale lock files removed.
    """
    lock_dir = _get_lock_directory()
    removed = 0
    now = time.time()

    try:
        for lock_file in lock_dir.glob("slopometry_*.lock"):
            try:
                age = now - lock_file.stat().st_mtime
                if age > STALE_LOCK_AGE_SECONDS:
                    lock_file.unlink(missing_ok=True)
                    removed += 1
                    logger.debug(f"Removed stale lock file: {lock_file} (age: {age:.0f}s)")
            except OSError as e:
                logger.debug(f"Failed to check/remove lock file {lock_file}: {e}")
    except OSError as e:
        logger.debug(f"Failed to scan lock directory {lock_dir}: {e}")

    return removed


class SlopometryLock:
    """Acquires a per-project file lock to prevent overlapping hook executions.

    Each project directory gets its own lock file so that a stuck analysis in
    one project does not block hooks in other projects.
    """

    def __init__(self, project_dir: str | None = None, timeout: float = 2.0):
        """Initialize a project-scoped lock.

        Args:
            project_dir: Project working directory. When None, falls back to
                         a global lock (backwards-compatible for call sites
                         that don't have a project directory yet).
            timeout: Maximum seconds to wait for lock acquisition.
        """
        self.timeout = timeout
        self.lock_file_path = self._get_lock_file_path(project_dir)
        self._lock_file_fd = None

    def _get_lock_file_path(self, project_dir: str | None) -> Path:
        """Get path for the per-project lock file."""
        lock_dir = _get_lock_directory()

        if project_dir:
            return lock_dir / _project_lock_name(project_dir)

        return lock_dir / "slopometry_global.lock"

    @contextmanager
    def acquire(self) -> Generator[bool]:
        """Attempt to acquire the lock.

        Cleans up stale lock files before attempting acquisition.

        Yields:
            True if lock acquired, False if timed out.
        """
        cleanup_stale_locks()

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

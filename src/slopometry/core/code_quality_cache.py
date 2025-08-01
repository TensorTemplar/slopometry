"""Code quality metrics caching for performance optimization."""

import json
import sqlite3
from datetime import datetime, timedelta

from slopometry.core.models import ComplexityDelta, ExtendedComplexityMetrics


class CodeQualityCacheManager:
    """Manages caching of code quality metrics to avoid redundant calculations."""

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize cache manager with database connection.

        Args:
            db_connection: SQLite database connection
        """
        self.db_connection = db_connection

    def get_cached_metrics(
        self, session_id: str, repository_path: str, commit_sha: str, working_tree_hash: str | None = None
    ) -> tuple[ExtendedComplexityMetrics | None, ComplexityDelta | None]:
        """Retrieve cached metrics for a specific session/repo/commit combination.

        Args:
            session_id: Session ID
            repository_path: Absolute path to repository
            commit_sha: Git commit SHA
            working_tree_hash: Working tree state hash for uncommitted changes (None for clean repos)

        Returns:
            Tuple of (complexity_metrics, complexity_delta) or (None, None) if not cached
        """
        try:
            # Handle both clean repos (working_tree_hash=None) and dirty repos (working_tree_hash=hash)
            if working_tree_hash is None:
                cursor = self.db_connection.execute(
                    """
                    SELECT complexity_metrics_json, complexity_delta_json 
                    FROM code_quality_cache 
                    WHERE session_id = ? AND repository_path = ? AND commit_sha = ? AND working_tree_hash IS NULL
                    """,
                    (session_id, repository_path, commit_sha),
                )
            else:
                cursor = self.db_connection.execute(
                    """
                    SELECT complexity_metrics_json, complexity_delta_json 
                    FROM code_quality_cache 
                    WHERE session_id = ? AND repository_path = ? AND commit_sha = ? AND working_tree_hash = ?
                    """,
                    (session_id, repository_path, commit_sha, working_tree_hash),
                )
            row = cursor.fetchone()

            if not row:
                return None, None

            metrics_json, delta_json = row

            metrics_data = json.loads(metrics_json)
            complexity_metrics = ExtendedComplexityMetrics(**metrics_data)

            complexity_delta = None
            if delta_json:
                delta_data = json.loads(delta_json)
                complexity_delta = ComplexityDelta(**delta_data)

            return complexity_metrics, complexity_delta

        except (sqlite3.Error, json.JSONDecodeError, Exception):
            return None, None

    def save_metrics_to_cache(
        self,
        session_id: str,
        repository_path: str,
        commit_sha: str,
        complexity_metrics: ExtendedComplexityMetrics,
        complexity_delta: ComplexityDelta | None = None,
        working_tree_hash: str | None = None,
    ) -> bool:
        """Save complexity metrics to cache.

        Args:
            session_id: Session ID
            repository_path: Absolute path to repository
            commit_sha: Git commit SHA
            complexity_metrics: Complexity metrics to cache
            complexity_delta: Optional complexity delta to cache
            working_tree_hash: Working tree state hash for uncommitted changes (None for clean repos)

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            metrics_json = complexity_metrics.model_dump_json()
            delta_json = complexity_delta.model_dump_json() if complexity_delta else None

            self.db_connection.execute(
                """
                INSERT OR REPLACE INTO code_quality_cache 
                (session_id, repository_path, commit_sha, calculated_at, complexity_metrics_json, complexity_delta_json, working_tree_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    repository_path,
                    commit_sha,
                    datetime.now().isoformat(),
                    metrics_json,
                    delta_json,
                    working_tree_hash,
                ),
            )
            self.db_connection.commit()
            return True

        except (sqlite3.Error, Exception):
            return False

    def is_cache_valid(
        self, repository_path: str, commit_sha: str, working_tree_hash: str | None = None
    ) -> bool:
        """Check if cached metrics are valid for current repository state.

        Args:
            repository_path: Absolute path to repository
            commit_sha: Current git commit SHA
            working_tree_hash: Working tree state hash (None for clean repos)

        Returns:
            True if cache is valid and should be used, False if fresh calculation needed
        """
        # Check if we have a cache entry for this exact state
        try:
            if working_tree_hash is None:
                cursor = self.db_connection.execute(
                    """
                    SELECT COUNT(*) FROM code_quality_cache 
                    WHERE repository_path = ? AND commit_sha = ? AND working_tree_hash IS NULL
                    """,
                    (repository_path, commit_sha),
                )
            else:
                cursor = self.db_connection.execute(
                    """
                    SELECT COUNT(*) FROM code_quality_cache 
                    WHERE repository_path = ? AND commit_sha = ? AND working_tree_hash = ?
                    """,
                    (repository_path, commit_sha, working_tree_hash),
                )
            count = cursor.fetchone()[0]
            return count > 0

        except sqlite3.Error:
            return False

    def cleanup_old_cache_entries(self, days_old: int = 30) -> int:
        """Remove cache entries older than specified days.

        Args:
            days_old: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        try:
            cutoff_date = datetime.now().replace(microsecond=0) - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()

            cursor = self.db_connection.execute(
                "DELETE FROM code_quality_cache WHERE calculated_at < ?",
                (cutoff_iso,),
            )
            self.db_connection.commit()
            return cursor.rowcount

        except sqlite3.Error:
            return 0

    def get_cache_statistics(self) -> dict[str, int]:
        """Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        try:
            cursor = self.db_connection.execute(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT repository_path) as unique_repositories,
                    COUNT(DISTINCT session_id) as unique_sessions
                FROM code_quality_cache
                """
            )
            row = cursor.fetchone()

            return {
                "total_entries": row[0],
                "unique_repositories": row[1],
                "unique_sessions": row[2],
            }

        except sqlite3.Error:
            return {"total_entries": 0, "unique_repositories": 0, "unique_sessions": 0}

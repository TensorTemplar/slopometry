"""Database migrations for slopometry."""

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Migration(ABC):
    """Base class for database migrations."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier for this migration."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this migration."""

    @abstractmethod
    def up(self, conn: sqlite3.Connection) -> None:
        """Apply the migration."""

    def down(self, conn: sqlite3.Connection) -> None:
        """Rollback the migration (optional)."""
        raise NotImplementedError("Rollback not implemented for this migration")


class Migration001AddTranscriptPath(Migration):
    """Add transcript_path column and composite index for performance."""

    @property
    def version(self) -> str:
        return "001"

    @property
    def description(self) -> str:
        return "Add transcript_path column and (session_id, timestamp) composite index"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add transcript_path column and composite index."""
        try:
            conn.execute("ALTER TABLE hook_events ADD COLUMN transcript_path TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hook_events_session_timestamp 
            ON hook_events(session_id, timestamp)
        """)


class Migration002AddCodeQualityCache(Migration):
    """Add code quality cache table for performance optimization."""

    @property
    def version(self) -> str:
        return "002"

    @property
    def description(self) -> str:
        return "Add code quality cache table with indexes for intelligent caching"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add code quality cache table and indexes."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_quality_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                repository_path TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                calculated_at TEXT NOT NULL,
                complexity_metrics_json TEXT NOT NULL,
                complexity_delta_json TEXT,
                UNIQUE(session_id, repository_path, commit_sha)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_code_quality_cache_repo_commit 
            ON code_quality_cache(repository_path, commit_sha)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_code_quality_cache_session 
            ON code_quality_cache(session_id)
        """)


class Migration003AddWorkingTreeHash(Migration):
    """Add working_tree_hash column for uncommitted changes caching."""

    @property
    def version(self) -> str:
        return "003"

    @property
    def description(self) -> str:
        return "Add working_tree_hash column to code_quality_cache for uncommitted changes caching"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add working_tree_hash column and update constraints."""

        try:
            conn.execute("ALTER TABLE code_quality_cache ADD COLUMN working_tree_hash TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        # WORKAROUND: SQLite doesn't support DROP CONSTRAINT, so we recreate the
        # table for empty DBs or skip constraint changes for existing data.
        cursor = conn.execute("SELECT COUNT(*) FROM code_quality_cache")
        row_count = cursor.fetchone()[0]

        if row_count == 0:
            conn.execute("DROP TABLE code_quality_cache")
            conn.execute("""
                CREATE TABLE code_quality_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    calculated_at TEXT NOT NULL,
                    complexity_metrics_json TEXT NOT NULL,
                    complexity_delta_json TEXT,
                    working_tree_hash TEXT,
                    UNIQUE(session_id, repository_path, commit_sha, working_tree_hash)
                )
            """)

            conn.execute("""
                CREATE INDEX idx_code_quality_cache_repo_commit 
                ON code_quality_cache(repository_path, commit_sha)
            """)
            conn.execute("""
                CREATE INDEX idx_code_quality_cache_session 
                ON code_quality_cache(session_id)
            """)
        else:
            # NOTE: Existing data preserved - working_tree_hash=NULL represents clean repos.
            # Cache manager handles dirty repo constraint logic separately.
            pass


class Migration004AddCalculatorVersion(Migration):
    """Add calculator_version column for cache invalidation."""

    @property
    def version(self) -> str:
        return "004"

    @property
    def description(self) -> str:
        return "Add calculator_version column to code_quality_cache for versioned metrics"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add calculator_version column."""

        try:
            conn.execute("ALTER TABLE code_quality_cache ADD COLUMN calculator_version TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        # WORKAROUND: Can't drop SQLite constraints without losing user data.
        # Cache manager enforces version validation in code instead.


class Migration005AddGalenRateColumns(Migration):
    """Add Galen Rate columns to repo_baselines for token productivity tracking."""

    @property
    def version(self) -> str:
        return "005"

    @property
    def description(self) -> str:
        return "Add oldest_commit_date, newest_commit_date, oldest_commit_tokens to repo_baselines"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add Galen Rate columns to repo_baselines."""
        # Check if table exists first (it's created by EventDatabase, not migrations)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_baselines'")
        if not cursor.fetchone():
            return  # Table doesn't exist yet, skip migration

        columns = [
            ("oldest_commit_date", "TEXT"),
            ("newest_commit_date", "TEXT"),
            ("oldest_commit_tokens", "INTEGER"),
        ]

        for column_name, column_type in columns:
            try:
                conn.execute(f"ALTER TABLE repo_baselines ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise


class Migration006AddQPEColumns(Migration):
    """Add QPE (Quality-Per-Effort) columns to experiment_progress."""

    @property
    def version(self) -> str:
        return "006"

    @property
    def description(self) -> str:
        return "Add qpe_score, smell_penalty, effort_tier columns to experiment_progress"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add QPE columns to experiment_progress table."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_progress'")
        if not cursor.fetchone():
            return  # Table doesn't exist yet, skip migration

        columns = [
            ("qpe_score", "REAL"),
            ("smell_penalty", "REAL"),
            ("effort_tier", "TEXT"),
        ]

        for column_name, column_type in columns:
            try:
                conn.execute(f"ALTER TABLE experiment_progress ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

        conn.execute("CREATE INDEX IF NOT EXISTS idx_progress_qpe ON experiment_progress(experiment_id, qpe_score)")


class Migration007AddQPELeaderboard(Migration):
    """Add QPE leaderboard table for cross-project comparison persistence."""

    @property
    def version(self) -> str:
        return "007"

    @property
    def description(self) -> str:
        return "Add qpe_leaderboard table for persistent cross-project comparison"

    def up(self, conn: sqlite3.Connection) -> None:
        """Create qpe_leaderboard table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qpe_leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_path TEXT NOT NULL,
                commit_sha_short TEXT NOT NULL,
                commit_sha_full TEXT NOT NULL,
                measured_at TEXT NOT NULL,
                qpe_score REAL NOT NULL,
                mi_normalized REAL NOT NULL,
                smell_penalty REAL NOT NULL,
                adjusted_quality REAL NOT NULL,
                effort_factor REAL NOT NULL,
                total_effort REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                UNIQUE(project_path, commit_sha_full)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_qpe ON qpe_leaderboard(qpe_score DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_project ON qpe_leaderboard(project_path, measured_at)")


class Migration008FixLeaderboardUniqueConstraint(Migration):
    """Fix leaderboard unique constraint to allow updating entries when commit changes.

    The original constraint UNIQUE(project_path, commit_sha_full) prevented updates
    when a project's commit changed (e.g., after git pull). This migration changes
    the constraint to just UNIQUE(project_path) so --append refreshes existing entries.
    """

    @property
    def version(self) -> str:
        return "008"

    @property
    def description(self) -> str:
        return "Fix qpe_leaderboard unique constraint to project_path only"

    def up(self, conn: sqlite3.Connection) -> None:
        """Recreate qpe_leaderboard with corrected unique constraint."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='qpe_leaderboard'")
        if not cursor.fetchone():
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS qpe_leaderboard_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_path TEXT NOT NULL UNIQUE,
                commit_sha_short TEXT NOT NULL,
                commit_sha_full TEXT NOT NULL,
                measured_at TEXT NOT NULL,
                qpe_score REAL NOT NULL,
                mi_normalized REAL NOT NULL,
                smell_penalty REAL NOT NULL,
                adjusted_quality REAL NOT NULL,
                effort_factor REAL NOT NULL,
                total_effort REAL NOT NULL,
                metrics_json TEXT NOT NULL
            )
        """)

        conn.execute("""
            INSERT OR REPLACE INTO qpe_leaderboard_new (
                project_name, project_path, commit_sha_short, commit_sha_full,
                measured_at, qpe_score, mi_normalized, smell_penalty,
                adjusted_quality, effort_factor, total_effort, metrics_json
            )
            SELECT project_name, project_path, commit_sha_short, commit_sha_full,
                   measured_at, qpe_score, mi_normalized, smell_penalty,
                   adjusted_quality, effort_factor, total_effort, metrics_json
            FROM qpe_leaderboard
            GROUP BY project_path
            HAVING measured_at = MAX(measured_at)
        """)

        conn.execute("DROP TABLE qpe_leaderboard")
        conn.execute("ALTER TABLE qpe_leaderboard_new RENAME TO qpe_leaderboard")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_qpe ON qpe_leaderboard(qpe_score DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_leaderboard_project ON qpe_leaderboard(project_path, measured_at)")


class Migration009AddBaselineStrategyColumn(Migration):
    """Add strategy_json column to repo_baselines for baseline strategy tracking."""

    @property
    def version(self) -> str:
        return "009"

    @property
    def description(self) -> str:
        return "Add strategy_json column to repo_baselines for baseline strategy tracking"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add strategy_json column to repo_baselines."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_baselines'")
        if not cursor.fetchone():
            return

        try:
            conn.execute("ALTER TABLE repo_baselines ADD COLUMN strategy_json TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise


class Migration010AddBaselineQPEColumns(Migration):
    """Add QPE stats and current QPE columns to repo_baselines.

    These columns were missing, causing the baseline cache to always appear stale
    (qpe_stats=None after load triggers recomputation on every run).
    """

    @property
    def version(self) -> str:
        return "010"

    @property
    def description(self) -> str:
        return "Add qpe_stats_json and current_qpe_json columns to repo_baselines for cache completeness"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add QPE columns to repo_baselines."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repo_baselines'")
        if not cursor.fetchone():
            return

        for column_name in ("qpe_stats_json", "current_qpe_json"):
            try:
                conn.execute(f"ALTER TABLE repo_baselines ADD COLUMN {column_name} TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise


class Migration011AddQPEWeightVersionColumn(Migration):
    """Add qpe_weight_version column to qpe_leaderboard and repo_baselines.

    Tracks which QPE_WEIGHT_VERSION was used to compute cached scores.
    Entries with NULL or mismatched versions trigger a warning and recomputation.
    """

    @property
    def version(self) -> str:
        return "011"

    @property
    def description(self) -> str:
        return "Add qpe_weight_version column to qpe_leaderboard and repo_baselines"

    def up(self, conn: sqlite3.Connection) -> None:
        for table in ("qpe_leaderboard", "repo_baselines"):
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not cursor.fetchone():
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN qpe_weight_version TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise


class Migration012AddNFPObjectiveToExperimentRuns(Migration):
    """Add nfp_objective_id column to experiment_runs.

    Previously added via a try/except ALTER TABLE in _ensure_tables.
    Moved to a proper migration for consistency.
    """

    @property
    def version(self) -> str:
        return "012"

    @property
    def description(self) -> str:
        return "Add nfp_objective_id column to experiment_runs"

    def up(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_runs'")
        if not cursor.fetchone():
            return
        try:
            conn.execute("""
                ALTER TABLE experiment_runs
                ADD COLUMN nfp_objective_id TEXT REFERENCES nfp_objectives(id)
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise


class Migration013AddSourceAndParentSession(Migration):
    """Add source and parent_session_id columns to hook_events for OpenCode integration.

    The source column distinguishes events from different agent tools (claude_code vs opencode).
    The parent_session_id column tracks subagent session relationships in OpenCode.
    """

    @property
    def version(self) -> str:
        return "013"

    @property
    def description(self) -> str:
        return "Add source and parent_session_id columns to hook_events for OpenCode integration"

    def up(self, conn: sqlite3.Connection) -> None:
        """Add source and parent_session_id columns."""
        columns = [
            ("source", "TEXT DEFAULT 'claude_code'"),
            ("parent_session_id", "TEXT"),
        ]

        for column_name, column_def in columns:
            try:
                conn.execute(f"ALTER TABLE hook_events ADD COLUMN {column_name} {column_def}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hook_events_source ON hook_events(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hook_events_parent_session ON hook_events(parent_session_id)"
        )


class MigrationRunner:
    """Manages database migrations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.migrations = [
            Migration001AddTranscriptPath(),
            Migration002AddCodeQualityCache(),
            Migration003AddWorkingTreeHash(),
            Migration004AddCalculatorVersion(),
            Migration005AddGalenRateColumns(),
            Migration006AddQPEColumns(),
            Migration007AddQPELeaderboard(),
            Migration008FixLeaderboardUniqueConstraint(),
            Migration009AddBaselineStrategyColumn(),
            Migration010AddBaselineQPEColumns(),
            Migration011AddQPEWeightVersionColumn(),
            Migration012AddNFPObjectiveToExperimentRuns(),
            Migration013AddSourceAndParentSession(),
        ]

    @contextmanager
    def _get_db_connection(self) -> Generator[sqlite3.Connection]:
        """Context manager that ensures database connections are properly closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_migrations_table(self, conn: sqlite3.Connection) -> None:
        """Create migrations table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _slopometry_migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)

    def _is_migration_applied(self, conn: sqlite3.Connection, version: str) -> bool:
        """Check if a migration has been applied."""
        cursor = conn.execute("SELECT 1 FROM _slopometry_migrations WHERE version = ?", (version,))
        return cursor.fetchone() is not None

    def _mark_migration_applied(self, conn: sqlite3.Connection, migration: Migration) -> None:
        """Mark a migration as applied."""
        from datetime import datetime

        conn.execute(
            """
            INSERT INTO _slopometry_migrations (version, description, applied_at)
            VALUES (?, ?, ?)
            """,
            (migration.version, migration.description, datetime.now().isoformat()),
        )

    def run_migrations(self) -> list[str]:
        """Run all pending migrations."""
        applied_migrations = []

        with self._get_db_connection() as conn:
            self._ensure_migrations_table(conn)

            for migration in self.migrations:
                if not self._is_migration_applied(conn, migration.version):
                    migration.up(conn)
                    self._mark_migration_applied(conn, migration)
                    applied_migrations.append(f"{migration.version}: {migration.description}")

        return applied_migrations

    def get_migration_status(self) -> dict[str, Any]:
        """Get status of all migrations."""
        status = {"applied": [], "pending": [], "total": len(self.migrations)}

        with self._get_db_connection() as conn:
            self._ensure_migrations_table(conn)

            for migration in self.migrations:
                migration_info = {"version": migration.version, "description": migration.description}

                if self._is_migration_applied(conn, migration.version):
                    cursor = conn.execute(
                        "SELECT applied_at FROM _slopometry_migrations WHERE version = ?", (migration.version,)
                    )
                    applied_at = cursor.fetchone()[0]
                    migration_info["applied_at"] = applied_at
                    status["applied"].append(migration_info)
                else:
                    status["pending"].append(migration_info)

        return status

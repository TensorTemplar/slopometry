"""Database migrations for slopometry."""

import sqlite3
from abc import ABC, abstractmethod
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
        # Add transcript_path column if it doesn't exist
        try:
            conn.execute("ALTER TABLE hook_events ADD COLUMN transcript_path TEXT")
        except sqlite3.OperationalError as e:
            # Column already exists
            if "duplicate column name" not in str(e).lower():
                raise

        # Add composite index if it doesn't exist
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hook_events_session_timestamp 
            ON hook_events(session_id, timestamp)
        """)


class MigrationRunner:
    """Manages database migrations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.migrations = [
            Migration001AddTranscriptPath(),
        ]

    def _get_db_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

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
                    # Get applied timestamp
                    cursor = conn.execute(
                        "SELECT applied_at FROM _slopometry_migrations WHERE version = ?", (migration.version,)
                    )
                    applied_at = cursor.fetchone()[0]
                    migration_info["applied_at"] = applied_at
                    status["applied"].append(migration_info)
                else:
                    status["pending"].append(migration_info)

        return status

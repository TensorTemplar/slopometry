"""Tests for database migrations."""

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from slopometry.core.migrations import MigrationRunner

# Derive expected count from the runner itself so adding a migration doesn't
# require updating every assertion in these tests.
EXPECTED_MIGRATION_COUNT = len(MigrationRunner(Path("/dev/null")).migrations)


class TestMigrations:
    """Test database migration functionality."""

    def test_migration_001__adds_transcript_path_column_and_index(self):
        """Test that migration 001 adds the transcript_path column and composite index."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            runner = MigrationRunner(db_path)

            with runner._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE hook_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                conn.commit()

            applied = runner.run_migrations()

            assert len(applied) == EXPECTED_MIGRATION_COUNT
            assert any("001" in migration and "transcript_path" in migration for migration in applied)
            assert any("002" in migration and "code quality cache" in migration for migration in applied)
            assert any("003" in migration and "working_tree_hash" in migration for migration in applied)
            assert any("004" in migration and "calculator_version" in migration for migration in applied)
            assert any("005" in migration and "oldest_commit" in migration for migration in applied)
            assert any("006" in migration and "qpe_score" in migration for migration in applied)
            assert any("007" in migration and "qpe_leaderboard" in migration for migration in applied)
            assert any("008" in migration and "unique constraint" in migration for migration in applied)

            with runner._get_db_connection() as conn:
                cursor = conn.execute("PRAGMA table_info(hook_events)")
                columns = [row[1] for row in cursor.fetchall()]
                assert "transcript_path" in columns

                cursor = conn.execute("PRAGMA index_list(hook_events)")
                indexes = [row[1] for row in cursor.fetchall()]
                assert "idx_hook_events_session_timestamp" in indexes

    def test_migration_runner__idempotent_execution(self):
        """Test that running migrations multiple times is safe."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            runner = MigrationRunner(db_path)

            with runner._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE hook_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                conn.commit()

            applied_first = runner.run_migrations()
            applied_second = runner.run_migrations()

            assert len(applied_first) == EXPECTED_MIGRATION_COUNT
            assert len(applied_second) == 0

            status = runner.get_migration_status()
            assert status["total"] == EXPECTED_MIGRATION_COUNT
            assert len(status["applied"]) == EXPECTED_MIGRATION_COUNT
            assert len(status["pending"]) == 0

    def test_migration_runner__tracks_migration_status(self):
        """Test that migration status tracking works correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            runner = MigrationRunner(db_path)

            with runner._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE hook_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                conn.commit()

            status_before = runner.get_migration_status()

            runner.run_migrations()

            status_after = runner.get_migration_status()

            assert status_before["total"] == EXPECTED_MIGRATION_COUNT
            assert len(status_before["applied"]) == 0
            assert len(status_before["pending"]) == EXPECTED_MIGRATION_COUNT

            assert status_after["total"] == EXPECTED_MIGRATION_COUNT
            assert len(status_after["applied"]) == EXPECTED_MIGRATION_COUNT
            assert len(status_after["pending"]) == 0

            migration_001 = next((m for m in status_after["applied"] if m["version"] == "001"), None)
            assert migration_001 is not None
            assert "applied_at" in migration_001

    def test_migration_001__handles_existing_column_gracefully(self):
        """Test that migration 001 handles existing transcript_path column."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            runner = MigrationRunner(db_path)

            with runner._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE hook_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        transcript_path TEXT
                    )
                """)
                conn.commit()

            applied = runner.run_migrations()

            assert len(applied) == EXPECTED_MIGRATION_COUNT

            with runner._get_db_connection() as conn:
                cursor = conn.execute("PRAGMA table_info(hook_events)")
                columns = [row[1] for row in cursor.fetchall()]
                transcript_path_count = columns.count("transcript_path")
                assert transcript_path_count == 1

    def test_migration_019__remaps_harness_and_alt_dialect_event_types_and_enables_idempotent_ingest(self):
        """Legacy and alt-dialect event types become canonical kinds with an event_id index."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            runner = MigrationRunner(db_path)

            with runner._get_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE hook_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT
                    )
                """)
                conn.execute(
                    "INSERT INTO hook_events (session_id, event_type, timestamp, source) VALUES ('s1', 'PreToolUse', '2026-01-01T00:00:00', NULL)"
                )
                conn.execute(
                    "INSERT INTO hook_events (session_id, event_type, timestamp, source) VALUES ('s1', 'SubagentStop', '2026-01-01T00:01:00', 'opencode')"
                )
                conn.execute(
                    "INSERT INTO hook_events (session_id, event_type, timestamp, source) VALUES ('s1', 'turn_completed', '2026-01-01T00:02:00', 'opencode')"
                )
                conn.execute(
                    "INSERT INTO hook_events (session_id, event_type, timestamp, source) VALUES ('s1', 'tool_call_completed', '2026-01-01T00:03:00', 'opencode')"
                )
                conn.commit()

            runner.run_migrations()

            with runner._get_db_connection() as conn:
                rows = conn.execute("SELECT event_type, source FROM hook_events ORDER BY id").fetchall()
                assert rows == [
                    ("tool_call", "claude_code"),
                    ("subagent_stop", "opencode"),
                    ("stop", "opencode"),
                    ("tool_result", "opencode"),
                ]

                columns = {row[1] for row in conn.execute("PRAGMA table_info(hook_events)").fetchall()}
                indexes = {row[1] for row in conn.execute("PRAGMA index_list(hook_events)").fetchall()}
                assert "event_id" in columns
                assert "idx_hook_events_source_event_id" in indexes

                conn.execute(
                    "INSERT INTO hook_events (session_id, event_type, timestamp, source, event_id) VALUES ('s2', 'stop', '2026-01-01T02:00:00', 'mmkr', 'evt-1')"
                )
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO hook_events (session_id, event_type, timestamp, source, event_id) VALUES ('s2', 'stop', '2026-01-01T02:01:00', 'mmkr', 'evt-1')"
                    )

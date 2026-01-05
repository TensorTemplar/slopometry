"""Tests for database migrations."""

from pathlib import Path
from tempfile import TemporaryDirectory

from slopometry.core.migrations import MigrationRunner


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

            assert len(applied) == 7
            assert any("001" in migration and "transcript_path" in migration for migration in applied)
            assert any("002" in migration and "code quality cache" in migration for migration in applied)
            assert any("003" in migration and "working_tree_hash" in migration for migration in applied)
            assert any("004" in migration and "calculator_version" in migration for migration in applied)
            assert any("005" in migration and "oldest_commit" in migration for migration in applied)
            assert any("006" in migration and "qpe_score" in migration for migration in applied)
            assert any("007" in migration and "qpe_leaderboard" in migration for migration in applied)

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

            assert len(applied_first) == 7
            assert len(applied_second) == 0

            status = runner.get_migration_status()
            assert status["total"] == 7
            assert len(status["applied"]) == 7
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

            assert status_before["total"] == 7
            assert len(status_before["applied"]) == 0
            assert len(status_before["pending"]) == 7

            assert status_after["total"] == 7
            assert len(status_after["applied"]) == 7
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

            assert len(applied) == 7

            with runner._get_db_connection() as conn:
                cursor = conn.execute("PRAGMA table_info(hook_events)")
                columns = [row[1] for row in cursor.fetchall()]
                transcript_path_count = columns.count("transcript_path")
                assert transcript_path_count == 1

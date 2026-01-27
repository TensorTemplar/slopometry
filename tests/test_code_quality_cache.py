import sqlite3
from datetime import datetime

import pytest
from conftest import make_test_metrics

from slopometry.core.code_quality_cache import CodeQualityCacheManager
from slopometry.core.models import ExtendedComplexityMetrics


class TestCodeQualityCacheManager:
    """Tests for CodeQualityCacheManager."""

    @pytest.fixture
    def db_connection(self):
        """Proof-of-concept fixture for in-memory DB with correct schema."""
        conn = sqlite3.connect(":memory:")
        # Create schema manually to ensure it matches current migration state
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
                calculator_version TEXT,
                UNIQUE(session_id, repository_path, commit_sha, working_tree_hash, calculator_version)
            )
        """)
        yield conn
        conn.close()

    def test_cache_miss_for_missing_entry(self, db_connection):
        """Test that get_cached_metrics returns None if no entry exists."""
        manager = CodeQualityCacheManager(db_connection)
        metrics, delta = manager.get_cached_metrics("sess_1", "/repo", "sha1")
        assert metrics is None
        assert delta is None

    def test_save_and_retrieve_cache_hit(self, db_connection):
        """Test that saved metrics can be retrieved correctly."""
        manager = CodeQualityCacheManager(db_connection)
        dummy_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=42))

        success = manager.save_metrics_to_cache("sess_1", "/repo", "sha1", dummy_metrics)
        assert success is True

        metrics, delta = manager.get_cached_metrics("sess_1", "/repo", "sha1")
        assert metrics is not None
        assert metrics.total_complexity == 42
        assert delta is None

        assert manager.is_cache_valid("/repo", "sha1") is True

    def test_version_mismatch_invalidation(self, db_connection):
        """Test that entries with missing or different calculator_version are ignored."""
        manager = CodeQualityCacheManager(db_connection)
        dummy_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=99))
        metrics_json = dummy_metrics.model_dump_json()

        # Insert a stale record (simulate old migration state or different version)
        db_connection.execute(
            """
            INSERT INTO code_quality_cache 
            (session_id, repository_path, commit_sha, calculated_at, complexity_metrics_json, complexity_delta_json, working_tree_hash, calculator_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("sess_1", "/repo", "sha1", datetime.now().isoformat(), metrics_json, None, None, "OLD_VERSION"),
        )
        db_connection.commit()

        # Verify it is NOT retrieved
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1")
        assert metrics is None
        assert manager.is_cache_valid("/repo", "sha1") is False

        # Insert a NULL version record
        db_connection.execute(
            """
            INSERT INTO code_quality_cache 
            (session_id, repository_path, commit_sha, calculated_at, complexity_metrics_json, complexity_delta_json, working_tree_hash, calculator_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("sess_2", "/repo", "sha1", datetime.now().isoformat(), metrics_json, None, None, None),
        )
        db_connection.commit()

        # Verify it is NOT retrieved
        metrics, _ = manager.get_cached_metrics("sess_2", "/repo", "sha1")
        assert metrics is None

    def test_working_tree_hash_support(self, db_connection):
        """Test that working_tree_hash is correctly used in cache keys."""
        manager = CodeQualityCacheManager(db_connection)
        dummy_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=100))

        # Save with a specific working tree hash
        manager.save_metrics_to_cache("sess_1", "/repo", "sha1", dummy_metrics, working_tree_hash="hash123")

        # Retrieve with correct hash
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1", working_tree_hash="hash123")
        assert metrics is not None
        assert metrics.total_complexity == 100

        # Retrieve with wrong hash -> Miss
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1", working_tree_hash="hash999")
        assert metrics is None

        # Retrieve clean (None hash) -> Miss
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1", working_tree_hash=None)
        assert metrics is None

    def test_update_cached_coverage__updates_existing_entry(self, db_connection):
        """Test that update_cached_coverage updates coverage fields in cached metrics."""
        manager = CodeQualityCacheManager(db_connection)
        dummy_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=50))

        # Save initial metrics without coverage
        manager.save_metrics_to_cache("sess_1", "/repo", "sha1", dummy_metrics)

        # Verify initial state has no coverage
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1")
        assert metrics is not None
        assert metrics.test_coverage_percent is None

        # Update with coverage
        success = manager.update_cached_coverage("sess_1", 85.5, "coverage.xml")
        assert success is True

        # Verify coverage was cached
        metrics, _ = manager.get_cached_metrics("sess_1", "/repo", "sha1")
        assert metrics is not None
        assert metrics.test_coverage_percent == 85.5
        assert metrics.test_coverage_source == "coverage.xml"

    def test_update_cached_coverage__returns_false_for_missing_session(self, db_connection):
        """Test that update_cached_coverage returns False if session doesn't exist."""
        manager = CodeQualityCacheManager(db_connection)
        success = manager.update_cached_coverage("nonexistent", 75.0, "coverage.xml")
        assert success is False

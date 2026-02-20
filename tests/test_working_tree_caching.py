"""Tests for working tree state caching functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from conftest import make_test_metrics

from slopometry.core.code_quality_cache import CodeQualityCacheManager
from slopometry.core.database import EventDatabase
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics


class TestWorkingTreeCacheManager:
    """Test the enhanced CodeQualityCacheManager with working tree support."""

    def test_cache_clean_repo_metrics(self):
        """Test caching metrics for clean repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=50))
            test_delta = ComplexityDelta(total_complexity_change=5)

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Save metrics for clean repo (working_tree_hash=None)
                success = cache_manager.save_metrics_to_cache(
                    session_id="session1",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    complexity_metrics=test_metrics,
                    complexity_delta=test_delta,
                    working_tree_hash=None,
                )
                assert success

                # Retrieve metrics for clean repo
                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    session_id="session1", repository_path="/test/repo", commit_sha="abc123", working_tree_hash=None
                )

                assert cached_metrics is not None
                assert cached_delta is not None
                assert cached_metrics.total_complexity == 50
                assert cached_delta.total_complexity_change == 5

    def test_cache_dirty_repo_metrics(self):
        """Test caching metrics for dirty repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=60))
            test_delta = ComplexityDelta(total_complexity_change=10)

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Save metrics for dirty repo (working_tree_hash=hash)
                success = cache_manager.save_metrics_to_cache(
                    session_id="session1",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    complexity_metrics=test_metrics,
                    complexity_delta=test_delta,
                    working_tree_hash="deadbeef12345678",
                )
                assert success

                # Retrieve metrics for dirty repo with same hash
                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    session_id="session1",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    working_tree_hash="deadbeef12345678",
                )

                assert cached_metrics is not None
                assert cached_delta is not None
                assert cached_metrics.total_complexity == 60
                assert cached_delta.total_complexity_change == 10

    def test_cache_miss_different_working_tree_hash(self):
        """Test cache miss when working tree hash is different."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=40))

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Save metrics with one working tree hash
                cache_manager.save_metrics_to_cache("session1", "/test/repo", "abc123", test_metrics, None, "hash1")

                # Try to retrieve with different working tree hash - should be cache miss
                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    "session1", "/test/repo", "abc123", "hash2"
                )

                assert cached_metrics is None
                assert cached_delta is None

    def test_clean_and_dirty_repo_coexistence(self):
        """Test that clean and dirty repo cache entries can coexist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            clean_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=30))
            dirty_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=35))

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Save metrics for clean repo
                cache_manager.save_metrics_to_cache("session1", "/test/repo", "abc123", clean_metrics, None, None)

                # Save metrics for dirty repo (same session, repo, commit)
                cache_manager.save_metrics_to_cache(
                    "session1", "/test/repo", "abc123", dirty_metrics, None, "dirtyhash"
                )

                # Retrieve clean repo metrics
                clean_cached, _ = cache_manager.get_cached_metrics("session1", "/test/repo", "abc123", None)

                # Retrieve dirty repo metrics
                dirty_cached, _ = cache_manager.get_cached_metrics("session1", "/test/repo", "abc123", "dirtyhash")

                assert clean_cached is not None
                assert dirty_cached is not None
                assert clean_cached.total_complexity == 30
                assert dirty_cached.total_complexity == 35

    def test_is_cache_valid_for_different_states(self):
        """Test cache validity checking for different repository states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics(**make_test_metrics())

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Cache entry for clean repo
                cache_manager.save_metrics_to_cache("session1", "/test/repo", "abc123", test_metrics, None, None)

                # Cache entry for dirty repo
                cache_manager.save_metrics_to_cache(
                    "session1", "/test/repo", "abc123", test_metrics, None, "workingHash"
                )

                # Test validity checks
                assert cache_manager.is_cache_valid("/test/repo", "abc123", None) is True
                assert cache_manager.is_cache_valid("/test/repo", "abc123", "workingHash") is True
                assert cache_manager.is_cache_valid("/test/repo", "abc123", "differentHash") is False
                assert cache_manager.is_cache_valid("/test/repo", "def456", None) is False


class TestWorkingTreeCachingIntegration:
    """Integration tests for the full working tree caching system."""

    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator.get_current_commit_sha")
    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator.has_uncommitted_changes")
    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator.calculate_working_tree_hash")
    def test_session_complexity_metrics_dirty_repo_caching(self, mock_calc_hash, mock_uncommitted, mock_get_sha):
        """Test that _get_session_complexity_metrics caches dirty repo metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            # Mock git state for dirty repo
            mock_get_sha.return_value = "abc123"
            mock_uncommitted.return_value = True
            mock_calc_hash.return_value = "workingTreeHash"

            # Mock the actual complexity calculation
            with patch.object(db, "calculate_extended_complexity_metrics") as mock_calc:
                expected_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=75))
                expected_delta = ComplexityDelta(total_complexity_change=25)
                mock_calc.return_value = (expected_metrics, expected_delta)

                # First call - should calculate and cache
                metrics1, delta1 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics1 is not None
                assert delta1 is not None
                assert metrics1.total_complexity == 75
                assert delta1.total_complexity_change == 25
                mock_calc.assert_called_once()

                # Second call with same working tree state - should hit cache
                metrics2, delta2 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics2 is not None
                assert delta2 is not None
                assert metrics2.total_complexity == 75
                assert delta2.total_complexity_change == 25
                # calculate_extended_complexity_metrics should still only be called once
                assert mock_calc.call_count == 1

    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator.get_current_commit_sha")
    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator.has_uncommitted_changes")
    def test_session_complexity_metrics_clean_repo_caching(self, mock_uncommitted, mock_get_sha):
        """Test that _get_session_complexity_metrics still works for clean repos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            # Mock git state for clean repo
            mock_get_sha.return_value = "abc123"
            mock_uncommitted.return_value = False

            # Mock the actual complexity calculation
            with patch.object(db, "calculate_extended_complexity_metrics") as mock_calc:
                expected_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=50))
                expected_delta = ComplexityDelta(total_complexity_change=0)
                mock_calc.return_value = (expected_metrics, expected_delta)

                # First call - should calculate and cache
                metrics1, delta1 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics1 is not None
                assert delta1 is not None
                assert metrics1.total_complexity == 50
                assert delta1.total_complexity_change == 0
                mock_calc.assert_called_once()

                # Second call - should hit cache
                metrics2, delta2 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics2 is not None
                assert delta2 is not None
                assert metrics2.total_complexity == 50
                assert delta2.total_complexity_change == 0
                # Should still only be called once due to caching
                assert mock_calc.call_count == 1

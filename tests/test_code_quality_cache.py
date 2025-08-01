"""Tests for code quality caching functionality."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from slopometry.core.code_quality_cache import CodeQualityCacheManager
from slopometry.core.database import EventDatabase
from slopometry.core.models import ComplexityDelta, ExtendedComplexityMetrics


class TestCodeQualityCache:
    """Test code quality caching functionality."""

    def test_cache_manager_save_and_retrieve(self):
        """Test that cache manager can save and retrieve metrics correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            metrics = ExtendedComplexityMetrics(
                total_complexity=100,
                average_complexity=10.0,
                max_complexity=20,
                min_complexity=5,
                total_files_analyzed=10,
            )

            delta = ComplexityDelta(
                total_complexity_change=5,
                files_added=["new_file.py"],
                files_removed=[],
                files_changed={"existing_file.py": 2},
                net_files_change=1,
                avg_complexity_change=0.5,
            )

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                success = cache_manager.save_metrics_to_cache(
                    session_id="test_session",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    complexity_metrics=metrics,
                    complexity_delta=delta,
                )

                assert success

                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    session_id="test_session",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                )

                assert cached_metrics is not None
                assert cached_delta is not None
                assert cached_metrics.total_complexity == 100
                assert cached_metrics.average_complexity == 10.0
                assert cached_delta.total_complexity_change == 5
                assert cached_delta.files_added == ["new_file.py"]

    def test_cache_manager_cache_miss(self):
        """Test that cache manager returns None for cache misses."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    session_id="nonexistent_session",
                    repository_path="/nonexistent/repo",
                    commit_sha="nonexistent_commit",
                )

                assert cached_metrics is None
                assert cached_delta is None

    def test_cache_validity_check(self):
        """Test that cache validity check works correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                # Clean repo (working_tree_hash=None) - should not be valid initially
                assert not cache_manager.is_cache_valid(
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    working_tree_hash=None,
                )

                # Dirty repo (working_tree_hash="some_hash") - should not be valid initially
                assert not cache_manager.is_cache_valid(
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    working_tree_hash="some_working_tree_hash",
                )

                # Save metrics for clean repo
                cache_manager.save_metrics_to_cache(
                    session_id="test_session",
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    complexity_metrics=ExtendedComplexityMetrics(),
                    complexity_delta=None,
                    working_tree_hash=None,
                )

                # Clean repo should now be valid
                assert cache_manager.is_cache_valid(
                    repository_path="/test/repo",
                    commit_sha="abc123",
                    working_tree_hash=None,
                )

                # Dirty repo should still not be valid
                assert not cache_manager.is_cache_valid(
                    repository_path="/test/repo",
                    commit_sha="abc123", 
                    working_tree_hash="some_working_tree_hash",
                )

    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator")
    def test_session_complexity_metrics_uses_cache(self, mock_working_tree_calculator):
        """Test that _get_session_complexity_metrics uses cache when available."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            # Mock working tree calculator for clean repo
            mock_calculator_instance = Mock()
            mock_calculator_instance.get_current_commit_sha.return_value = "abc123"
            mock_calculator_instance.has_uncommitted_changes.return_value = False
            mock_working_tree_calculator.return_value = mock_calculator_instance

            test_metrics = ExtendedComplexityMetrics(total_complexity=42)
            test_delta = ComplexityDelta(total_complexity_change=1)

            # Pre-populate cache with clean repo entry (working_tree_hash=None)
            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)
                cache_manager.save_metrics_to_cache(
                    session_id="test_session",
                    repository_path=str(Path(temp_dir).resolve()),
                    commit_sha="abc123",
                    complexity_metrics=test_metrics,
                    complexity_delta=test_delta,
                    working_tree_hash=None,  # Clean repo
                )

            # Should use cache and not call calculate_extended_complexity_metrics
            metrics, delta = db._get_session_complexity_metrics(
                session_id="test_session",
                working_directory=temp_dir,
            )

            assert metrics is not None
            assert delta is not None
            assert metrics.total_complexity == 42
            assert delta.total_complexity_change == 1

    @patch("slopometry.core.working_tree_state.WorkingTreeStateCalculator")
    def test_session_complexity_metrics_recalculates_with_uncommitted_changes(self, mock_working_tree_calculator):
        """Test that _get_session_complexity_metrics recalculates when there are uncommitted changes."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            # Mock working tree calculator for dirty repo
            mock_calculator_instance = Mock()
            mock_calculator_instance.get_current_commit_sha.return_value = "abc123"
            mock_calculator_instance.has_uncommitted_changes.return_value = True
            mock_calculator_instance.calculate_working_tree_hash.return_value = "dirty_tree_hash"
            mock_working_tree_calculator.return_value = mock_calculator_instance

            with patch.object(db, "calculate_extended_complexity_metrics") as mock_calc:
                mock_calc.return_value = (
                    ExtendedComplexityMetrics(total_complexity=99),
                    ComplexityDelta(total_complexity_change=5),
                )

                metrics, delta = db._get_session_complexity_metrics(
                    session_id="test_session",
                    working_directory=temp_dir,
                )

                assert metrics is not None
                assert delta is not None
                assert metrics.total_complexity == 99
                assert delta.total_complexity_change == 5
                mock_calc.assert_called_once_with(temp_dir)

    def test_cache_statistics(self):
        """Test that cache statistics are calculated correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            with db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                stats = cache_manager.get_cache_statistics()
                assert stats["total_entries"] == 0
                assert stats["unique_repositories"] == 0
                assert stats["unique_sessions"] == 0

                test_metrics = ExtendedComplexityMetrics()

                cache_manager.save_metrics_to_cache("session1", "/repo1", "commit1", test_metrics)
                cache_manager.save_metrics_to_cache("session1", "/repo1", "commit2", test_metrics)
                cache_manager.save_metrics_to_cache("session2", "/repo2", "commit1", test_metrics)

                stats = cache_manager.get_cache_statistics()
                assert stats["total_entries"] == 3
                assert stats["unique_repositories"] == 2
                assert stats["unique_sessions"] == 2

"""Tests for working tree state caching functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from slopometry.core.code_quality_cache import CodeQualityCacheManager
from slopometry.core.database import EventDatabase
from slopometry.core.models import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.working_tree_state import WorkingTreeStateCalculator


class TestWorkingTreeStateCalculator:
    """Test the WorkingTreeStateCalculator class."""

    def test_calculate_working_tree_hash_with_no_files(self):
        """Test working tree hash calculation with no Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            hash_result = calculator.calculate_working_tree_hash("abc123")

            assert isinstance(hash_result, str)
            assert len(hash_result) == 16  # SHA256 truncated to 16 chars

            # Same inputs should produce same hash
            hash_result2 = calculator.calculate_working_tree_hash("abc123")
            assert hash_result == hash_result2

    def test_calculate_working_tree_hash_with_python_files(self):
        """Test working tree hash calculation with Python files."""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some Python files
            (temp_path / "test1.py").write_text("print('hello')")
            (temp_path / "test2.py").write_text("x = 42")
            (temp_path / "subdir").mkdir()
            (temp_path / "subdir" / "test3.py").write_text("def foo(): pass")

            calculator = WorkingTreeStateCalculator(temp_dir)
            hash_result = calculator.calculate_working_tree_hash("abc123")

            assert isinstance(hash_result, str)
            assert len(hash_result) == 16

            # Same inputs should produce same hash
            hash_result_repeat = calculator.calculate_working_tree_hash("abc123")
            assert hash_result == hash_result_repeat

            # Sleep briefly to ensure modification time changes
            time.sleep(0.01)

            # Modify a file and hash should change
            (temp_path / "test1.py").write_text("print('modified')")
            hash_result2 = calculator.calculate_working_tree_hash("abc123")
            assert hash_result != hash_result2

    def test_calculate_working_tree_hash_different_commits(self):
        """Test that different commit SHAs produce different hashes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('test')")

            calculator = WorkingTreeStateCalculator(temp_dir)
            hash1 = calculator.calculate_working_tree_hash("commit1")
            hash2 = calculator.calculate_working_tree_hash("commit2")

            assert hash1 != hash2

    def test_get_python_files_excludes_common_directories(self):
        """Test that _get_python_files excludes common directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create Python files in various locations
            (temp_path / "main.py").write_text("# main")
            (temp_path / "src").mkdir()
            (temp_path / "src" / "app.py").write_text("# app")

            # Create files in directories that should be excluded
            (temp_path / "__pycache__").mkdir()
            (temp_path / "__pycache__" / "cached.py").write_text("# cached")
            (temp_path / ".venv").mkdir()
            (temp_path / ".venv" / "lib.py").write_text("# lib")
            (temp_path / "node_modules").mkdir()
            (temp_path / "node_modules" / "module.py").write_text("# module")

            calculator = WorkingTreeStateCalculator(temp_dir)
            python_files = calculator._get_python_files()

            # Should include main files but exclude cached/venv/node_modules
            file_names = {f.name for f in python_files}
            assert "main.py" in file_names
            assert "app.py" in file_names
            assert "cached.py" not in file_names
            assert "lib.py" not in file_names
            assert "module.py" not in file_names

    @patch("subprocess.run")
    def test_get_current_commit_sha_success(self, mock_run):
        """Test successful commit SHA retrieval."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "abcdef123456\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            commit_sha = calculator.get_current_commit_sha()

            assert commit_sha == "abcdef123456"
            mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_current_commit_sha_failure(self, mock_run):
        """Test commit SHA retrieval failure."""
        mock_run.return_value.returncode = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            commit_sha = calculator.get_current_commit_sha()

            assert commit_sha is None

    @patch("subprocess.run")
    def test_has_uncommitted_changes_clean_repo(self, mock_run):
        """Test uncommitted changes detection for clean repo."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            has_changes = calculator.has_uncommitted_changes()

            assert has_changes is False

    @patch("subprocess.run")
    def test_has_uncommitted_changes_dirty_repo(self, mock_run):
        """Test uncommitted changes detection for dirty repo."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = " M modified_file.py\n?? new_file.py\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            calculator = WorkingTreeStateCalculator(temp_dir)
            has_changes = calculator.has_uncommitted_changes()

            assert has_changes is True


class TestWorkingTreeCacheManager:
    """Test the enhanced CodeQualityCacheManager with working tree support."""

    def test_cache_clean_repo_metrics(self):
        """Test caching metrics for clean repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics(total_complexity=50)
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

            test_metrics = ExtendedComplexityMetrics(total_complexity=60)
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

            test_metrics = ExtendedComplexityMetrics(total_complexity=40)

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

            clean_metrics = ExtendedComplexityMetrics(total_complexity=30)
            dirty_metrics = ExtendedComplexityMetrics(total_complexity=35)

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

                assert clean_cached.total_complexity == 30
                assert dirty_cached.total_complexity == 35

    def test_is_cache_valid_for_different_states(self):
        """Test cache validity checking for different repository states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            test_metrics = ExtendedComplexityMetrics()

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
                expected_metrics = ExtendedComplexityMetrics(total_complexity=75)
                expected_delta = ComplexityDelta(total_complexity_change=25)
                mock_calc.return_value = (expected_metrics, expected_delta)

                # First call - should calculate and cache
                metrics1, delta1 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics1.total_complexity == 75
                assert delta1.total_complexity_change == 25
                mock_calc.assert_called_once()

                # Second call with same working tree state - should hit cache
                metrics2, delta2 = db._get_session_complexity_metrics("test_session", temp_dir, None)

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
                expected_metrics = ExtendedComplexityMetrics(total_complexity=50)
                expected_delta = ComplexityDelta(total_complexity_change=0)
                mock_calc.return_value = (expected_metrics, expected_delta)

                # First call - should calculate and cache
                metrics1, delta1 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics1.total_complexity == 50
                assert delta1.total_complexity_change == 0
                mock_calc.assert_called_once()

                # Second call - should hit cache
                metrics2, delta2 = db._get_session_complexity_metrics("test_session", temp_dir, None)

                assert metrics2.total_complexity == 50
                assert delta2.total_complexity_change == 0
                # Should still only be called once due to caching
                assert mock_calc.call_count == 1

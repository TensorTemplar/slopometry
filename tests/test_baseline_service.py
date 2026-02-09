"""Tests for baseline_service.py."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import make_test_metrics

from slopometry.core.models import ExtendedComplexityMetrics, HistoricalMetricStats, QPEScore, RepoBaseline
from slopometry.summoner.services.baseline_service import (
    BaselineService,
    CommitInfo,
    _compute_single_delta_task,
)


class TestComputeStats:
    """Tests for BaselineService._compute_stats."""

    def test_compute_stats__handles_empty_values(self):
        """Test that empty values return zeroed stats."""
        service = BaselineService(db=MagicMock())
        stats = service._compute_stats("test_metric", [])

        assert stats.metric_name == "test_metric"
        assert stats.mean == 0.0
        assert stats.std_dev == 0.0
        assert stats.median == 0.0
        assert stats.min_value == 0.0
        assert stats.max_value == 0.0
        assert stats.sample_count == 0
        assert stats.trend_coefficient == 0.0

    def test_compute_stats__single_value_stdev_zero(self):
        """Test that single value has stdev of zero."""
        service = BaselineService(db=MagicMock())
        stats = service._compute_stats("cc_delta", [5.0])

        assert stats.mean == 5.0
        assert stats.std_dev == 0.0
        assert stats.median == 5.0
        assert stats.min_value == 5.0
        assert stats.max_value == 5.0
        assert stats.sample_count == 1

    def test_compute_stats__calculates_mean_median_stdev_correctly(self):
        """Test that statistics are calculated correctly."""
        service = BaselineService(db=MagicMock())
        # Values: 1, 2, 3, 4, 5 => mean=3, median=3, stdev~1.58
        stats = service._compute_stats("effort_delta", [1.0, 2.0, 3.0, 4.0, 5.0])

        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
        assert stats.sample_count == 5
        # stdev of [1,2,3,4,5] is sqrt(2) ≈ 1.5811
        assert abs(stats.std_dev - 1.5811) < 0.001

    def test_compute_stats__metric_name_preserved_in_result(self):
        """Test that metric name is preserved in result."""
        service = BaselineService(db=MagicMock())
        stats = service._compute_stats("mi_delta", [10.0, 20.0])

        assert stats.metric_name == "mi_delta"


class TestComputeTrend:
    """Tests for BaselineService._compute_trend."""

    def test_compute_trend__handles_single_value(self):
        """Test that single value returns zero trend."""
        service = BaselineService(db=MagicMock())
        trend = service._compute_trend([5.0])

        assert trend == 0.0

    def test_compute_trend__handles_empty_list(self):
        """Test that empty list returns zero trend."""
        service = BaselineService(db=MagicMock())
        trend = service._compute_trend([])

        assert trend == 0.0

    def test_compute_trend__positive_when_increasing(self):
        """Test positive trend when values increase over time.

        Note: Values arrive in reverse chronological order (newest first).
        So [5, 4, 3, 2, 1] means oldest=1, newest=5, which is increasing.
        """
        service = BaselineService(db=MagicMock())
        # Reverse chronological: newest=5, ..., oldest=1 => increasing over time
        trend = service._compute_trend([5.0, 4.0, 3.0, 2.0, 1.0])

        assert trend > 0, f"Expected positive trend, got {trend}"

    def test_compute_trend__negative_when_decreasing(self):
        """Test negative trend when values decrease over time.

        Note: Values arrive in reverse chronological order (newest first).
        So [1, 2, 3, 4, 5] means oldest=5, newest=1, which is decreasing.
        """
        service = BaselineService(db=MagicMock())
        # Reverse chronological: newest=1, ..., oldest=5 => decreasing over time
        trend = service._compute_trend([1.0, 2.0, 3.0, 4.0, 5.0])

        assert trend < 0, f"Expected negative trend, got {trend}"

    def test_compute_trend__zero_when_flat(self):
        """Test zero trend when values are constant."""
        service = BaselineService(db=MagicMock())
        trend = service._compute_trend([3.0, 3.0, 3.0, 3.0, 3.0])

        assert trend == 0.0

    def test_compute_trend__two_values_computes_slope(self):
        """Test that two values correctly compute slope."""
        service = BaselineService(db=MagicMock())
        # [10, 5] in reverse chronological => oldest=5, newest=10
        # chronological: [5, 10] => slope = (10-5)/(1-0) = 5
        trend = service._compute_trend([10.0, 5.0])

        assert trend == 5.0


class TestGetOrComputeBaseline:
    """Tests for BaselineService.get_or_compute_baseline."""

    def test_get_or_compute_baseline__returns_cached_when_head_unchanged(self, tmp_path: Path):
        """Test that cached baseline is returned when HEAD hasn't changed."""
        mock_db = MagicMock()
        cached_baseline = RepoBaseline(
            repository_path=str(tmp_path),
            computed_at=datetime.now(),
            head_commit_sha="abc123",
            total_commits_analyzed=10,
            cc_delta_stats=HistoricalMetricStats(
                metric_name="cc_delta",
                mean=1.0,
                std_dev=0.5,
                median=1.0,
                min_value=0.0,
                max_value=2.0,
                sample_count=10,
                trend_coefficient=0.1,
            ),
            effort_delta_stats=HistoricalMetricStats(
                metric_name="effort_delta",
                mean=100.0,
                std_dev=50.0,
                median=100.0,
                min_value=0.0,
                max_value=200.0,
                sample_count=10,
                trend_coefficient=0.2,
            ),
            mi_delta_stats=HistoricalMetricStats(
                metric_name="mi_delta",
                mean=-0.5,
                std_dev=0.25,
                median=-0.5,
                min_value=-1.0,
                max_value=0.0,
                sample_count=10,
                trend_coefficient=-0.05,
            ),
            current_metrics=ExtendedComplexityMetrics(**make_test_metrics(total_complexity=100)),
            oldest_commit_date=datetime.now(),
            newest_commit_date=datetime.now(),
            qpe_stats=HistoricalMetricStats(
                metric_name="qpe_delta",
                mean=0.001,
                std_dev=0.005,
                median=0.001,
                min_value=-0.01,
                max_value=0.02,
                sample_count=10,
                trend_coefficient=0.0,
            ),
            current_qpe=QPEScore(
                qpe=0.45,
                mi_normalized=0.5,
                smell_penalty=0.1,
                adjusted_quality=0.45,
                smell_counts={},
            ),
        )
        mock_db.get_cached_baseline.return_value = cached_baseline

        service = BaselineService(db=mock_db)

        with patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker:
            mock_git = MockGitTracker.return_value
            mock_git._get_current_commit_sha.return_value = "abc123"

            result = service.get_or_compute_baseline(tmp_path)

        assert result == cached_baseline
        mock_db.get_cached_baseline.assert_called_once_with(str(tmp_path.resolve()), "abc123")

    def test_get_or_compute_baseline__recomputes_when_flag_set(self, tmp_path: Path):
        """Test that baseline is recomputed when recompute=True."""
        mock_db = MagicMock()
        mock_db.get_cached_baseline.return_value = None  # Shouldn't be called anyway

        service = BaselineService(db=mock_db)

        with (
            patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker,
            patch.object(service, "compute_full_baseline") as mock_compute,
        ):
            mock_git = MockGitTracker.return_value
            mock_git._get_current_commit_sha.return_value = "abc123"
            mock_compute.return_value = None

            service.get_or_compute_baseline(tmp_path, recompute=True)

        # get_cached_baseline should NOT be called when recompute=True
        mock_db.get_cached_baseline.assert_not_called()
        mock_compute.assert_called_once()

    def test_get_or_compute_baseline__saves_to_database(self, tmp_path: Path):
        """Test that computed baseline is saved to database."""
        mock_db = MagicMock()
        mock_db.get_cached_baseline.return_value = None

        new_baseline = RepoBaseline(
            repository_path=str(tmp_path),
            computed_at=datetime.now(),
            head_commit_sha="abc123",
            total_commits_analyzed=5,
            cc_delta_stats=HistoricalMetricStats(
                metric_name="cc_delta",
                mean=0.0,
                std_dev=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                sample_count=0,
                trend_coefficient=0.0,
            ),
            effort_delta_stats=HistoricalMetricStats(
                metric_name="effort_delta",
                mean=0.0,
                std_dev=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                sample_count=0,
                trend_coefficient=0.0,
            ),
            mi_delta_stats=HistoricalMetricStats(
                metric_name="mi_delta",
                mean=0.0,
                std_dev=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                sample_count=0,
                trend_coefficient=0.0,
            ),
            current_metrics=ExtendedComplexityMetrics(**make_test_metrics()),
            oldest_commit_date=datetime.now(),
            newest_commit_date=datetime.now(),
        )

        service = BaselineService(db=mock_db)

        with (
            patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker,
            patch.object(service, "compute_full_baseline") as mock_compute,
        ):
            mock_git = MockGitTracker.return_value
            mock_git._get_current_commit_sha.return_value = "abc123"
            mock_compute.return_value = new_baseline

            result = service.get_or_compute_baseline(tmp_path)

        assert result == new_baseline
        mock_db.save_baseline.assert_called_once_with(new_baseline)

    def test_get_or_compute_baseline__returns_none_if_no_head_commit(self, tmp_path: Path):
        """Test that None is returned when no HEAD commit exists."""
        mock_db = MagicMock()
        service = BaselineService(db=mock_db)

        with patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker:
            mock_git = MockGitTracker.return_value
            mock_git._get_current_commit_sha.return_value = None

            result = service.get_or_compute_baseline(tmp_path)

        assert result is None


class TestComputeFullBaseline:
    """Tests for BaselineService.compute_full_baseline."""

    def test_compute_full_baseline__returns_none_with_less_than_2_commits(self, tmp_path: Path):
        """Test that None is returned when fewer than 2 commits exist."""
        mock_db = MagicMock()
        service = BaselineService(db=mock_db)

        with patch.object(service, "_get_all_commits") as mock_get_commits:
            mock_get_commits.return_value = [CommitInfo(sha="abc123", timestamp=datetime.now())]

            result = service.compute_full_baseline(tmp_path)

        assert result is None

    def test_compute_full_baseline__returns_none_when_no_deltas(self, tmp_path: Path):
        """Test that None is returned when no deltas could be computed."""
        mock_db = MagicMock()
        service = BaselineService(db=mock_db)

        commits = [
            CommitInfo(sha="newest", timestamp=datetime.now()),
            CommitInfo(sha="oldest", timestamp=datetime.now()),
        ]

        with (
            patch.object(service, "_get_all_commits") as mock_get_commits,
            patch.object(service, "_compute_deltas_parallel") as mock_deltas,
            patch("slopometry.summoner.services.baseline_service.ComplexityAnalyzer") as MockAnalyzer,
        ):
            mock_get_commits.return_value = commits
            mock_deltas.return_value = []  # No deltas computed
            MockAnalyzer.return_value.analyze_extended_complexity.return_value = ExtendedComplexityMetrics(
                **make_test_metrics()
            )

            result = service.compute_full_baseline(tmp_path)

        assert result is None


class TestComputeSingleDeltaTask:
    """Tests for module-level _compute_single_delta_task function."""

    def test_compute_single_delta_task__returns_zero_delta_when_no_changed_files(self, tmp_path: Path):
        """Test that zero delta is returned when no Python files changed."""
        with patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker:
            mock_git = MockGitTracker.return_value
            mock_git.get_changed_python_files.return_value = []

            result = _compute_single_delta_task(tmp_path, "parent_sha", "child_sha")

        assert result is not None
        assert result.cc_delta == 0.0
        assert result.effort_delta == 0.0
        assert result.mi_delta == 0.0

    def test_compute_single_delta_task__returns_none_when_both_dirs_missing(self, tmp_path: Path):
        """Test that None is returned when neither parent nor child can be extracted."""
        with patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker:
            mock_git = MockGitTracker.return_value
            mock_git.get_changed_python_files.return_value = ["file.py"]
            mock_git.extract_specific_files_from_commit.return_value = None

            result = _compute_single_delta_task(tmp_path, "parent_sha", "child_sha")

        assert result is None

    def test_compute_single_delta_task__computes_delta_correctly(self, tmp_path: Path):
        """Test that delta is correctly computed as child - parent."""
        parent_metrics = ExtendedComplexityMetrics(
            **make_test_metrics(total_complexity=10, total_effort=100.0, total_mi=80.0)
        )
        child_metrics = ExtendedComplexityMetrics(
            **make_test_metrics(total_complexity=15, total_effort=150.0, total_mi=75.0)
        )

        with (
            patch("slopometry.summoner.services.baseline_service.GitTracker") as MockGitTracker,
            patch("slopometry.summoner.services.baseline_service.ComplexityAnalyzer") as MockAnalyzer,
            patch("slopometry.summoner.services.baseline_service.shutil.rmtree"),
        ):
            mock_git = MockGitTracker.return_value
            mock_git.get_changed_python_files.return_value = ["file.py"]
            mock_git.extract_specific_files_from_commit.side_effect = [
                tmp_path / "parent",
                tmp_path / "child",
            ]

            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze_extended_complexity.side_effect = [parent_metrics, child_metrics]

            result = _compute_single_delta_task(tmp_path, "parent_sha", "child_sha")

        assert result is not None
        assert result.cc_delta == 5  # 15 - 10
        assert result.effort_delta == 50.0  # 150 - 100
        assert result.mi_delta == -5.0  # 75 - 80

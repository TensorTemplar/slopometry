from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.complexity_analyzer import ComplexityAnalyzer


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path


def test_analyze_directory_aggregation(mock_path):
    """Test standard complexity analysis aggregation."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    with (
        patch("slopometry.core.git_tracker.GitTracker") as MockTracker,
        patch("radon.complexity.cc_visit") as mock_cc_visit,
    ):
        # Mock connection to GitTracker
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "a.py"
        file2 = mock_path / "b.py"
        file1.touch()
        file2.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1, file2]

        # Mock radon results
        # File 1: Two blocks, complexity 5 and 3 -> Total 8
        block1 = MagicMock(complexity=5)
        block2 = MagicMock(complexity=3)

        # File 2: One block, complexity 2 -> Total 2
        block3 = MagicMock(complexity=2)

        mock_cc_visit.side_effect = [[block1, block2], [block3]]

        metrics = analyzer._analyze_directory(mock_path)

        assert metrics.total_files_analyzed == 2
        assert metrics.total_complexity == 10  # 8 + 2
        assert metrics.average_complexity == 5.0  # 10 / 2
        assert metrics.max_complexity == 8
        assert metrics.min_complexity == 2


def test_analyze_directory_syntax_error(mock_path):
    """Test handling of syntax errors during analysis."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    with (
        patch("slopometry.core.git_tracker.GitTracker") as MockTracker,
        patch("radon.complexity.cc_visit") as mock_cc_visit,
    ):
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "good.py"
        file2 = mock_path / "bad.py"
        file1.touch()
        file2.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1, file2]

        # Good file returns complexity 5
        block1 = MagicMock(complexity=5)

        # Bad file raises SyntaxError
        def side_effect(content):
            if content == "GOOD":
                return [block1]
            raise SyntaxError("Invalid syntax")

        mock_cc_visit.side_effect = [[block1], SyntaxError("Fail")]

        metrics = analyzer._analyze_directory(mock_path)

        # Should only count the valid file
        # Note: current implementation skips errors in aggregation
        assert metrics.total_files_analyzed == 1
        assert metrics.total_complexity == 5


def test_analyze_extended_complexity(mock_path):
    """Test extended complexity analysis (CC + Halstead + MI)."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    with (
        patch("slopometry.core.git_tracker.GitTracker") as MockTracker,
        patch("radon.complexity.cc_visit") as mock_cc,
        patch("radon.metrics.h_visit") as mock_hal,
        patch("radon.metrics.mi_visit") as mock_mi,
        patch("slopometry.core.complexity_analyzer.PythonFeatureAnalyzer") as MockFeatureAnalyzer,
    ):
        # Mock Tracker
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "f.py"
        file1.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1]

        # Mock CC (Complexity 10)
        mock_cc.return_value = [MagicMock(complexity=10)]

        # Mock Halstead
        mock_h_data = MagicMock()
        mock_h_data.total.volume = 100.0
        mock_h_data.total.difficulty = 5.0
        mock_h_data.total.effort = 500.0
        mock_hal.return_value = mock_h_data

        # Mock MI
        mock_mi.return_value = 80.0

        # Mock Feature Analyzer
        mock_features = MockFeatureAnalyzer.return_value
        mock_feature_stats = MagicMock()
        mock_feature_stats.deprecations_count = 3
        mock_feature_stats.docstrings_count = 5
        mock_feature_stats.functions_count = 5
        mock_feature_stats.classes_count = 0

        # Mock coverage stats for division
        mock_feature_stats.args_count = 10
        mock_feature_stats.returns_count = 10
        mock_feature_stats.annotated_args_count = 5
        mock_feature_stats.annotated_returns_count = 5
        # (5+5)/(10+10) = 50%

        mock_features.analyze_directory.return_value = mock_feature_stats

        metrics = analyzer.analyze_extended_complexity(mock_path)

        # Checks
        assert metrics.total_complexity == 10
        assert metrics.average_volume == 100.0
        assert metrics.average_effort == 500.0
        assert metrics.average_mi == 80.0
        assert metrics.deprecation_count == 3
        assert metrics.type_hint_coverage == 50.0

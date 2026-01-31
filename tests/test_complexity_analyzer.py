from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import FileAnalysisResult


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path


def test_analyze_directory_aggregation(mock_path):
    """Test standard complexity analysis aggregation."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    mock_code_analyzer = MagicMock()
    analyzer._code_analyzer = mock_code_analyzer

    with patch("slopometry.core.git_tracker.GitTracker") as MockTracker:
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "a.py"
        file2 = mock_path / "b.py"
        file1.touch()
        file2.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1, file2]
        mock_tracker_instance.get_tracked_rust_files.return_value = []

        mock_code_analyzer.analyze_files.return_value = [
            FileAnalysisResult(
                path=str(file1),
                complexity=8,
                volume=100.0,
                difficulty=5.0,
                effort=500.0,
                mi=80.0,
                tokens=100,
            ),
            FileAnalysisResult(
                path=str(file2),
                complexity=2,
                volume=50.0,
                difficulty=2.0,
                effort=100.0,
                mi=90.0,
                tokens=50,
            ),
        ]

        metrics = analyzer._analyze_directory(mock_path)

        assert metrics.total_files_analyzed == 2
        assert metrics.total_complexity == 10  # 8 + 2
        assert metrics.average_complexity == 5.0  # 10 / 2
        assert metrics.max_complexity == 8
        assert metrics.min_complexity == 2


def test_analyze_directory_syntax_error(mock_path):
    """Test handling of syntax errors during analysis."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    mock_code_analyzer = MagicMock()
    analyzer._code_analyzer = mock_code_analyzer

    with patch("slopometry.core.git_tracker.GitTracker") as MockTracker:
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "good.py"
        file2 = mock_path / "bad.py"
        file1.touch()
        file2.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1, file2]
        mock_tracker_instance.get_tracked_rust_files.return_value = []

        mock_code_analyzer.analyze_files.return_value = [
            FileAnalysisResult(
                path=str(file1),
                complexity=5,
                volume=100.0,
                difficulty=5.0,
                effort=500.0,
                mi=80.0,
                tokens=100,
            ),
            FileAnalysisResult(
                path=str(file2),
                complexity=0,
                volume=0.0,
                difficulty=0.0,
                effort=0.0,
                mi=0.0,
                tokens=0,
                error="SyntaxError: Invalid syntax",
            ),
        ]

        metrics = analyzer._analyze_directory(mock_path)

        assert metrics.total_files_analyzed == 1
        assert metrics.total_complexity == 5


def test_analyze_extended_complexity(mock_path):
    """Test extended complexity analysis (CC + Halstead + MI)."""
    analyzer = ComplexityAnalyzer(working_directory=mock_path)

    mock_code_analyzer = MagicMock()
    analyzer._code_analyzer = mock_code_analyzer

    with (
        patch("slopometry.core.git_tracker.GitTracker") as MockTracker,
        patch("slopometry.core.complexity_analyzer.PythonFeatureAnalyzer") as MockFeatureAnalyzer,
    ):
        mock_tracker_instance = MockTracker.return_value
        file1 = mock_path / "f.py"
        file1.touch()
        mock_tracker_instance.get_tracked_python_files.return_value = [file1]
        mock_tracker_instance.get_tracked_rust_files.return_value = []

        mock_code_analyzer.analyze_files.return_value = [
            FileAnalysisResult(
                path=str(file1),
                complexity=10,
                volume=100.0,
                difficulty=5.0,
                effort=500.0,
                mi=80.0,
                tokens=100,
            ),
        ]

        mock_features = MockFeatureAnalyzer.return_value
        mock_feature_stats = MagicMock()
        mock_feature_stats.deprecations_count = 3
        mock_feature_stats.docstrings_count = 5
        mock_feature_stats.functions_count = 5
        mock_feature_stats.classes_count = 0
        mock_feature_stats.args_count = 10
        mock_feature_stats.returns_count = 10
        mock_feature_stats.annotated_args_count = 5
        mock_feature_stats.annotated_returns_count = 5
        mock_feature_stats.total_type_references = 20
        mock_feature_stats.any_type_count = 2
        mock_feature_stats.str_type_count = 5
        mock_feature_stats.orphan_comment_count = 0
        mock_feature_stats.orphan_comment_files = []
        mock_feature_stats.untracked_todo_count = 0
        mock_feature_stats.untracked_todo_files = []
        mock_feature_stats.inline_import_count = 0
        mock_feature_stats.inline_import_files = []
        mock_feature_stats.dict_get_with_default_count = 0
        mock_feature_stats.dict_get_with_default_files = []
        mock_feature_stats.hasattr_getattr_count = 0
        mock_feature_stats.hasattr_getattr_files = []
        mock_feature_stats.nonempty_init_count = 0
        mock_feature_stats.nonempty_init_files = []
        mock_feature_stats.test_skip_count = 0
        mock_feature_stats.test_skip_files = []
        mock_feature_stats.swallowed_exception_count = 0
        mock_feature_stats.swallowed_exception_files = []
        mock_feature_stats.type_ignore_count = 0
        mock_feature_stats.type_ignore_files = []
        mock_feature_stats.dynamic_execution_count = 0
        mock_feature_stats.dynamic_execution_files = []
        mock_feature_stats.single_method_class_count = 0
        mock_feature_stats.single_method_class_files = []
        mock_feature_stats.deep_inheritance_count = 0
        mock_feature_stats.deep_inheritance_files = []
        mock_feature_stats.passthrough_wrapper_count = 0
        mock_feature_stats.passthrough_wrapper_files = []
        mock_feature_stats.total_loc = 100
        mock_feature_stats.code_loc = 80
        mock_features.analyze_directory.return_value = mock_feature_stats

        metrics = analyzer.analyze_extended_complexity(mock_path)

        assert metrics.total_complexity == 10
        assert metrics.average_volume == 100.0
        assert metrics.average_effort == 500.0
        assert metrics.average_mi == 80.0
        assert metrics.deprecation_count == 3
        assert metrics.type_hint_coverage == 50.0
        assert metrics.any_type_percentage == 10.0
        assert metrics.str_type_percentage == 25.0
        assert "f.py" in metrics.files_by_effort
        assert metrics.files_by_effort["f.py"] == 500.0
        assert "f.py" in metrics.files_by_mi
        assert metrics.files_by_mi["f.py"] == 80.0

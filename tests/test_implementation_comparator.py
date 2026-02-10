"""Tests for implementation_comparator.py."""

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_test_metrics

from slopometry.core.models import ExtendedComplexityMetrics, QPEScore
from slopometry.summoner.services.implementation_comparator import (
    SubtreeExtractionError,
    _extract_subtree,
    compare_subtrees,
)


def _make_tar_with_python(name: str = "vendor/lib-a/main.py", content: bytes = b"print('hello')") -> bytes:
    """Create a tar archive containing a single Python file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_tar_without_python() -> bytes:
    """Create a tar archive with no Python files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="vendor/lib-a/README.md")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))
    return buf.getvalue()


def test_extract_subtree__returns_false_when_no_python_files(tmp_path: Path) -> None:
    """Test that False is returned when no .py files exist in the archive."""
    with patch("slopometry.summoner.services.implementation_comparator.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _make_tar_without_python()
        mock_run.return_value = mock_result

        result = _extract_subtree(tmp_path, "HEAD", "vendor/lib-a", tmp_path / "dest")

    assert result is False


def test_extract_subtree__returns_true_when_python_files_extracted(tmp_path: Path) -> None:
    """Test that True is returned when .py files are extracted."""
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    with patch("slopometry.summoner.services.implementation_comparator.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _make_tar_with_python()
        mock_run.return_value = mock_result

        result = _extract_subtree(tmp_path, "HEAD", "vendor/lib-a", dest_dir)

    assert result is True


def test_extract_subtree__raises_on_git_failure(tmp_path: Path) -> None:
    """Test that SubtreeExtractionError is raised on git archive failure."""
    with patch("slopometry.summoner.services.implementation_comparator.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = b"fatal: not a valid object name"
        mock_run.return_value = mock_result

        with pytest.raises(SubtreeExtractionError, match="git archive failed"):
            _extract_subtree(tmp_path, "HEAD", "vendor/nonexistent", tmp_path / "dest")


def test_compare_subtrees__returns_none_when_prefix_a_has_no_python(tmp_path: Path) -> None:
    """Test returns None when first prefix has no Python files."""
    with patch(
        "slopometry.summoner.services.implementation_comparator._extract_subtree",
        side_effect=[False, True],
    ):
        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b")

    assert result is None


def test_compare_subtrees__returns_none_when_prefix_b_has_no_python(tmp_path: Path) -> None:
    """Test returns None when second prefix has no Python files."""
    with patch(
        "slopometry.summoner.services.implementation_comparator._extract_subtree",
        side_effect=[True, False],
    ):
        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b")

    assert result is None


def test_compare_subtrees__returns_comparison_with_winner(tmp_path: Path) -> None:
    """Test returns valid ImplementationComparison with winner determination."""
    metrics_a = ExtendedComplexityMetrics(**make_test_metrics(average_mi=60.0, total_files_analyzed=5))
    metrics_b = ExtendedComplexityMetrics(**make_test_metrics(average_mi=80.0, total_files_analyzed=5))

    qpe_a = QPEScore(qpe=0.5, mi_normalized=0.6, smell_penalty=0.1, adjusted_quality=0.5)
    qpe_b = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.05, adjusted_quality=0.7)

    with (
        patch(
            "slopometry.summoner.services.implementation_comparator._extract_subtree",
            return_value=True,
        ),
        patch("slopometry.summoner.services.implementation_comparator.ComplexityAnalyzer") as MockAnalyzer,
        patch("slopometry.summoner.services.implementation_comparator.calculate_qpe") as mock_calc_qpe,
    ):
        MockAnalyzer.return_value.analyze_extended_complexity.side_effect = [metrics_a, metrics_b]
        mock_calc_qpe.side_effect = [qpe_a, qpe_b]

        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b", ref="main")

    assert result is not None
    assert result.prefix_a == "vendor/lib-a"
    assert result.prefix_b == "vendor/lib-b"
    assert result.ref == "main"
    assert result.aggregate_advantage > 0  # B is better
    assert result.winner == "vendor/lib-b"


def test_compare_subtrees__returns_tie_within_deadband(tmp_path: Path) -> None:
    """Test returns tie when advantage is within deadband."""
    metrics = ExtendedComplexityMetrics(**make_test_metrics(average_mi=70.0, total_files_analyzed=5))
    qpe = QPEScore(qpe=0.65, mi_normalized=0.7, smell_penalty=0.05, adjusted_quality=0.65)

    with (
        patch(
            "slopometry.summoner.services.implementation_comparator._extract_subtree",
            return_value=True,
        ),
        patch("slopometry.summoner.services.implementation_comparator.ComplexityAnalyzer") as MockAnalyzer,
        patch("slopometry.summoner.services.implementation_comparator.calculate_qpe") as mock_calc_qpe,
    ):
        MockAnalyzer.return_value.analyze_extended_complexity.return_value = metrics
        mock_calc_qpe.return_value = qpe

        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b")

    assert result is not None
    assert result.winner == "tie"
    assert abs(result.aggregate_advantage) < 0.01


def test_compare_subtrees__includes_smell_advantages(tmp_path: Path) -> None:
    """Test that smell_advantages is populated when smells differ."""
    metrics_a = ExtendedComplexityMetrics(**make_test_metrics(average_mi=60.0, total_files_analyzed=5))
    metrics_b = ExtendedComplexityMetrics(**make_test_metrics(average_mi=80.0, total_files_analyzed=5))

    qpe_a = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"swallowed_exception": 5},
    )
    qpe_b = QPEScore(
        qpe=0.7,
        mi_normalized=0.8,
        smell_penalty=0.05,
        adjusted_quality=0.7,
        smell_counts={"swallowed_exception": 1},
    )

    with (
        patch(
            "slopometry.summoner.services.implementation_comparator._extract_subtree",
            return_value=True,
        ),
        patch("slopometry.summoner.services.implementation_comparator.ComplexityAnalyzer") as MockAnalyzer,
        patch("slopometry.summoner.services.implementation_comparator.calculate_qpe") as mock_calc_qpe,
    ):
        MockAnalyzer.return_value.analyze_extended_complexity.side_effect = [metrics_a, metrics_b]
        mock_calc_qpe.side_effect = [qpe_a, qpe_b]

        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b")

    assert result is not None
    assert len(result.smell_advantages) > 0
    swallowed = next(sa for sa in result.smell_advantages if sa.smell_name == "swallowed_exception")
    assert swallowed.baseline_count == 5
    assert swallowed.candidate_count == 1
    assert swallowed.weighted_delta < 0  # B improved on this smell


def test_compare_subtrees__json_serialization(tmp_path: Path) -> None:
    """Test that result serializes to valid JSON for GRPO pipeline."""
    metrics = ExtendedComplexityMetrics(**make_test_metrics(average_mi=70.0, total_files_analyzed=5))

    qpe_a = QPEScore(qpe=0.5, mi_normalized=0.6, smell_penalty=0.1, adjusted_quality=0.5)
    qpe_b = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.05, adjusted_quality=0.7)

    with (
        patch(
            "slopometry.summoner.services.implementation_comparator._extract_subtree",
            return_value=True,
        ),
        patch("slopometry.summoner.services.implementation_comparator.ComplexityAnalyzer") as MockAnalyzer,
        patch("slopometry.summoner.services.implementation_comparator.calculate_qpe") as mock_calc_qpe,
    ):
        MockAnalyzer.return_value.analyze_extended_complexity.return_value = metrics
        mock_calc_qpe.side_effect = [qpe_a, qpe_b]

        result = compare_subtrees(tmp_path, "vendor/lib-a", "vendor/lib-b")

    assert result is not None
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    assert "aggregate_advantage" in parsed
    assert "winner" in parsed
    assert "smell_advantages" in parsed

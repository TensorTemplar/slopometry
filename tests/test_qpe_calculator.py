"""Tests for QPE (Quality-Per-Effort) Calculator functionality."""

import math
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from conftest import make_test_metrics

from slopometry.core.models import ExtendedComplexityMetrics, QPEScore
from slopometry.summoner.services.qpe_calculator import (
    CrossProjectComparator,
    QPECalculator,
    grpo_advantage,
)

# Known checkpoint commit for integration tests (Merge PR #29)
KNOWN_CHECKPOINT_COMMIT = "0a74cc3"


class TestQPECalculator:
    """Test the QPE (Quality-Per-Effort) calculator."""

    def test_calculate_qpe__returns_positive_score_for_quality_codebase(self):
        """Test that QPE calculation returns positive score for good quality code."""
        calculator = QPECalculator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_mi=75.0,  # Good MI
                total_files_analyzed=10,
                # No code smells
                hasattr_getattr_count=0,
                swallowed_exception_count=0,
                type_ignore_count=0,
                dynamic_execution_count=0,
                test_skip_count=0,
                dict_get_with_default_count=0,
                inline_import_count=0,
                orphan_comment_count=0,
                untracked_todo_count=0,
                nonempty_init_count=0,
            )
        )

        qpe_score = calculator.calculate_qpe(metrics)

        assert qpe_score.qpe > 0
        assert qpe_score.mi_normalized == 0.75
        assert qpe_score.smell_penalty == 0.0
        assert qpe_score.adjusted_quality == 0.75

    def test_calculate_qpe__smell_penalty_reduces_adjusted_quality(self):
        """Test that code smells reduce adjusted quality via smell penalty."""
        calculator = QPECalculator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_mi=75.0,
                total_files_analyzed=10,
                # Add some code smells
                hasattr_getattr_count=5,  # 0.10 weight each
                swallowed_exception_count=3,  # 0.15 weight each
            )
        )

        qpe_score = calculator.calculate_qpe(metrics)

        # Smell penalty should be > 0
        assert qpe_score.smell_penalty > 0
        # Adjusted quality should be less than MI normalized
        assert qpe_score.adjusted_quality < qpe_score.mi_normalized
        # Formula: adjusted = mi_normalized * (1 - smell_penalty)
        expected_adjusted = qpe_score.mi_normalized * (1 - qpe_score.smell_penalty)
        assert abs(qpe_score.adjusted_quality - expected_adjusted) < 0.001

    def test_calculate_qpe__smell_penalty_capped_at_0_5(self):
        """Test that smell penalty is capped at 0.5 even with many smells."""
        calculator = QPECalculator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_mi=75.0,
                total_files_analyzed=2,  # Few files
                # Many smells per file
                hasattr_getattr_count=100,
                swallowed_exception_count=100,
                type_ignore_count=100,
                dynamic_execution_count=100,
            )
        )

        qpe_score = calculator.calculate_qpe(metrics)

        assert qpe_score.smell_penalty <= 0.5

    def test_calculate_qpe__effort_factor_uses_log_scale(self):
        """Test that effort factor uses log scale for diminishing returns."""
        calculator = QPECalculator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_mi=75.0,
                total_files_analyzed=10,
            )
        )

        qpe_score = calculator.calculate_qpe(metrics)

        expected_effort_factor = math.log(50000.0 + 1)
        assert abs(qpe_score.effort_factor - expected_effort_factor) < 0.001

    def test_calculate_qpe__smell_counts_populated(self):
        """Test that smell counts are populated for debugging."""
        calculator = QPECalculator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_effort=50000.0,
                average_mi=75.0,
                total_files_analyzed=10,
                hasattr_getattr_count=5,
                type_ignore_count=3,
            )
        )

        qpe_score = calculator.calculate_qpe(metrics)

        assert "hasattr_getattr" in qpe_score.smell_counts
        assert qpe_score.smell_counts["hasattr_getattr"] == 5
        assert qpe_score.smell_counts["type_ignore"] == 3


class TestGRPOAdvantage:
    """Test the GRPO advantage calculation function."""

    def test_grpo_advantage__returns_positive_when_candidate_is_better(self):
        """Test that advantage is positive when candidate has higher QPE."""
        baseline = QPEScore(
            qpe=0.05,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
        )

        candidate = QPEScore(
            qpe=0.07,  # Higher QPE
            mi_normalized=0.8,
            smell_penalty=0.05,
            adjusted_quality=0.76,
            effort_factor=10.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        assert advantage > 0

    def test_grpo_advantage__returns_negative_when_candidate_is_worse(self):
        """Test that advantage is negative when candidate has lower QPE."""
        baseline = QPEScore(
            qpe=0.07,
            mi_normalized=0.8,
            smell_penalty=0.05,
            adjusted_quality=0.76,
            effort_factor=10.0,
        )

        candidate = QPEScore(
            qpe=0.05,  # Lower QPE
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        assert advantage < 0

    def test_grpo_advantage__returns_zero_when_qpe_matches(self):
        """Test that advantage is zero when QPE scores are equal."""
        baseline = QPEScore(
            qpe=0.05,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
        )

        candidate = QPEScore(
            qpe=0.05,  # Same QPE
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        assert advantage == 0.0

    def test_grpo_advantage__bounded_between_minus_1_and_1(self):
        """Test that advantage is bounded in [-1, 1] via tanh."""
        # Extreme improvement case
        baseline = QPEScore(
            qpe=0.01,
            mi_normalized=0.5,
            smell_penalty=0.3,
            adjusted_quality=0.35,
            effort_factor=10.0,
        )

        candidate = QPEScore(
            qpe=1.0,  # 100x improvement
            mi_normalized=1.0,
            smell_penalty=0.0,
            adjusted_quality=1.0,
            effort_factor=1.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        # tanh approaches ±1 asymptotically, so we allow the boundary
        assert -1 <= advantage <= 1

        # Extreme degradation case
        worse_candidate = QPEScore(
            qpe=0.0001,  # Much worse
            mi_normalized=0.1,
            smell_penalty=0.5,
            adjusted_quality=0.05,
            effort_factor=20.0,
        )

        degradation = grpo_advantage(baseline, worse_candidate)

        assert -1 <= degradation <= 1

    def test_grpo_advantage__handles_zero_baseline(self):
        """Test that advantage handles zero baseline QPE gracefully."""
        baseline = QPEScore(
            qpe=0.0,  # Zero baseline
            mi_normalized=0.0,
            smell_penalty=0.5,
            adjusted_quality=0.0,
            effort_factor=10.0,
        )

        candidate = QPEScore(
            qpe=0.05,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        # Should still work and be positive
        assert advantage > 0


class TestCrossProjectComparator:
    """Test the cross-project comparison functionality."""

    def test_compare_metrics__returns_flat_rankings(self):
        """Test that projects are returned in a flat ranking by QPE."""
        comparator = CrossProjectComparator()

        metrics_a = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=5000.0, average_mi=75.0, total_files_analyzed=5)
        )
        metrics_b = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_mi=70.0, total_files_analyzed=10)
        )

        result = comparator.compare_metrics(
            [
                ("project-a", metrics_a),
                ("project-b", metrics_b),
            ]
        )

        assert result.total_projects == 2
        assert len(result.rankings) == 2

    def test_compare_metrics__ranks_by_qpe_highest_first(self):
        """Test that projects are ranked by QPE from highest to lowest."""
        comparator = CrossProjectComparator()

        # Create two projects with different quality
        high_quality = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_mi=90.0, total_files_analyzed=10)
        )
        low_quality = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=55000.0, average_mi=60.0, total_files_analyzed=10)
        )

        result = comparator.compare_metrics(
            [
                ("low-quality", low_quality),
                ("high-quality", high_quality),
            ]
        )

        # High quality should be ranked first (higher QPE)
        assert result.rankings[0].project_name == "high-quality"
        assert result.rankings[1].project_name == "low-quality"
        assert result.rankings[0].qpe_score.qpe > result.rankings[1].qpe_score.qpe

    def test_compare_metrics__includes_qpe_details(self):
        """Test that ranking results include QPE score details."""
        comparator = CrossProjectComparator()

        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_mi=75.0, total_files_analyzed=10)
        )

        result = comparator.compare_metrics([("test-project", metrics)])

        assert result.rankings[0].project_name == "test-project"
        assert result.rankings[0].qpe_score.qpe > 0
        assert result.rankings[0].qpe_score.mi_normalized > 0
        assert result.rankings[0].metrics is not None


class TestQPEIntegration:
    """Integration tests for QPE using the actual slopometry repository.

    These tests verify the full QPE pipeline works against real code,
    using a known checkpoint commit as a stable baseline for assertions.
    """

    @pytest.fixture
    def repo_path(self) -> Path:
        """Return the path to the slopometry repository root."""
        return Path(__file__).parent.parent

    def test_qpe_cli_command__runs_without_error(self, repo_path: Path) -> None:
        """Test that the qpe CLI command executes without errors."""
        result = subprocess.run(
            ["uv", "run", "slopometry", "summoner", "qpe", "--repo-path", str(repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"qpe command failed with: {result.stderr}"
        assert "Quality-Per-Effort Score" in result.stdout
        assert "QPE:" in result.stdout

    def test_qpe_cli_command__json_output_is_valid(self, repo_path: Path) -> None:
        """Test that --json flag produces valid JSON output."""
        import json

        result = subprocess.run(
            ["uv", "run", "slopometry", "summoner", "qpe", "--repo-path", str(repo_path), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"qpe --json failed with: {result.stderr}"

        qpe_data = json.loads(result.stdout)

        assert "qpe" in qpe_data
        assert "mi_normalized" in qpe_data
        assert "smell_penalty" in qpe_data
        assert "adjusted_quality" in qpe_data
        assert "effort_factor" in qpe_data
        assert "smell_counts" in qpe_data

        assert isinstance(qpe_data["qpe"], float)
        assert qpe_data["qpe"] > 0

    def test_qpe_calculator__real_codebase_produces_consistent_results(self, repo_path: Path) -> None:
        """Test QPE calculation on real codebase produces stable, sensible values."""
        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics = analyzer.analyze_extended_complexity()

        calculator = QPECalculator()
        qpe_score = calculator.calculate_qpe(metrics)

        # QPE should be positive for a working codebase
        assert qpe_score.qpe > 0

        # MI normalized should be in valid range (0-1)
        assert 0 <= qpe_score.mi_normalized <= 1

        # Smell penalty should be capped at 0.5
        assert 0 <= qpe_score.smell_penalty <= 0.5

        # Adjusted quality should be MI * (1 - smell_penalty)
        expected_adjusted = qpe_score.mi_normalized * (1 - qpe_score.smell_penalty)
        assert abs(qpe_score.adjusted_quality - expected_adjusted) < 0.001

        # Effort factor should be log(effort + 1)
        expected_effort_factor = math.log(metrics.total_effort + 1)
        assert abs(qpe_score.effort_factor - expected_effort_factor) < 0.001

        # QPE formula verification: adjusted_quality / effort_factor
        expected_qpe = qpe_score.adjusted_quality / qpe_score.effort_factor
        assert abs(qpe_score.qpe - expected_qpe) < 0.0001

    def test_display_qpe_score__renders_without_error(self, repo_path: Path) -> None:
        """Test that display_qpe_score renders without AttributeError (regression test for effort_tier bug)."""
        from rich.console import Console

        from slopometry.core.complexity_analyzer import ComplexityAnalyzer
        from slopometry.display.formatters import display_qpe_score

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics = analyzer.analyze_extended_complexity()

        calculator = QPECalculator()
        qpe_score = calculator.calculate_qpe(metrics)

        # Capture output to verify no errors
        console_output = StringIO()
        console = Console(file=console_output, force_terminal=True, width=120)

        # This should not raise AttributeError: 'QPEScore' object has no attribute 'effort_tier'
        display_qpe_score(qpe_score, metrics)

    def test_qpe_score_model__serializes_to_json_without_effort_tier(self) -> None:
        """Test that QPEScore model serializes correctly without effort_tier field."""
        qpe_score = QPEScore(
            qpe=0.05,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
            smell_counts={"hasattr_getattr": 5, "type_ignore": 3},
        )

        json_output = qpe_score.model_dump_json()

        assert "qpe" in json_output
        assert "effort_tier" not in json_output

        # Verify round-trip
        restored = QPEScore.model_validate_json(json_output)
        assert restored.qpe == 0.05
        assert restored.smell_counts["hasattr_getattr"] == 5

    def test_qpe_calculator__handles_empty_codebase_gracefully(self, tmp_path: Path) -> None:
        """Test that QPE calculator handles empty directory without crashing."""
        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=tmp_path)
        metrics = analyzer.analyze_extended_complexity()

        calculator = QPECalculator()
        qpe_score = calculator.calculate_qpe(metrics)

        # Should handle gracefully (might return 0 but shouldn't crash)
        assert qpe_score.qpe >= 0

    def test_qpe_at_known_checkpoint__has_expected_characteristics(self, repo_path: Path) -> None:
        """Test QPE at known checkpoint has expected quality characteristics.

        This test documents expected quality metrics at a known commit,
        allowing detection of unexpected regressions in the codebase quality.
        """
        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics = analyzer.analyze_extended_complexity()

        calculator = QPECalculator()
        qpe_score = calculator.calculate_qpe(metrics)

        # Documented expectations for slopometry codebase quality
        # These are loose bounds that should remain stable across minor changes

        # MI should be in reasonable range for a Python codebase (40-70 typical)
        assert 30 <= metrics.average_mi <= 80, f"MI {metrics.average_mi} outside expected range"

        # Should analyze multiple files
        assert metrics.total_files_analyzed > 10, "Expected to analyze more than 10 Python files"

        # QPE should be positive and in typical range for a Python project
        assert 0.01 <= qpe_score.qpe <= 0.15, f"QPE {qpe_score.qpe} outside expected range"

        # Smell counts should be populated
        total_smells = sum(qpe_score.smell_counts.values())
        assert total_smells > 0, "Expected some code smells in a real codebase"

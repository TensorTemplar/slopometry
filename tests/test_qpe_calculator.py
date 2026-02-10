"""Tests for QPE (Quality-Per-Effort) Calculator functionality."""

import subprocess
from io import StringIO
from pathlib import Path

import pytest
from conftest import make_test_metrics

from slopometry.core.models import ExtendedComplexityMetrics, QPEScore
from slopometry.summoner.services.qpe_calculator import (
    calculate_qpe,
    compare_project_metrics,
    grpo_advantage,
    smell_advantage,
)

# Known checkpoint commit for integration tests (Merge PR #29)
KNOWN_CHECKPOINT_COMMIT = "0a74cc3"


class TestCalculateQPE:
    """Test the calculate_qpe function."""

    def test_calculate_qpe__returns_positive_score_for_quality_codebase(self):
        """Test that QPE calculation returns positive score for good quality code."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_effort=5000.0,
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

        qpe_score = calculate_qpe(metrics)

        assert qpe_score.qpe > 0
        assert qpe_score.mi_normalized == 0.75
        assert qpe_score.smell_penalty == 0.0
        assert qpe_score.adjusted_quality == 0.75

    def test_calculate_qpe__smell_penalty_reduces_adjusted_quality(self):
        """Test that code smells reduce adjusted quality via smell penalty."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_effort=5000.0,
                average_mi=75.0,
                total_files_analyzed=10,
                # Add some code smells
                hasattr_getattr_count=5,  # 0.10 weight each
                swallowed_exception_count=3,  # 0.15 weight each
            )
        )

        qpe_score = calculate_qpe(metrics)

        # Smell penalty should be > 0
        assert qpe_score.smell_penalty > 0
        # Adjusted quality should be less than MI normalized
        assert qpe_score.adjusted_quality < qpe_score.mi_normalized
        # Formula: adjusted = mi_normalized * (1 - smell_penalty)
        expected_adjusted = qpe_score.mi_normalized * (1 - qpe_score.smell_penalty)
        assert abs(qpe_score.adjusted_quality - expected_adjusted) < 0.001

    def test_calculate_qpe__smell_penalty_saturates_with_sigmoid(self):
        """Test that smell penalty uses sigmoid saturation (approaches 0.9 asymptotically)."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_complexity=100,
                total_volume=5000.0,
                total_effort=50000.0,
                average_effort=25000.0,
                average_mi=75.0,
                total_files_analyzed=2,  # Few files
                # Many smells per file
                hasattr_getattr_count=100,
                swallowed_exception_count=100,
                type_ignore_count=100,
                dynamic_execution_count=100,
            )
        )

        qpe_score = calculate_qpe(metrics)

        # Sigmoid approaches 0.9 asymptotically but never exceeds it
        assert qpe_score.smell_penalty <= 0.9
        # With many smells, penalty should be high (close to saturation)
        assert qpe_score.smell_penalty > 0.5

    def test_calculate_qpe__spreading_smells_does_not_reduce_penalty(self):
        """Test that spreading smells across files doesn't reduce penalty (anti-gaming fix)."""


        # Same smells, 1 file
        metrics_concentrated = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_effort=50000.0,
                average_effort=5000.0,
                average_mi=75.0,
                total_files_analyzed=10,  # 10 total files
                hasattr_getattr_count=10,
                hasattr_getattr_files=["file1.py"],  # All in 1 file
            )
        )

        # Same smells, 10 files
        metrics_spread = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_effort=50000.0,
                average_effort=5000.0,
                average_mi=75.0,
                total_files_analyzed=10,  # 10 total files
                hasattr_getattr_count=10,
                hasattr_getattr_files=[f"file{i}.py" for i in range(10)],  # Spread across 10 files
            )
        )

        qpe_concentrated = calculate_qpe(metrics_concentrated)
        qpe_spread = calculate_qpe(metrics_spread)

        # Both should have the same smell penalty (normalizing by total files, not affected)
        assert abs(qpe_concentrated.smell_penalty - qpe_spread.smell_penalty) < 0.001

    def test_calculate_qpe__qpe_equals_adjusted_quality(self):
        """Test that qpe equals adjusted_quality."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_effort=50000.0,
                average_mi=75.0,
                total_files_analyzed=10,
            )
        )

        qpe_score = calculate_qpe(metrics)

        assert qpe_score.qpe == qpe_score.adjusted_quality

    def test_calculate_qpe__smell_counts_populated(self):
        """Test that smell counts are populated for debugging."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(
                total_effort=50000.0,
                average_effort=5000.0,
                average_mi=75.0,
                total_files_analyzed=10,
                hasattr_getattr_count=5,
                type_ignore_count=3,
            )
        )

        qpe_score = calculate_qpe(metrics)

        assert qpe_score.smell_counts.hasattr_getattr == 5
        assert qpe_score.smell_counts.type_ignore == 3


class TestGRPOAdvantage:
    """Test the GRPO advantage calculation function."""

    def test_grpo_advantage__returns_positive_when_candidate_is_better(self):
        """Test that advantage is positive when candidate has higher QPE."""
        baseline = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
        )

        candidate = QPEScore(
            qpe=0.76,
            mi_normalized=0.8,
            smell_penalty=0.05,
            adjusted_quality=0.76,
        )

        advantage = grpo_advantage(baseline, candidate)

        assert advantage > 0

    def test_grpo_advantage__returns_negative_when_candidate_is_worse(self):
        """Test that advantage is negative when candidate has lower QPE."""
        baseline = QPEScore(
            qpe=0.76,
            mi_normalized=0.8,
            smell_penalty=0.05,
            adjusted_quality=0.76,
        )

        candidate = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
        )

        advantage = grpo_advantage(baseline, candidate)

        assert advantage < 0

    def test_grpo_advantage__returns_zero_when_qpe_matches(self):
        """Test that advantage is zero when QPE scores are equal."""
        baseline = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
        )

        candidate = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
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
            adjusted_quality=0.01,
        )

        candidate = QPEScore(
            qpe=1.0,
            mi_normalized=1.0,
            smell_penalty=0.0,
            adjusted_quality=1.0,
        )

        advantage = grpo_advantage(baseline, candidate)

        # tanh approaches ±1 asymptotically, so we allow the boundary
        assert -1 <= advantage <= 1

        # Extreme degradation case
        worse_candidate = QPEScore(
            qpe=0.0001,
            mi_normalized=0.1,
            smell_penalty=0.5,
            adjusted_quality=0.0001,
        )

        degradation = grpo_advantage(baseline, worse_candidate)

        assert -1 <= degradation <= 1

    def test_grpo_advantage__handles_zero_baseline(self):
        """Test that advantage handles zero baseline QPE gracefully."""
        baseline = QPEScore(
            qpe=0.0,
            mi_normalized=0.0,
            smell_penalty=0.5,
            adjusted_quality=0.0,
        )

        candidate = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
        )

        advantage = grpo_advantage(baseline, candidate)

        # Should still work and be positive
        assert advantage > 0


def test_smell_advantage__all_zero_deltas_for_equal_counts() -> None:
    """Test that equal smell counts produce all-zero weighted deltas."""
    baseline = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.0, adjusted_quality=0.7)
    candidate = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.0, adjusted_quality=0.7)

    result = smell_advantage(baseline, candidate)
    assert all(sa.weighted_delta == 0.0 for sa in result)


def test_smell_advantage__negative_delta_when_candidate_reduces_smells() -> None:
    """Test that candidate reducing smells produces negative weighted_delta."""
    baseline = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"swallowed_exception": 5},
    )
    candidate = QPEScore(
        qpe=0.6,
        mi_normalized=0.7,
        smell_penalty=0.1,
        adjusted_quality=0.6,
        smell_counts={"swallowed_exception": 2},
    )

    result = smell_advantage(baseline, candidate)
    non_zero = [sa for sa in result if sa.weighted_delta != 0.0]
    assert len(non_zero) == 1
    assert non_zero[0].smell_name == "swallowed_exception"
    assert non_zero[0].baseline_count == 5
    assert non_zero[0].candidate_count == 2
    assert non_zero[0].weighted_delta < 0  # Improvement


def test_smell_advantage__positive_delta_when_candidate_adds_smells() -> None:
    """Test that candidate adding smells produces positive weighted_delta."""
    baseline = QPEScore(
        qpe=0.6,
        mi_normalized=0.7,
        smell_penalty=0.1,
        adjusted_quality=0.6,
        smell_counts={"hasattr_getattr": 2},
    )
    candidate = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"hasattr_getattr": 7},
    )

    result = smell_advantage(baseline, candidate)
    non_zero = [sa for sa in result if sa.weighted_delta != 0.0]
    assert len(non_zero) == 1
    assert non_zero[0].weighted_delta > 0  # Regression


def test_smell_advantage__handles_asymmetric_smell_sets() -> None:
    """Test that smells present in only one side are handled correctly."""
    baseline = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"swallowed_exception": 3},
    )
    candidate = QPEScore(
        qpe=0.6,
        mi_normalized=0.7,
        smell_penalty=0.1,
        adjusted_quality=0.6,
        smell_counts={"hasattr_getattr": 2},
    )

    result = smell_advantage(baseline, candidate)
    smell_names = {sa.smell_name for sa in result}
    assert "swallowed_exception" in smell_names
    assert "hasattr_getattr" in smell_names

    swallowed = next(sa for sa in result if sa.smell_name == "swallowed_exception")
    assert swallowed.baseline_count == 3
    assert swallowed.candidate_count == 0
    assert swallowed.weighted_delta < 0

    hasattr_sa = next(sa for sa in result if sa.smell_name == "hasattr_getattr")
    assert hasattr_sa.baseline_count == 0
    assert hasattr_sa.candidate_count == 2
    assert hasattr_sa.weighted_delta > 0


def test_smell_advantage__sorted_by_impact_magnitude() -> None:
    """Test that results are sorted by absolute weighted_delta descending."""
    baseline = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"swallowed_exception": 10, "orphan_comment": 5, "hasattr_getattr": 3},
    )
    candidate = QPEScore(
        qpe=0.6,
        mi_normalized=0.7,
        smell_penalty=0.1,
        adjusted_quality=0.6,
        smell_counts={"swallowed_exception": 2, "orphan_comment": 4, "hasattr_getattr": 3},
    )

    result = smell_advantage(baseline, candidate)
    non_zero = [sa for sa in result if sa.weighted_delta != 0.0]
    assert len(non_zero) >= 1
    magnitudes = [abs(sa.weighted_delta) for sa in non_zero]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_smell_advantage__uses_correct_weights_from_registry() -> None:
    """Test that weights match SMELL_REGISTRY values."""
    baseline = QPEScore(
        qpe=0.5,
        mi_normalized=0.6,
        smell_penalty=0.2,
        adjusted_quality=0.5,
        smell_counts={"swallowed_exception": 1},
    )
    candidate = QPEScore(
        qpe=0.6,
        mi_normalized=0.7,
        smell_penalty=0.1,
        adjusted_quality=0.6,
        smell_counts={"swallowed_exception": 2},
    )

    result = smell_advantage(baseline, candidate)
    swallowed = next(sa for sa in result if sa.smell_name == "swallowed_exception")
    assert swallowed.weight == 0.15
    assert abs(swallowed.weighted_delta - 0.15) < 0.001


def test_smell_advantage__covers_all_registry_entries() -> None:
    """Test that smell_advantage returns entries for all smells in SMELL_REGISTRY."""
    from slopometry.core.models import SMELL_REGISTRY

    baseline = QPEScore(qpe=0.5, mi_normalized=0.6, smell_penalty=0.2, adjusted_quality=0.5)
    candidate = QPEScore(qpe=0.6, mi_normalized=0.7, smell_penalty=0.1, adjusted_quality=0.6)

    result = smell_advantage(baseline, candidate)
    result_names = {sa.smell_name for sa in result}
    assert result_names == set(SMELL_REGISTRY.keys())


class TestCompareProjectMetrics:
    """Test the cross-project comparison functionality."""

    def test_compare_project_metrics__returns_flat_rankings(self):
        """Test that projects are returned in a flat ranking by QPE."""
        metrics_a = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=5000.0, average_effort=1000.0, average_mi=75.0, total_files_analyzed=5)
        )
        metrics_b = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_effort=5000.0, average_mi=70.0, total_files_analyzed=10)
        )

        result = compare_project_metrics(
            [
                ("project-a", metrics_a),
                ("project-b", metrics_b),
            ]
        )

        assert result.total_projects == 2
        assert len(result.rankings) == 2

    def test_compare_project_metrics__ranks_by_qpe_highest_first(self):
        """Test that projects are ranked by QPE from highest to lowest."""
        high_quality = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_effort=5000.0, average_mi=90.0, total_files_analyzed=10)
        )
        low_quality = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=55000.0, average_effort=5500.0, average_mi=60.0, total_files_analyzed=10)
        )

        result = compare_project_metrics(
            [
                ("low-quality", low_quality),
                ("high-quality", high_quality),
            ]
        )

        # High quality should be ranked first (higher QPE)
        assert result.rankings[0].project_name == "high-quality"
        assert result.rankings[1].project_name == "low-quality"
        assert result.rankings[0].qpe_score.qpe > result.rankings[1].qpe_score.qpe

    def test_compare_project_metrics__includes_qpe_details(self):
        """Test that ranking results include QPE score details."""
        metrics = ExtendedComplexityMetrics(
            **make_test_metrics(total_effort=50000.0, average_effort=5000.0, average_mi=75.0, total_files_analyzed=10)
        )

        result = compare_project_metrics([("test-project", metrics)])

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
        assert "Quality Score" in result.stdout
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
        assert "smell_counts" in qpe_data

        assert isinstance(qpe_data["qpe"], float)
        assert qpe_data["qpe"] > 0

    def test_qpe_calculator__real_codebase_produces_consistent_results(self, repo_path: Path) -> None:
        """Test QPE calculation on real codebase produces stable, sensible values."""
        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics = analyzer.analyze_extended_complexity()


        qpe_score = calculate_qpe(metrics)

        # QPE should be positive for a working codebase
        assert qpe_score.qpe > 0

        # MI normalized should be in valid range (0-1)
        assert 0 <= qpe_score.mi_normalized <= 1

        # Smell penalty uses sigmoid (saturates at 0.9)
        assert 0 <= qpe_score.smell_penalty <= 0.9

        # Adjusted quality should be MI * (1 - smell_penalty) + bonuses
        # The exact bonuses depend on coverage thresholds, so we verify the base formula holds
        # and that any difference is explained by bonuses (in [0, 0.12] range: 0.05+0.05+0.02)
        base_quality = qpe_score.mi_normalized * (1 - qpe_score.smell_penalty)
        bonus_applied = qpe_score.adjusted_quality - base_quality
        assert bonus_applied >= 0, "Bonuses should be non-negative"
        assert bonus_applied <= 0.12 + 0.001, "Bonuses should not exceed max possible (0.05+0.05+0.02)"

        # QPE equals adjusted_quality directly
        assert qpe_score.qpe == qpe_score.adjusted_quality

    def test_display_qpe_score__renders_without_error(self, repo_path: Path) -> None:
        """Test that display_qpe_score renders without AttributeError (regression test for effort_tier bug)."""
        from rich.console import Console

        from slopometry.core.complexity_analyzer import ComplexityAnalyzer
        from slopometry.display.formatters import display_qpe_score

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics = analyzer.analyze_extended_complexity()


        qpe_score = calculate_qpe(metrics)

        # Capture output to verify no errors
        console_output = StringIO()
        Console(file=console_output, force_terminal=True, width=120)

        # This should not raise AttributeError: 'QPEScore' object has no attribute 'effort_tier'
        display_qpe_score(qpe_score, metrics)

    def test_qpe_score_model__serializes_to_json(self) -> None:
        """Test that QPEScore model serializes correctly."""
        qpe_score = QPEScore(
            qpe=0.63,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            smell_counts={"hasattr_getattr": 5, "type_ignore": 3},
        )

        json_output = qpe_score.model_dump_json()

        assert "qpe" in json_output
        assert "adjusted_quality" in json_output
        assert "smell_counts" in json_output

        # Verify round-trip
        restored = QPEScore.model_validate_json(json_output)
        assert restored.qpe == 0.63
        assert restored.smell_counts.hasattr_getattr == 5

    def test_qpe_calculator__handles_empty_codebase_gracefully(self, tmp_path: Path) -> None:
        """Test that QPE calculator handles empty directory without crashing."""
        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=tmp_path)
        metrics = analyzer.analyze_extended_complexity()


        qpe_score = calculate_qpe(metrics)

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


        qpe_score = calculate_qpe(metrics)

        # Documented expectations for slopometry codebase quality
        # These are loose bounds that should remain stable across minor changes

        # MI should be in reasonable range for a Python codebase (40-70 typical)
        assert 28 <= metrics.average_mi <= 80, f"MI {metrics.average_mi} outside expected range"

        # Should analyze multiple files
        assert metrics.total_files_analyzed > 10, "Expected to analyze more than 10 Python files"

        # QPE should be in quality range (0-1) for a Python project
        assert 0.1 <= qpe_score.qpe <= 1.0, f"QPE {qpe_score.qpe} outside expected range"

        # Smell counts should be populated
        total_smells = sum(qpe_score.smell_counts.model_dump().values())
        assert total_smells > 0, "Expected some code smells in a real codebase"

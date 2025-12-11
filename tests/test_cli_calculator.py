"""Tests for CLI Calculator functionality."""

from slopometry.core.models import ExtendedComplexityMetrics
from slopometry.summoner.services.cli_calculator import CLICalculator


class TestCLICalculator:
    """Test the CLI (Completeness Likelihood Improval) calculator."""

    def test_perfect_match_gives_high_score__returns_1_0_when_metrics_match(self):
        """Test that perfect match gives score of 1.0 when metrics match."""
        calculator = CLICalculator()

        metrics = ExtendedComplexityMetrics(
            total_complexity=100,
            total_volume=500.0,
            average_difficulty=12.5,
            total_effort=6250.0,
            average_mi=60.0,
            total_files_analyzed=5,
        )

        cli_score, components = calculator.calculate_cli(metrics, metrics)

        assert cli_score == 1.0
        assert components["complexity"] == 1.0
        assert components["halstead"] == 1.0
        assert components["maintainability"] == 1.0

    def test_undershooting_target_gives_partial_score__returns_proportional_score_when_current_less_than_target(self):
        """Test that undershooting target gives proportional score when current less than target."""
        calculator = CLICalculator()

        current = ExtendedComplexityMetrics(
            total_complexity=50,
            total_volume=250.0,
            average_difficulty=6.25,
            total_effort=1562.5,
            average_mi=30.0,
            total_files_analyzed=5,
        )

        target = ExtendedComplexityMetrics(
            total_complexity=100,
            total_volume=500.0,
            average_difficulty=12.5,
            total_effort=3125.0,
            average_mi=60.0,
            total_files_analyzed=5,
        )

        cli_score, components = calculator.calculate_cli(current, target)

        # Should be around 0.5 for complexity metrics and 0.5 for MI
        assert 0.4 < cli_score < 0.6
        assert components["complexity"] == 0.5
        assert components["maintainability"] == 0.5

    def test_overshooting_target_gives_penalty__returns_negative_score_when_current_exceeds_target(self):
        """Test that overshooting target gives negative penalty when current exceeds target."""
        calculator = CLICalculator()

        current = ExtendedComplexityMetrics(
            total_complexity=200,
            total_volume=1000.0,
            average_difficulty=25.0,
            total_effort=25000.0,
            average_mi=120.0,
            total_files_analyzed=5,
        )

        target = ExtendedComplexityMetrics(
            total_complexity=100,
            total_volume=500.0,
            average_difficulty=12.5,
            total_effort=12500.0,
            average_mi=60.0,
            total_files_analyzed=5,
        )

        cli_score, components = calculator.calculate_cli(current, target)

        # Should be negative due to penalty
        assert cli_score < 0
        assert components["complexity"] < 0
        assert components["halstead"] < 0
        assert components["maintainability"] == 1.0  # MI higher is better

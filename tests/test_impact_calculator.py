"""Tests for the impact calculator."""

from datetime import datetime

import pytest

from slopometry.core.models import (
    ComplexityDelta,
    ExtendedComplexityMetrics,
    HistoricalMetricStats,
    ImpactCategory,
    RepoBaseline,
)
from slopometry.summoner.services.impact_calculator import ImpactCalculator


@pytest.fixture
def calculator():
    return ImpactCalculator()


@pytest.fixture
def baseline_metrics():
    return ExtendedComplexityMetrics(
        total_complexity=100,
        average_complexity=5.0,
        max_complexity=20,
        min_complexity=1,
        total_volume=5000.0,
        total_effort=2500000.0,
        average_volume=250.0,
        average_effort=125000.0,
        average_difficulty=25.0,
        total_mi=1500.0,
        average_mi=75.0,
        total_files_analyzed=20,
    )


@pytest.fixture
def repo_baseline(baseline_metrics):
    return RepoBaseline(
        repository_path="/test/repo",
        computed_at=datetime.now(),
        head_commit_sha="abc123",
        total_commits_analyzed=50,
        cc_delta_stats=HistoricalMetricStats(
            metric_name="cc_delta",
            mean=2.0,
            std_dev=3.0,
            median=1.0,
            min_value=-5.0,
            max_value=15.0,
            sample_count=50,
            trend_coefficient=0.01,
        ),
        effort_delta_stats=HistoricalMetricStats(
            metric_name="effort_delta",
            mean=50.0,
            std_dev=25.0,
            median=40.0,
            min_value=-10.0,
            max_value=150.0,
            sample_count=50,
            trend_coefficient=-0.5,
        ),
        mi_delta_stats=HistoricalMetricStats(
            metric_name="mi_delta",
            mean=-0.5,
            std_dev=1.0,
            median=-0.3,
            min_value=-3.0,
            max_value=2.0,
            sample_count=50,
            trend_coefficient=-0.01,
        ),
        current_metrics=baseline_metrics,
    )


class TestImpactCalculator:
    def test_calculate_impact__returns_neutral_when_deltas_match_baseline_average(self, calculator, repo_baseline):
        """When deltas match baseline mean exactly, z-scores are zero and impact is neutral."""
        delta = ComplexityDelta(
            total_complexity_change=2,
            avg_complexity_change=2.0,  # Matches mean (which is 2.0 in the fixture)
            total_volume_change=100.0,
            avg_volume_change=5.0,
            avg_difficulty_change=0.5,
            total_effort_change=1000.0,
            avg_effort_change=50.0,  # Matches mean
            total_mi_change=-0.5,
            avg_mi_change=-0.5,  # Matches mean
            net_files_change=0,
        )

        assessment = calculator.calculate_impact(delta, repo_baseline)

        assert assessment.impact_category == ImpactCategory.NEUTRAL
        assert abs(assessment.cc_z_score) < 0.01
        assert abs(assessment.effort_z_score) < 0.01
        assert abs(assessment.mi_z_score) < 0.01
        assert abs(assessment.impact_score) < 0.1

    def test_calculate_impact__returns_improvement_when_all_metrics_better_than_average(
        self, calculator, repo_baseline
    ):
        """When all metrics are better than average, should return positive impact score."""
        delta = ComplexityDelta(
            total_complexity_change=-4,
            avg_complexity_change=-4.0,  # Below average (good, mean is 2)
            total_volume_change=-100.0,
            avg_volume_change=-5.0,
            avg_difficulty_change=-0.5,
            total_effort_change=-500.0,
            avg_effort_change=-25.0,  # Below average (good)
            total_mi_change=2.0,
            avg_mi_change=2.0,  # Above average (good, higher MI is better)
            net_files_change=0,
        )

        assessment = calculator.calculate_impact(delta, repo_baseline)

        # CC z-score should be negative (below mean)
        assert assessment.cc_z_score < 0
        # Effort z-score should be negative (below mean)
        assert assessment.effort_z_score < 0
        # MI z-score should be positive (above mean, which is good)
        assert assessment.mi_z_score > 0
        # Overall impact should be positive (improvement)
        assert assessment.impact_score > 0
        assert assessment.impact_category in (
            ImpactCategory.MINOR_IMPROVEMENT,
            ImpactCategory.SIGNIFICANT_IMPROVEMENT,
        )

    def test_calculate_impact__returns_degradation_when_all_metrics_worse_than_average(self, calculator, repo_baseline):
        """When all metrics are worse than average, should return negative impact score."""
        delta = ComplexityDelta(
            total_complexity_change=10,
            avg_complexity_change=10.0,  # Above average (bad, mean is 2.0)
            total_volume_change=500.0,
            avg_volume_change=25.0,
            avg_difficulty_change=2.5,
            total_effort_change=3000.0,
            avg_effort_change=150.0,  # Above average (bad)
            total_mi_change=-2.0,
            avg_mi_change=-2.0,  # Below average (bad, lower MI is worse)
            net_files_change=0,
        )

        assessment = calculator.calculate_impact(delta, repo_baseline)

        # CC z-score should be positive (above mean, bad)
        assert assessment.cc_z_score > 0
        # Effort z-score should be positive (above mean, bad)
        assert assessment.effort_z_score > 0
        # MI z-score should be negative (below mean, bad)
        assert assessment.mi_z_score < 0
        # Overall impact should be negative (degradation)
        assert assessment.impact_score < 0
        assert assessment.impact_category in (
            ImpactCategory.MINOR_DEGRADATION,
            ImpactCategory.SIGNIFICANT_DEGRADATION,
        )

    def test_safe_z_score__returns_zero_when_std_is_zero_and_value_equals_mean(self, calculator):
        """When std dev is zero and value equals mean, z-score is zero."""
        z = calculator._safe_z_score(5.0, 5.0, 0.0)
        assert z == 0.0

    def test_safe_z_score__returns_clamped_value_when_std_is_zero_and_value_differs(self, calculator):
        """When std dev is zero but value differs from mean, returns ±1."""
        z_above = calculator._safe_z_score(10.0, 5.0, 0.0)
        assert z_above == 1.0

        z_below = calculator._safe_z_score(1.0, 5.0, 0.0)
        assert z_below == -1.0

    def test_categorize_impact__returns_significant_improvement_above_threshold(self, calculator):
        """Score > 1.0 is significant improvement."""
        assert calculator._categorize_impact(1.5) == ImpactCategory.SIGNIFICANT_IMPROVEMENT

    def test_categorize_impact__returns_minor_improvement_in_range(self, calculator):
        """Score between 0.5 and 1.0 is minor improvement."""
        assert calculator._categorize_impact(0.75) == ImpactCategory.MINOR_IMPROVEMENT

    def test_categorize_impact__returns_neutral_in_range(self, calculator):
        """Score between -0.5 and 0.5 is neutral."""
        assert calculator._categorize_impact(0.0) == ImpactCategory.NEUTRAL
        assert calculator._categorize_impact(0.4) == ImpactCategory.NEUTRAL
        assert calculator._categorize_impact(-0.4) == ImpactCategory.NEUTRAL

    def test_categorize_impact__returns_minor_degradation_in_range(self, calculator):
        """Score between -1.0 and -0.5 is minor degradation."""
        assert calculator._categorize_impact(-0.75) == ImpactCategory.MINOR_DEGRADATION

    def test_categorize_impact__returns_significant_degradation_below_threshold(self, calculator):
        """Score < -1.0 is significant degradation."""
        assert calculator._categorize_impact(-1.5) == ImpactCategory.SIGNIFICANT_DEGRADATION

    def test_calculate_impact__preserves_raw_deltas_in_assessment(self, calculator, repo_baseline):
        """Assessment should contain the original raw delta values."""
        delta = ComplexityDelta(
            total_complexity_change=7,
            avg_complexity_change=0.3,
            total_volume_change=200.0,
            avg_volume_change=10.0,
            avg_difficulty_change=1.0,
            total_effort_change=1500.0,
            avg_effort_change=75.0,
            total_mi_change=-0.8,
            avg_mi_change=-0.8,
            net_files_change=2,
        )

        assessment = calculator.calculate_impact(delta, repo_baseline)

        assert assessment.cc_delta == 0.3
        assert assessment.effort_delta == 75.0
        assert assessment.mi_delta == -0.8

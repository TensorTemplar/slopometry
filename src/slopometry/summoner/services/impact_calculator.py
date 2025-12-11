"""Impact calculator for staged changes analysis."""

from slopometry.core.models import (
    ComplexityDelta,
    ImpactAssessment,
    ImpactCategory,
    RepoBaseline,
)


class ImpactCalculator:
    """Calculates impact score for staged changes against baseline."""

    # Use existing CLI weights
    CC_WEIGHT = 0.40
    HALSTEAD_WEIGHT = 0.35  # Applied to Effort
    MI_WEIGHT = 0.25

    def calculate_impact(
        self,
        staged_delta: ComplexityDelta,
        baseline: RepoBaseline,
    ) -> ImpactAssessment:
        """Calculate impact assessment using Z-score approach.

        The impact score represents how the staged changes compare to
        the repository's historical average commit:
        - Positive score = better than average (improvements)
        - Negative score = worse than average (degradation)

        Formula:
        1. Compute Z-score for each metric delta
        2. Normalize directions (lower CC/Effort is better, higher MI is better)
        3. Weighted composite using CLI weights

        Args:
            staged_delta: Complexity delta from staged changes
            baseline: Repository baseline statistics

        Returns:
            ImpactAssessment with Z-scores and composite score
        """

        # Use average complexity change to avoid penalizing features with many simple files
        cc_delta = staged_delta.avg_complexity_change
        effort_delta = staged_delta.avg_effort_change
        mi_delta = staged_delta.avg_mi_change

        cc_z = self._safe_z_score(
            cc_delta,
            baseline.cc_delta_stats.mean,
            baseline.cc_delta_stats.std_dev,
        )
        effort_z = self._safe_z_score(
            effort_delta,
            baseline.effort_delta_stats.mean,
            baseline.effort_delta_stats.std_dev,
        )
        mi_z = self._safe_z_score(
            mi_delta,
            baseline.mi_delta_stats.mean,
            baseline.mi_delta_stats.std_dev,
        )

        # Normalize directions for impact score:
        # For CC/Effort: negative z-score = below average increase = GOOD
        #   So we negate: -z makes "below average" positive
        # For MI: positive z-score = above average increase = GOOD
        #   So we keep as-is

        normalized_cc = -cc_z
        normalized_effort = -effort_z
        normalized_mi = mi_z

        impact_score = (
            self.CC_WEIGHT * normalized_cc + self.HALSTEAD_WEIGHT * normalized_effort + self.MI_WEIGHT * normalized_mi
        )

        impact_category = self._categorize_impact(impact_score)

        return ImpactAssessment(
            cc_z_score=cc_z,
            effort_z_score=effort_z,
            mi_z_score=mi_z,
            impact_score=impact_score,
            impact_category=impact_category,
            cc_delta=cc_delta,
            effort_delta=effort_delta,
            mi_delta=mi_delta,
        )

    def _safe_z_score(self, value: float, mean: float, std: float) -> float:
        """Compute z-score with handling for zero std dev.

        When std dev is essentially zero (all values the same),
        we return a clamped value to indicate above/below mean.

        Args:
            value: Observed value
            mean: Mean of the distribution
            std: Standard deviation of the distribution

        Returns:
            Z-score (clamped for edge cases)
        """
        if std < 1e-10:  # Essentially zero variance
            if abs(value - mean) < 1e-10:
                return 0.0  # At the mean
            return 1.0 if value > mean else -1.0  # Above or below mean

        return (value - mean) / std

    def _categorize_impact(self, score: float) -> ImpactCategory:
        """Categorize impact score into human-readable category.

        Categories:
        - > 1.0: Significant improvement
        - > 0.5: Minor improvement
        - [-0.5, 0.5]: Neutral (within typical variance)
        - < -0.5: Minor degradation
        - < -1.0: Significant degradation
        """
        if score > 1.0:
            return ImpactCategory.SIGNIFICANT_IMPROVEMENT
        elif score > 0.5:
            return ImpactCategory.MINOR_IMPROVEMENT
        elif score >= -0.5:
            return ImpactCategory.NEUTRAL
        elif score >= -1.0:
            return ImpactCategory.MINOR_DEGRADATION
        else:
            return ImpactCategory.SIGNIFICANT_DEGRADATION

"""Impact calculator for staged changes analysis."""

from slopometry.core.models.baseline import ImpactAssessment, ImpactCategory, RepoBaseline
from slopometry.core.models.complexity import ComplexityDelta
from slopometry.core.settings import settings


class ImpactCalculator:
    """Calculates impact score for staged changes against baseline.

    MI is weighted highest (50% default) as it's already a balanced composite metric
    that factors in CC, Volume, and LOC. This prevents raw CC/Effort from
    dominating the score for sessions that add many files.

    Weights are configurable via settings (SLOPOMETRY_IMPACT_CC_WEIGHT, etc.).
    """

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
        3. Weighted composite with MI at 50%, CC/Effort at 25% each

        Args:
            staged_delta: Complexity delta from staged changes
            baseline: Repository baseline statistics

        Returns:
            ImpactAssessment with Z-scores and composite score
        """
        # Use totals to match baseline stats (which track total delta per commit)
        cc_delta = staged_delta.total_complexity_change
        effort_delta = staged_delta.total_effort_change
        mi_delta = staged_delta.total_mi_change

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

        qpe_delta = staged_delta.qpe_change
        qpe_z = 0.0
        if baseline.qpe_stats:
            qpe_z = self._safe_z_score(
                qpe_delta,
                baseline.qpe_stats.mean,
                baseline.qpe_stats.std_dev,
            )

        token_delta = staged_delta.total_tokens_change
        token_z = 0.0
        if baseline.token_delta_stats:
            token_z = self._safe_z_score(
                token_delta,
                baseline.token_delta_stats.mean,
                baseline.token_delta_stats.std_dev,
            )

        # NOTE: Normalize z-score directions for impact scoring:
        # CC/Effort: negate (lower=better), MI/QPE: keep (higher=better)
        normalized_cc = -cc_z
        normalized_effort = -effort_z
        normalized_mi = mi_z

        impact_score = (
            settings.impact_cc_weight * normalized_cc
            + settings.impact_effort_weight * normalized_effort
            + settings.impact_mi_weight * normalized_mi
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
            qpe_delta=qpe_delta,
            qpe_z_score=qpe_z,
            token_delta=token_delta,
            token_z_score=token_z,
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
        if std < 1e-10:
            if abs(value - mean) < 1e-10:
                return 0.0
            return 1.0 if value > mean else -1.0

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

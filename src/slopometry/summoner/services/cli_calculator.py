"""CLI (Completeness Likelihood Improval) calculator for experiment tracking."""

from slopometry.core.models import ExtendedComplexityMetrics


class CLICalculator:
    """Calculates Completeness Likelihood Improval score."""

    def calculate_cli(
        self, current: ExtendedComplexityMetrics, target: ExtendedComplexityMetrics
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate CLI score where:
        - 1.0 = perfect match to target
        - 0.0-1.0 = approaching target
        - <0 = overshooting target (penalized)

        Args:
            current: Current metrics from agent's code
            target: Target metrics from HEAD commit

        Returns:
            Tuple of (cli_score, component_scores)
        """
        # Cyclomatic Complexity Score (lower is better, but target is goal)
        complexity_ratio = current.total_complexity / max(target.total_complexity, 1)
        complexity_score = self._score_with_penalty(complexity_ratio, optimal=1.0)

        # Halstead Score (composite of volume, difficulty, effort)
        volume_ratio = current.total_volume / max(target.total_volume, 1)
        difficulty_ratio = current.total_difficulty / max(target.total_difficulty, 1)
        effort_ratio = current.total_effort / max(target.total_effort, 1)

        halstead_score = (
            self._score_with_penalty(volume_ratio, optimal=1.0) * 0.4
            + self._score_with_penalty(difficulty_ratio, optimal=1.0) * 0.3
            + self._score_with_penalty(effort_ratio, optimal=1.0) * 0.3
        )

        # Maintainability Index Score (higher is better)
        # MI is on 0-100 scale where higher = more maintainable
        mi_ratio = current.average_mi / max(target.average_mi, 1) if target.average_mi > 0 else 0.0
        mi_score = self._score_with_penalty(mi_ratio, optimal=1.0, higher_is_better=True)

        # Weighted CLI score
        cli_score = complexity_score * 0.4 + halstead_score * 0.35 + mi_score * 0.25

        component_scores = {
            "complexity": complexity_score,
            "halstead": halstead_score,
            "maintainability": mi_score,
        }

        return cli_score, component_scores

    def _score_with_penalty(
        self, ratio: float, optimal: float = 1.0, penalty_factor: float = 2.0, higher_is_better: bool = False
    ) -> float:
        """
        Score with asymmetric penalty for overshooting.

        Args:
            ratio: current/target ratio
            optimal: optimal ratio (usually 1.0)
            penalty_factor: how much to penalize overshooting
            higher_is_better: for metrics like MI where higher is better

        Returns:
            Score between -infinity and 1.0
        """
        if higher_is_better:
            # For MI: going below target is bad
            if ratio >= optimal:
                return 1.0  # At or above target is perfect
            else:
                return ratio  # Linear decrease below target
        else:
            # For complexity metrics: going above target is bad
            if ratio <= optimal:
                return ratio / optimal  # Approaching target
            else:
                # Penalty for overshooting
                overshoot = ratio - optimal
                return 1.0 - (overshoot * penalty_factor)

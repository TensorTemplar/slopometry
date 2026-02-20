"""CLI (Completeness Likelihood Improval) calculator for experiment tracking.

DEPRECATED: The CLI score has known issues:
- Double-counting: CC + Halstead + MI, but MI already incorporates CC and Volume
- Scale-sensitive: Ratio-based scoring penalizes differently based on target magnitude
- Unbounded output: Not suitable for stable RL training

Use calculate_qpe() from slopometry.summoner.services.qpe_calculator instead.
"""

import warnings

from slopometry.core.models.baseline import QPEScore
from slopometry.core.models.complexity import ExtendedComplexityMetrics
from slopometry.summoner.services.qpe_calculator import calculate_qpe as _calculate_qpe


class CLICalculator:
    """Calculates Completeness Likelihood Improval score.

    DEPRECATED: Use calculate_qpe() instead. See qpe_calculator.py for the
    principled replacement that:
    - Uses MI as sole quality signal (no double-counting)
    - Normalizes by Halstead Effort for fair comparison
    - Produces bounded output suitable for GRPO
    """

    def calculate_qpe(self, metrics: ExtendedComplexityMetrics) -> QPEScore:
        """Calculate Quality-Per-Effort score (recommended).

        This is the principled replacement for calculate_cli().

        Args:
            metrics: Extended complexity metrics for the codebase

        Returns:
            QPEScore with component breakdown
        """
        return _calculate_qpe(metrics)

    def calculate_cli(
        self, current: ExtendedComplexityMetrics, target: ExtendedComplexityMetrics
    ) -> tuple[float, dict[str, float]]:
        """Calculate CLI score (DEPRECATED - use calculate_qpe instead).

        Issues with this method:
        - Double-counts CC and Volume (already in MI)
        - Scale-sensitive ratio comparisons
        - Unbounded output not suitable for RL

        Use calculate_qpe() for principled quality measurement.

        Args:
            current: Current metrics from agent's code
            target: Target metrics from HEAD commit

        Returns:
            Tuple of (cli_score, component_scores)
        """
        warnings.warn(
            "calculate_cli() is deprecated. Use calculate_qpe() for principled quality measurement.",
            DeprecationWarning,
            stacklevel=2,
        )
        complexity_ratio = current.total_complexity / max(target.total_complexity, 1)
        complexity_score = self._score_with_penalty(complexity_ratio, optimal=1.0)

        volume_ratio = current.total_volume / max(target.total_volume, 1)
        difficulty_ratio = current.average_difficulty / max(target.average_difficulty, 1)
        effort_ratio = current.total_effort / max(target.total_effort, 1)

        halstead_score = (
            self._score_with_penalty(volume_ratio, optimal=1.0) * 0.4
            + self._score_with_penalty(difficulty_ratio, optimal=1.0) * 0.3
            + self._score_with_penalty(effort_ratio, optimal=1.0) * 0.3
        )

        mi_ratio = current.average_mi / max(target.average_mi, 1) if target.average_mi > 0 else 0.0
        mi_score = self._score_with_penalty(mi_ratio, optimal=1.0, higher_is_better=True)

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

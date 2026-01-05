"""Quality-Per-Effort (QPE) calculator for principled code quality comparison.

QPE provides a single metric for:
1. GRPO rollout comparison (same-spec implementations)
2. Cross-project comparison

Key properties:
- Uses MI as sole quality signal (no double-counting with CC/Volume)
- Normalizes by Halstead Effort for fair comparison
- Includes code smell penalties with explicit weights
- Bounded output via tanh for stable RL training
"""

import math
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import (
    CrossProjectComparison,
    ExtendedComplexityMetrics,
    ProjectQPEResult,
    QPEScore,
)


class QPECalculator:
    """Quality-Per-Effort calculator for principled comparison."""

    # Smell weights with explicit rationale
    # Sum to ~0.7 so maximum penalty (all smells present) approaches 0.5 cap
    SMELL_WEIGHTS: dict[str, float] = {
        "hasattr_getattr": 0.10,  # Indicates missing domain models
        "swallowed_exception": 0.15,  # Can hide real bugs
        "type_ignore": 0.08,  # Type system bypass
        "dynamic_execution": 0.12,  # Security/maintainability risk
        "test_skip": 0.10,  # Missing coverage
        "dict_get_with_default": 0.05,  # Minor modeling gap
        "inline_import": 0.03,  # Style issue
        "orphan_comment": 0.02,  # Documentation noise
        "untracked_todo": 0.02,  # Debt tracking
        "nonempty_init": 0.03,  # Structural issue
    }

    def calculate_qpe(self, metrics: ExtendedComplexityMetrics) -> QPEScore:
        """Calculate Quality-Per-Effort score.

        Formula:
            QPE = adjusted_quality / effort_factor

        Where:
            adjusted_quality = mi_normalized * (1 - smell_penalty)
            mi_normalized = average_mi / 100.0
            smell_penalty = min(weighted_smell_sum / files_analyzed, 0.5)
            effort_factor = log(total_halstead_effort + 1)

        Args:
            metrics: Extended complexity metrics for the codebase

        Returns:
            QPEScore with component breakdown
        """
        # 1. Quality signal: MI (0-100) normalized to 0-1
        mi_normalized = metrics.average_mi / 100.0

        # 2. Collect smell counts and compute weighted penalty
        smell_counts: dict[str, int] = {
            "hasattr_getattr": metrics.hasattr_getattr_count,
            "swallowed_exception": metrics.swallowed_exception_count,
            "type_ignore": metrics.type_ignore_count,
            "dynamic_execution": metrics.dynamic_execution_count,
            "test_skip": metrics.test_skip_count,
            "dict_get_with_default": metrics.dict_get_with_default_count,
            "inline_import": metrics.inline_import_count,
            "orphan_comment": metrics.orphan_comment_count,
            "untracked_todo": metrics.untracked_todo_count,
            "nonempty_init": metrics.nonempty_init_count,
        }

        weighted_smell_sum = sum(smell_counts[smell_name] * weight for smell_name, weight in self.SMELL_WEIGHTS.items())

        # Normalize by file count and cap at 0.5
        files_analyzed = max(metrics.total_files_analyzed, 1)
        smell_penalty = min(weighted_smell_sum / files_analyzed, 0.5)

        # 3. Adjusted quality
        adjusted_quality = mi_normalized * (1 - smell_penalty)

        # 4. Effort normalization using log for diminishing returns
        effort_factor = math.log(metrics.total_effort + 1)

        # 5. QPE: quality per log-effort (higher = better)
        qpe = adjusted_quality / effort_factor if effort_factor > 0 else 0.0

        return QPEScore(
            qpe=qpe,
            mi_normalized=mi_normalized,
            smell_penalty=smell_penalty,
            adjusted_quality=adjusted_quality,
            effort_factor=effort_factor,
            smell_counts=smell_counts,
        )


def grpo_advantage(baseline: QPEScore, candidate: QPEScore) -> float:
    """Compute advantage for GRPO (Group Relative Policy Optimization).

    Compares two implementations of the same spec and returns a bounded
    advantage value suitable for RL training.

    Args:
        baseline: QPE score of the baseline implementation
        candidate: QPE score of the candidate implementation

    Returns:
        Bounded value in (-1, 1) where:
        - Positive = candidate is better than baseline
        - Negative = candidate is worse than baseline
        - Zero = equivalent quality
    """
    qpe_delta = candidate.qpe - baseline.qpe

    # Normalize by baseline QPE for relative comparison
    if baseline.qpe > 0:
        relative_improvement = qpe_delta / baseline.qpe
    else:
        # Baseline is zero or negative, use absolute delta
        relative_improvement = qpe_delta

    # Apply tanh for bounded output in (-1, 1)
    return math.tanh(relative_improvement)


class CrossProjectComparator:
    """Compare multiple projects using QPE."""

    def __init__(self) -> None:
        self.qpe_calculator = QPECalculator()

    def compare(
        self,
        project_paths: list[Path],
    ) -> CrossProjectComparison:
        """Compare projects by QPE, ranked from highest to lowest.

        Args:
            project_paths: List of paths to project directories

        Returns:
            CrossProjectComparison with flat rankings
        """
        results: list[ProjectQPEResult] = []

        for project_path in project_paths:
            analyzer = ComplexityAnalyzer(working_directory=project_path)
            metrics = analyzer.analyze_extended_complexity()
            qpe_score = self.qpe_calculator.calculate_qpe(metrics)

            results.append(
                ProjectQPEResult(
                    project_path=str(project_path),
                    project_name=project_path.name,
                    qpe_score=qpe_score,
                    metrics=metrics,
                )
            )

        # Sort by QPE (highest first)
        rankings = sorted(results, key=lambda x: x.qpe_score.qpe, reverse=True)

        return CrossProjectComparison(
            total_projects=len(results),
            rankings=rankings,
        )

    def compare_metrics(
        self,
        metrics_list: list[tuple[str, ExtendedComplexityMetrics]],
    ) -> CrossProjectComparison:
        """Compare pre-computed metrics by QPE.

        Useful when metrics are already available (e.g., from database).

        Args:
            metrics_list: List of (project_name, metrics) tuples

        Returns:
            CrossProjectComparison with flat rankings
        """
        results: list[ProjectQPEResult] = []

        for project_name, metrics in metrics_list:
            qpe_score = self.qpe_calculator.calculate_qpe(metrics)

            results.append(
                ProjectQPEResult(
                    project_path="",
                    project_name=project_name,
                    qpe_score=qpe_score,
                    metrics=metrics,
                )
            )

        # Sort by QPE (highest first)
        rankings = sorted(results, key=lambda x: x.qpe_score.qpe, reverse=True)

        return CrossProjectComparison(
            total_projects=len(results),
            rankings=rankings,
        )

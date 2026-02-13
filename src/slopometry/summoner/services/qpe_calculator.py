"""Quality (QPE) calculator for principled code quality comparison.

Single metric: adjusted quality = MI * (1 - smell_penalty) + bonuses.
Used for temporal delta tracking, cross-project comparison, and GRPO advantage.

Key properties:
- Uses MI as sole quality signal (no double-counting with CC/Volume)
- Sigmoid-saturated smell penalty (configurable steepness)
- All smells weighted equally regardless of file complexity
- Bounded GRPO advantage via tanh for stable RL training
"""

import math
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer

# Bump this when SMELL_REGISTRY weights or QPE formula parameters change.
# Used to detect stale cached QPE scores computed with old weights.
QPE_WEIGHT_VERSION = "2"

from slopometry.core.models import (
    SMELL_REGISTRY,
    CrossProjectComparison,
    ExtendedComplexityMetrics,
    ProjectQPEResult,
    QPEScore,
    SmellAdvantage,
)
from slopometry.core.settings import settings


def calculate_qpe(metrics: ExtendedComplexityMetrics) -> QPEScore:
    """Calculate quality score.

    Formula:
        qpe = mi_normalized * (1 - smell_penalty) + bonuses

    Where:
        mi_normalized = average_mi / 100.0
        smell_penalty = 0.9 * (1 - exp(-smell_penalty_raw * steepness))
        smell_penalty_raw = weighted_smell_sum / effective_files
        bonuses = test_bonus + type_bonus + docstring_bonus

    Smell penalty uses:
        - Effective files (files with min LOC) to prevent gaming via tiny files
        - No effort multiplier: all smells penalize equally
        - Sigmoid saturation instead of hard cap

    Bonuses (positive signals):
        - Test coverage bonus when >= threshold
        - Type hint coverage bonus when >= threshold
        - Docstring coverage bonus when >= threshold

    Args:
        metrics: Extended complexity metrics for the codebase

    Returns:
        QPEScore with component breakdown
    """
    mi_normalized = metrics.average_mi / 100.0

    smell_counts = metrics.get_smell_counts()
    weighted_smell_sum = 0.0

    for smell in metrics.get_smells():
        weighted_smell_sum += smell.count * smell.weight

    # Use files_by_loc for anti-gaming file filtering, fallback to total_files
    if metrics.files_by_loc:
        effective_files = sum(1 for loc in metrics.files_by_loc.values() if loc >= settings.qpe_min_loc_per_file)
    else:
        effective_files = metrics.total_files_analyzed

    total_files = max(effective_files, 1)
    smell_penalty_raw = weighted_smell_sum / total_files

    # Sigmoid saturation with configurable steepness (approaches 0.9 asymptotically)
    smell_penalty = 0.9 * (1 - math.exp(-smell_penalty_raw * settings.qpe_sigmoid_steepness))

    # Positive bonuses (configurable thresholds and amounts)
    test_bonus = (
        settings.qpe_test_coverage_bonus
        if (metrics.test_coverage_percent or 0) >= settings.qpe_test_coverage_threshold
        else 0.0
    )
    type_bonus = (
        settings.qpe_type_coverage_bonus if metrics.type_hint_coverage >= settings.qpe_type_coverage_threshold else 0.0
    )
    docstring_bonus = (
        settings.qpe_docstring_coverage_bonus
        if metrics.docstring_coverage >= settings.qpe_docstring_coverage_threshold
        else 0.0
    )
    total_bonus = test_bonus + type_bonus + docstring_bonus

    adjusted_quality = mi_normalized * (1 - smell_penalty) + total_bonus

    return QPEScore(
        qpe=adjusted_quality,
        mi_normalized=mi_normalized,
        smell_penalty=smell_penalty,
        adjusted_quality=adjusted_quality,
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


def smell_advantage(baseline: QPEScore, candidate: QPEScore) -> list[SmellAdvantage]:
    """Decompose the aggregate GRPO advantage into per-smell contributions.

    Uses smell weights from SMELL_REGISTRY to compute weighted deltas for each
    smell type. Iterates all registered smells since SmellCounts always has all
    fields (defaulting to 0), so there are no asymmetric key sets.

    The primary signal remains aggregate grpo_advantage(); this decomposition
    is auxiliary for interpretability and optional per-smell reward shaping.

    Args:
        baseline: QPE score of the baseline/reference implementation
        candidate: QPE score of the candidate implementation

    Returns:
        List of SmellAdvantage sorted by absolute weighted_delta (highest impact first)
    """
    advantages = []
    for name, defn in SMELL_REGISTRY.items():
        baseline_count = getattr(baseline.smell_counts, name)
        candidate_count = getattr(candidate.smell_counts, name)
        weighted_delta = (candidate_count - baseline_count) * defn.weight

        advantages.append(
            SmellAdvantage(
                smell_name=name,
                baseline_count=baseline_count,
                candidate_count=candidate_count,
                weight=defn.weight,
                weighted_delta=weighted_delta,
            )
        )

    return sorted(advantages, key=lambda a: abs(a.weighted_delta), reverse=True)


def compare_projects(
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
        qpe_score = calculate_qpe(metrics)

        results.append(
            ProjectQPEResult(
                project_path=str(project_path),
                project_name=project_path.name,
                qpe_score=qpe_score,
                metrics=metrics,
            )
        )

    rankings = sorted(results, key=lambda x: x.qpe_score.qpe, reverse=True)

    return CrossProjectComparison(
        total_projects=len(results),
        rankings=rankings,
    )


def compare_project_metrics(
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
        qpe_score = calculate_qpe(metrics)

        results.append(
            ProjectQPEResult(
                project_path="",
                project_name=project_name,
                qpe_score=qpe_score,
                metrics=metrics,
            )
        )

    rankings = sorted(results, key=lambda x: x.qpe_score.qpe, reverse=True)

    return CrossProjectComparison(
        total_projects=len(results),
        rankings=rankings,
    )

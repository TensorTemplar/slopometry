"""Current (uncommitted) impact analysis service."""

import shutil
from datetime import datetime
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import (
    ComplexityDelta,
    CurrentChangesAnalysis,
    RepoBaseline,
)
from slopometry.core.working_tree_extractor import WorkingTreeExtractor
from slopometry.summoner.services.impact_calculator import ImpactCalculator


class CurrentImpactService:
    """Service for analyzing impact of uncommitted changes."""

    def __init__(self):
        self.impact_calculator = ImpactCalculator()

    def analyze_uncommitted_changes(
        self,
        repo_path: Path,
        baseline: RepoBaseline,
    ) -> CurrentChangesAnalysis | None:
        """Analyze uncommitted changes against repository baseline.

        Args:
            repo_path: Path to the repository
            baseline: Pre-computed repository baseline

        Returns:
            CurrentChangesAnalysis or None if analysis fails
        """
        repo_path = repo_path.resolve()
        extractor = WorkingTreeExtractor(repo_path)
        analyzer = ComplexityAnalyzer(working_directory=repo_path)

        changed_files = extractor.get_changed_python_files()
        if not changed_files:
            return None

        baseline_metrics = baseline.current_metrics

        temp_dir = extractor.extract_working_state()

        if not temp_dir:
            current_metrics = analyzer.analyze_extended_complexity()
        else:
            try:
                current_metrics = analyzer.analyze_extended_complexity(temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        current_delta = self._compute_delta(baseline_metrics, current_metrics)

        assessment = self.impact_calculator.calculate_impact(current_delta, baseline)

        return CurrentChangesAnalysis(
            repository_path=str(repo_path),
            analysis_timestamp=datetime.now(),
            changed_files=changed_files,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            assessment=assessment,
            baseline=baseline,
        )

    def _compute_delta(
        self,
        baseline_metrics,
        current_metrics,
    ) -> ComplexityDelta:
        """Compute complexity delta between baseline and current metrics."""
        return ComplexityDelta(
            total_complexity_change=(current_metrics.total_complexity - baseline_metrics.total_complexity),
            avg_complexity_change=(current_metrics.average_complexity - baseline_metrics.average_complexity),
            total_volume_change=(current_metrics.total_volume - baseline_metrics.total_volume),
            avg_volume_change=(current_metrics.average_volume - baseline_metrics.average_volume),
            avg_difficulty_change=(current_metrics.average_difficulty - baseline_metrics.average_difficulty),
            total_effort_change=(current_metrics.total_effort - baseline_metrics.total_effort),
            avg_effort_change=(current_metrics.average_effort - baseline_metrics.average_effort),
            total_mi_change=current_metrics.total_mi - baseline_metrics.total_mi,
            avg_mi_change=current_metrics.average_mi - baseline_metrics.average_mi,
            net_files_change=(current_metrics.total_files_analyzed - baseline_metrics.total_files_analyzed),
        )


# NOTE: Backwards compatibility alias for renamed service
StagedImpactService = CurrentImpactService

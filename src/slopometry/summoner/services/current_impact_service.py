"""Current (uncommitted) impact analysis service."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import (
    ComplexityDelta,
    CurrentChangesAnalysis,
    GalenMetrics,
    RepoBaseline,
)
from slopometry.core.working_tree_extractor import WorkingTreeExtractor
from slopometry.summoner.services.impact_calculator import ImpactCalculator

logger = logging.getLogger(__name__)


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

        from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

        coverage_analyzer = ContextCoverageAnalyzer(repo_path)
        blind_spots = coverage_analyzer.get_affected_dependents(set(changed_files))

        filtered_coverage = None
        filtered_coverage = None
        try:
            from slopometry.core.coverage_analyzer import CoverageAnalyzer

            cov_analyzer = CoverageAnalyzer(repo_path)
            cov_result = cov_analyzer.analyze_coverage()

            if cov_result.coverage_available and cov_result.file_coverage:
                filtered_coverage = {}
                for file_path in changed_files:
                    if file_path in cov_result.file_coverage:
                        filtered_coverage[file_path] = cov_result.file_coverage[file_path]
        except Exception:
            # Coverage analysis is optional
            pass

        # Calculate token impact
        blind_spot_tokens = 0
        changed_files_tokens = 0

        # Helper to get token count for a file path
        def get_token_count(path_str: str) -> int:
            # path_str is relative
            return current_metrics.files_by_token_count.get(path_str, 0)

        for file_path in changed_files:
            changed_files_tokens += get_token_count(file_path)

        for file_path in blind_spots:
            blind_spot_tokens += get_token_count(file_path)

        complete_picture_context_size = changed_files_tokens + blind_spot_tokens

        # Calculate Galen metrics based on commit history token growth
        galen_metrics = self._calculate_galen_metrics(baseline, current_metrics)

        return CurrentChangesAnalysis(
            repository_path=str(repo_path),
            analysis_timestamp=datetime.now(),
            changed_files=changed_files,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            assessment=assessment,
            baseline=baseline,
            blind_spots=blind_spots,
            filtered_coverage=filtered_coverage,
            blind_spot_tokens=blind_spot_tokens,
            changed_files_tokens=changed_files_tokens,
            complete_picture_context_size=complete_picture_context_size,
            galen_metrics=galen_metrics,
        )

    def _calculate_galen_metrics(
        self,
        baseline: RepoBaseline,
        current_metrics,
    ) -> GalenMetrics | None:
        """Calculate Galen productivity metrics from commit history token growth.

        Uses the baseline's commit date range and oldest commit token count
        to calculate the token productivity rate (Galen Rate).

        Galen Rate = (current_tokens - oldest_commit_tokens) / period_days / GALEN_TOKENS_PER_DAY
        """
        if not baseline.oldest_commit_date or not baseline.newest_commit_date:
            return None

        if baseline.oldest_commit_tokens is None:
            return None

        time_delta = baseline.newest_commit_date - baseline.oldest_commit_date
        period_days = time_delta.total_seconds() / 86400

        if period_days <= 0:
            return None

        tokens_changed = current_metrics.total_tokens - baseline.oldest_commit_tokens

        return GalenMetrics.calculate(tokens_changed=tokens_changed, period_days=period_days)

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

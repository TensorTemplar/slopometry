"""Current (uncommitted) impact analysis service."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.database import EventDatabase
from slopometry.core.models import (
    AnalysisSource,
    ComplexityDelta,
    CurrentChangesAnalysis,
    ExtendedComplexityMetrics,
    GalenMetrics,
    QPEScore,
    RepoBaseline,
    SmellAdvantage,
)
from slopometry.core.working_tree_extractor import WorkingTreeExtractor
from slopometry.core.working_tree_state import WorkingTreeStateCalculator
from slopometry.summoner.services.impact_calculator import ImpactCalculator
from slopometry.summoner.services.qpe_calculator import calculate_qpe, smell_advantage

logger = logging.getLogger(__name__)

CURRENT_IMPACT_SESSION_ID = "current-impact"


class CurrentImpactService:
    """Service for analyzing impact of uncommitted changes."""

    def __init__(self, db: EventDatabase | None = None):
        self.impact_calculator = ImpactCalculator()
        self.db = db or EventDatabase()

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

        wt_calculator = WorkingTreeStateCalculator(repo_path)
        commit_sha = wt_calculator.get_current_commit_sha()
        working_tree_hash = wt_calculator.calculate_working_tree_hash(commit_sha) if commit_sha else None

        current_metrics = self._get_or_compute_metrics(repo_path, commit_sha, working_tree_hash, extractor, analyzer)

        current_delta, smell_advantages, _, _ = self._compute_delta(baseline_metrics, current_metrics)

        assessment = self.impact_calculator.calculate_impact(current_delta, baseline)

        from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

        coverage_analyzer = ContextCoverageAnalyzer(repo_path)
        blind_spots = coverage_analyzer.get_affected_dependents(set(changed_files))

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
        except Exception as e:
            logger.debug(f"Coverage analysis failed (optional feature): {e}")

        blind_spot_tokens = 0
        changed_files_tokens = 0

        def get_token_count(path_str: str) -> int:
            """Get token count for a relative file path."""
            return current_metrics.files_by_token_count.get(path_str, 0)

        for file_path in changed_files:
            changed_files_tokens += get_token_count(file_path)

        for file_path in blind_spots:
            blind_spot_tokens += get_token_count(file_path)

        complete_picture_context_size = changed_files_tokens + blind_spot_tokens

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
            smell_advantages=smell_advantages,
        )

    def analyze_previous_commit(
        self,
        repo_path: Path,
        baseline: RepoBaseline,
    ) -> CurrentChangesAnalysis | None:
        """Analyze the previous commit (HEAD) against its parent (HEAD~1).

        Used as fallback when there are no uncommitted changes.

        Args:
            repo_path: Path to the repository
            baseline: Pre-computed repository baseline

        Returns:
            CurrentChangesAnalysis or None if analysis fails
        """
        from slopometry.core.git_tracker import GitOperationError, GitTracker

        repo_path = repo_path.resolve()
        git_tracker = GitTracker(repo_path)
        analyzer = ComplexityAnalyzer(working_directory=repo_path)

        if not git_tracker.has_previous_commit():
            return None

        head_sha = git_tracker._get_current_commit_sha()
        if not head_sha:
            return None

        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD~1"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.debug(f"git rev-parse HEAD~1 failed: {result.stderr.strip()}")
                return None
            parent_sha = result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to get parent commit SHA: {e}")
            return None

        try:
            changed_files = git_tracker.get_changed_python_files(parent_sha, head_sha)
        except GitOperationError as e:
            logger.debug(f"Failed to get changed files between {parent_sha[:8]} and {head_sha[:8]}: {e}")
            return None

        if not changed_files:
            return None

        try:
            with git_tracker.extract_files_from_commit_ctx(parent_sha) as parent_dir:
                with git_tracker.extract_files_from_commit_ctx(head_sha) as head_dir:
                    if not parent_dir or not head_dir:
                        logger.debug(
                            f"Failed to extract files from commits: parent_dir={parent_dir}, head_dir={head_dir}"
                        )
                        return None

                    parent_metrics = analyzer.analyze_extended_complexity(parent_dir)
                    head_metrics = analyzer.analyze_extended_complexity(head_dir)
        except GitOperationError as e:
            logger.debug(f"Failed to extract or analyze commits {parent_sha[:8]}..{head_sha[:8]}: {e}")
            return None

        # Use parent as baseline, HEAD as current
        current_delta, smell_advantages, _, _ = self._compute_delta(parent_metrics, head_metrics)
        assessment = self.impact_calculator.calculate_impact(current_delta, baseline)

        from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

        coverage_analyzer = ContextCoverageAnalyzer(repo_path)
        blind_spots = coverage_analyzer.get_affected_dependents(set(changed_files))

        blind_spot_tokens = 0
        changed_files_tokens = 0

        for file_path in changed_files:
            changed_files_tokens += head_metrics.files_by_token_count.get(file_path, 0)

        for file_path in blind_spots:
            blind_spot_tokens += head_metrics.files_by_token_count.get(file_path, 0)

        complete_picture_context_size = changed_files_tokens + blind_spot_tokens

        galen_metrics = self._calculate_galen_metrics(baseline, head_metrics)

        return CurrentChangesAnalysis(
            repository_path=str(repo_path),
            analysis_timestamp=datetime.now(),
            source=AnalysisSource.PREVIOUS_COMMIT,
            analyzed_commit_sha=head_sha[:8],
            base_commit_sha=parent_sha[:8],
            changed_files=changed_files,
            current_metrics=head_metrics,
            baseline_metrics=parent_metrics,
            assessment=assessment,
            baseline=baseline,
            blind_spots=blind_spots,
            filtered_coverage=None,  # Coverage not meaningful for committed changes
            blind_spot_tokens=blind_spot_tokens,
            changed_files_tokens=changed_files_tokens,
            complete_picture_context_size=complete_picture_context_size,
            galen_metrics=galen_metrics,
            smell_advantages=smell_advantages,
        )

    def _calculate_galen_metrics(
        self,
        baseline: RepoBaseline,
        current_metrics: ExtendedComplexityMetrics,
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

    def _get_or_compute_metrics(
        self,
        repo_path: Path,
        commit_sha: str | None,
        working_tree_hash: str | None,
        extractor: WorkingTreeExtractor,
        analyzer: ComplexityAnalyzer,
    ) -> ExtendedComplexityMetrics:
        """Get metrics from cache or compute fresh.

        Args:
            repo_path: Path to the repository
            commit_sha: Current commit SHA
            working_tree_hash: Hash of uncommitted changes
            extractor: Working tree extractor for temp dir creation
            analyzer: Complexity analyzer

        Returns:
            ExtendedComplexityMetrics for current state
        """
        from slopometry.core.code_quality_cache import CodeQualityCacheManager

        if commit_sha:
            with self.db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)
                cached_metrics, _ = cache_manager.get_cached_metrics(
                    CURRENT_IMPACT_SESSION_ID, str(repo_path), commit_sha, working_tree_hash
                )
                if cached_metrics:
                    logger.debug("Using cached metrics for current-impact")
                    return cached_metrics

        temp_dir = extractor.extract_working_state()

        if not temp_dir:
            current_metrics = analyzer.analyze_extended_complexity()
        else:
            try:
                current_metrics = analyzer.analyze_extended_complexity(temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if commit_sha:
            with self.db._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)
                cache_manager.save_metrics_to_cache(
                    CURRENT_IMPACT_SESSION_ID,
                    str(repo_path),
                    commit_sha,
                    current_metrics,
                    working_tree_hash=working_tree_hash,
                )
                logger.debug("Cached metrics for current-impact")

        return current_metrics

    def _compute_delta(
        self,
        baseline_metrics: ExtendedComplexityMetrics,
        current_metrics: ExtendedComplexityMetrics,
    ) -> tuple[ComplexityDelta, list[SmellAdvantage], QPEScore, QPEScore]:
        """Compute complexity delta between baseline and current metrics.

        Returns:
            Tuple of (delta, smell_advantages, baseline_qpe_score, current_qpe_score)
        """
        baseline_qpe_score = calculate_qpe(baseline_metrics)
        current_qpe_score = calculate_qpe(current_metrics)

        delta = ComplexityDelta(
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
            qpe_change=current_qpe_score.qpe - baseline_qpe_score.qpe,
        )

        advantages = smell_advantage(baseline_qpe_score, current_qpe_score)

        return delta, advantages, baseline_qpe_score, current_qpe_score

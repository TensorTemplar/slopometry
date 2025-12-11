"""Baseline computation service for staged-impact analysis."""

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.database import EventDatabase
from slopometry.core.git_tracker import GitTracker
from slopometry.core.models import (
    HistoricalMetricStats,
    RepoBaseline,
)


@dataclass
class CommitDelta:
    """Metrics delta between two consecutive commits."""

    cc_delta: float
    effort_delta: float
    mi_delta: float


class BaselineService:
    """Computes and manages repository complexity baselines."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def get_or_compute_baseline(
        self, repo_path: Path, recompute: bool = False, max_workers: int = 4
    ) -> RepoBaseline | None:
        """Get cached baseline or compute if stale (HEAD changed).

        Args:
            repo_path: Path to the repository
            recompute: Force recomputation even if cache is valid
            max_workers: Number of parallel workers for analysis

        Returns:
            RepoBaseline or None if computation fails
        """
        repo_path = repo_path.resolve()
        git_tracker = GitTracker(repo_path)
        head_sha = git_tracker._get_current_commit_sha()

        if not head_sha:
            return None

        if not recompute:
            cached = self.db.get_cached_baseline(str(repo_path), head_sha)
            if cached:
                return cached

        baseline = self.compute_full_baseline(repo_path, max_workers=max_workers)

        if baseline:
            self.db.save_baseline(baseline)

        return baseline

    def compute_full_baseline(self, repo_path: Path, max_workers: int = 4) -> RepoBaseline | None:
        """Compute baseline from entire git history with parallel analysis.

        Args:
            repo_path: Path to the repository
            max_workers: Number of parallel workers for commit analysis

        Returns:
            RepoBaseline or None if computation fails
        """
        repo_path = repo_path.resolve()

        # Get all commits in topological order
        commits = self._get_all_commits(repo_path)
        if len(commits) < 2:
            # Need at least 2 commits to compute deltas
            return None

        head_sha = commits[0]

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        current_metrics = analyzer.analyze_extended_complexity()

        # Create commit pairs for delta computation
        # We compute delta between consecutive commits: (parent, child)
        commit_pairs = [(commits[i + 1], commits[i]) for i in range(len(commits) - 1)]

        deltas = self._compute_deltas_parallel(repo_path, commit_pairs, max_workers)

        if not deltas:
            return None

        cc_deltas = [d.cc_delta for d in deltas]
        effort_deltas = [d.effort_delta for d in deltas]
        mi_deltas = [d.mi_delta for d in deltas]

        return RepoBaseline(
            repository_path=str(repo_path),
            computed_at=datetime.now(),
            head_commit_sha=head_sha,
            total_commits_analyzed=len(deltas),
            cc_delta_stats=self._compute_stats("cc_delta", cc_deltas),
            effort_delta_stats=self._compute_stats("effort_delta", effort_deltas),
            mi_delta_stats=self._compute_stats("mi_delta", mi_deltas),
            current_metrics=current_metrics,
        )

    def _get_all_commits(self, repo_path: Path) -> list[str]:
        """Get all commit SHAs in topological order (newest first)."""
        result = subprocess.run(
            ["git", "rev-list", "--topo-order", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return []

        commits = result.stdout.strip().split("\n")
        valid_commits = [c.strip() for c in commits if c.strip()]
        # Limit to last 100 commits for performance
        return valid_commits[:100]

    def _compute_deltas_parallel(
        self,
        repo_path: Path,
        commit_pairs: list[tuple[str, str]],
        max_workers: int,
    ) -> list[CommitDelta]:
        """Compute deltas for commit pairs in parallel."""
        deltas = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._compute_single_delta, repo_path, parent, child): (parent, child)
                for parent, child in commit_pairs
            }

            for future in as_completed(futures):
                try:
                    delta = future.result(timeout=120)
                    if delta:
                        deltas.append(delta)
                except Exception:
                    pass

        return deltas

    def _compute_single_delta(self, repo_path: Path, parent_sha: str, child_sha: str) -> CommitDelta | None:
        """Compute metrics delta between two commits.

        Args:
            repo_path: Path to the repository
            parent_sha: Parent commit SHA
            child_sha: Child commit SHA

        Returns:
            CommitDelta or None if computation fails
        """
        git_tracker = GitTracker(repo_path)

        parent_dir = git_tracker.extract_files_from_commit(parent_sha)
        child_dir = git_tracker.extract_files_from_commit(child_sha)

        try:
            if not parent_dir or not child_dir:
                return None

            analyzer = ComplexityAnalyzer(working_directory=repo_path)

            parent_metrics = analyzer.analyze_extended_complexity(parent_dir)
            child_metrics = analyzer.analyze_extended_complexity(child_dir)

            return CommitDelta(
                cc_delta=child_metrics.average_complexity - parent_metrics.average_complexity,
                effort_delta=child_metrics.average_effort - parent_metrics.average_effort,
                mi_delta=child_metrics.average_mi - parent_metrics.average_mi,
            )

        finally:
            if parent_dir:
                shutil.rmtree(parent_dir, ignore_errors=True)
            if child_dir:
                shutil.rmtree(child_dir, ignore_errors=True)

    def _compute_stats(self, metric_name: str, values: list[float] | list[int]) -> HistoricalMetricStats:
        """Compute statistical summary for a list of values."""
        if not values:
            return HistoricalMetricStats(
                metric_name=metric_name,
                mean=0.0,
                std_dev=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                sample_count=0,
                trend_coefficient=0.0,
            )

        # Handle case where all values are the same (std_dev would be 0)
        std = stdev(values) if len(values) > 1 else 0.0

        return HistoricalMetricStats(
            metric_name=metric_name,
            mean=mean(values),
            std_dev=std,
            median=median(values),
            min_value=min(values),
            max_value=max(values),
            sample_count=len(values),
            trend_coefficient=self._compute_trend(values),
        )

    def _compute_trend(self, values: list[float] | list[int]) -> float:
        """Compute linear regression slope to detect improvement/degradation trend.

        Positive slope = metrics increasing over time (worse for CC/Effort, better for MI)
        Negative slope = metrics decreasing over time

        Uses simple linear regression: slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        where x = commit index (0, 1, 2, ...) and y = metric value
        """
        if len(values) < 2:
            return 0.0

        n = len(values)
        # values are in chronological order (oldest first in original data,
        # but we received them in reverse order from git rev-list)
        # So index 0 is most recent, index n-1 is oldest
        # We want to compute trend from oldest to newest, so reverse
        chronological_values = list(reversed(values))

        x_mean = (n - 1) / 2.0
        y_mean = mean(chronological_values)

        numerator = sum((i - x_mean) * (chronological_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

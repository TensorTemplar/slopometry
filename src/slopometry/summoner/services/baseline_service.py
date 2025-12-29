"""Baseline computation service for staged-impact analysis."""

import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Commit SHA and timestamp."""

    sha: str
    timestamp: datetime


@dataclass
class CommitDelta:
    """Metrics delta between two consecutive commits."""

    cc_delta: float
    effort_delta: float
    mi_delta: float


def _compute_single_delta_task(repo_path: Path, parent_sha: str, child_sha: str) -> CommitDelta | None:
    """Compute metrics delta between two commits (module-level for ProcessPoolExecutor).

    NOTE: Must be at module level because ProcessPoolExecutor requires picklable callables.
    """
    git_tracker = GitTracker(repo_path)

    changed_files = git_tracker.get_changed_python_files(parent_sha, child_sha)
    if not changed_files:
        return CommitDelta(cc_delta=0.0, effort_delta=0.0, mi_delta=0.0)

    parent_dir = git_tracker.extract_specific_files_from_commit(parent_sha, changed_files)
    child_dir = git_tracker.extract_specific_files_from_commit(child_sha, changed_files)

    try:
        if not parent_dir and not child_dir:
            return None

        analyzer = ComplexityAnalyzer(working_directory=repo_path)

        parent_metrics = analyzer.analyze_extended_complexity(parent_dir) if parent_dir else None
        child_metrics = analyzer.analyze_extended_complexity(child_dir) if child_dir else None

        parent_cc = parent_metrics.total_complexity if parent_metrics else 0
        parent_effort = parent_metrics.total_effort if parent_metrics else 0.0
        parent_mi = parent_metrics.total_mi if parent_metrics else 0.0

        child_cc = child_metrics.total_complexity if child_metrics else 0
        child_effort = child_metrics.total_effort if child_metrics else 0.0
        child_mi = child_metrics.total_mi if child_metrics else 0.0

        return CommitDelta(
            cc_delta=child_cc - parent_cc,
            effort_delta=child_effort - parent_effort,
            mi_delta=child_mi - parent_mi,
        )

    finally:
        if parent_dir:
            shutil.rmtree(parent_dir, ignore_errors=True)
        if child_dir:
            shutil.rmtree(child_dir, ignore_errors=True)


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

        commits = self._get_all_commits(repo_path)
        if len(commits) < 2:
            return None

        head_commit = commits[0]

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        current_metrics = analyzer.analyze_extended_complexity()

        commit_pairs = [(commits[i + 1].sha, commits[i].sha) for i in range(len(commits) - 1)]

        deltas = self._compute_deltas_parallel(repo_path, commit_pairs, max_workers)

        if not deltas:
            return None

        cc_deltas = [d.cc_delta for d in deltas]
        effort_deltas = [d.effort_delta for d in deltas]
        mi_deltas = [d.mi_delta for d in deltas]

        # commits are in reverse chronological order (newest first, oldest last)
        newest_commit_date = commits[0].timestamp
        oldest_commit_date = commits[-1].timestamp

        oldest_commit_tokens = self._get_commit_token_count(repo_path, commits[-1].sha, analyzer)

        return RepoBaseline(
            repository_path=str(repo_path),
            computed_at=datetime.now(),
            head_commit_sha=head_commit.sha,
            total_commits_analyzed=len(deltas),
            cc_delta_stats=self._compute_stats("cc_delta", cc_deltas),
            effort_delta_stats=self._compute_stats("effort_delta", effort_deltas),
            mi_delta_stats=self._compute_stats("mi_delta", mi_deltas),
            current_metrics=current_metrics,
            oldest_commit_date=oldest_commit_date,
            newest_commit_date=newest_commit_date,
            oldest_commit_tokens=oldest_commit_tokens,
        )

    def _get_all_commits(self, repo_path: Path) -> list[CommitInfo]:
        """Get all commits with timestamps in topological order (newest first)."""
        result = subprocess.run(
            ["git", "log", "--format=%H %ct", "--topo-order", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                sha, timestamp_str = parts
                timestamp = datetime.fromtimestamp(int(timestamp_str))
                commits.append(CommitInfo(sha=sha, timestamp=timestamp))

        # PERF: Limit commits to avoid slow analysis on large repos
        return commits[: settings.baseline_max_commits]

    def _compute_deltas_parallel(
        self,
        repo_path: Path,
        commit_pairs: list[tuple[str, str]],
        max_workers: int,
    ) -> list[CommitDelta]:
        """Compute deltas for commit pairs in parallel."""
        deltas = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single_delta_task, repo_path, parent, child): (parent, child)
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

    def _get_commit_token_count(self, repo_path: Path, commit_sha: str, analyzer: ComplexityAnalyzer) -> int | None:
        """Get total token count for a specific commit.

        Args:
            repo_path: Path to the repository
            commit_sha: The commit SHA to analyze
            analyzer: ComplexityAnalyzer instance

        Returns:
            Total token count or None if analysis fails
        """
        git_tracker = GitTracker(repo_path)
        commit_dir = git_tracker.extract_files_from_commit(commit_sha)

        if not commit_dir:
            return None

        try:
            metrics = analyzer.analyze_extended_complexity(commit_dir)
            return metrics.total_tokens
        except Exception:
            return None
        finally:
            if commit_dir:
                shutil.rmtree(commit_dir, ignore_errors=True)

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
        # NOTE: Values arrive in reverse chronological order from git rev-list
        # (index 0 = most recent). Reverse to compute trend oldest-to-newest.
        chronological_values = list(reversed(values))

        x_mean = (n - 1) / 2.0
        y_mean = mean(chronological_values)

        numerator = sum((i - x_mean) * (chronological_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

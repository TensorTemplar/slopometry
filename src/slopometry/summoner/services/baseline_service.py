"""Baseline computation service for staged-impact analysis."""

import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.database import EventDatabase
from slopometry.core.git_tracker import GitOperationError, GitTracker
from slopometry.core.models import (
    BaselineStrategy,
    HistoricalMetricStats,
    RepoBaseline,
    ResolvedBaselineStrategy,
)
from slopometry.core.settings import settings
from slopometry.summoner.services.qpe_calculator import calculate_qpe

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
    qpe_delta: float


def _compute_single_delta_task(repo_path: Path, parent_sha: str, child_sha: str) -> CommitDelta | None:
    """Compute metrics delta between two commits (module-level for ProcessPoolExecutor).

    NOTE: Must be at module level because ProcessPoolExecutor requires picklable callables.
    """
    git_tracker = GitTracker(repo_path)
    parent_dir = None
    child_dir = None

    try:
        changed_files = git_tracker.get_changed_python_files(parent_sha, child_sha)
        if not changed_files:
            return CommitDelta(cc_delta=0.0, effort_delta=0.0, mi_delta=0.0, qpe_delta=0.0)

        parent_dir = git_tracker.extract_specific_files_from_commit(parent_sha, changed_files)
        child_dir = git_tracker.extract_specific_files_from_commit(child_sha, changed_files)

        if not parent_dir and not child_dir:
            return None

        analyzer = ComplexityAnalyzer(working_directory=repo_path)

        parent_metrics = analyzer.analyze_extended_complexity(parent_dir) if parent_dir else None
        child_metrics = analyzer.analyze_extended_complexity(child_dir) if child_dir else None

        parent_cc = parent_metrics.total_complexity if parent_metrics else 0
        parent_effort = parent_metrics.total_effort if parent_metrics else 0.0
        parent_mi = parent_metrics.total_mi if parent_metrics else 0.0
        parent_qpe = calculate_qpe(parent_metrics).qpe if parent_metrics else 0.0

        child_cc = child_metrics.total_complexity if child_metrics else 0
        child_effort = child_metrics.total_effort if child_metrics else 0.0
        child_mi = child_metrics.total_mi if child_metrics else 0.0
        child_qpe = calculate_qpe(child_metrics).qpe if child_metrics else 0.0

        return CommitDelta(
            cc_delta=child_cc - parent_cc,
            effort_delta=child_effort - parent_effort,
            mi_delta=child_mi - parent_mi,
            qpe_delta=child_qpe - parent_qpe,
        )

    except GitOperationError as e:
        logger.warning(f"Git operation failed for {parent_sha}..{child_sha}: {e}")
        return None

    finally:
        if parent_dir:
            shutil.rmtree(parent_dir, ignore_errors=True)
        if child_dir:
            shutil.rmtree(child_dir, ignore_errors=True)


def _parse_commit_log(output: str) -> list[CommitInfo]:
    """Parse git log output in '%H %ct' format into CommitInfo list."""
    commits = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            sha, timestamp_str = parts
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            commits.append(CommitInfo(sha=sha, timestamp=timestamp))
    return commits


class BaselineService:
    """Computes and manages repository complexity baselines."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def get_or_compute_baseline(
        self, repo_path: Path, recompute: bool = False, max_workers: int = 4
    ) -> RepoBaseline | None:
        """Get cached baseline or compute if stale (HEAD changed or strategy mismatch).

        Cache invalidation rules:
        - HEAD changed -> recompute
        - Cached baseline has no qpe_stats -> recompute (legacy)
        - Cached baseline has no strategy -> recompute (pre-strategy baseline)
        - Explicit strategy setting doesn't match cached resolved strategy -> recompute
        - AUTO accepts any concrete resolved strategy in cache
        """
        repo_path = repo_path.resolve()
        git_tracker = GitTracker(repo_path)
        head_sha = git_tracker._get_current_commit_sha()

        if not head_sha:
            return None

        if not recompute:
            cached = self.db.get_cached_baseline(str(repo_path), head_sha)
            if cached and cached.qpe_stats is not None:
                if self._is_cache_strategy_compatible(cached):
                    return cached

        baseline = self.compute_full_baseline(repo_path, max_workers=max_workers)

        if baseline:
            self.db.save_baseline(baseline)

        return baseline

    def _is_cache_strategy_compatible(self, cached: RepoBaseline) -> bool:
        """Check if cached baseline's strategy is compatible with current settings.

        Returns True if cache can be reused, False if recomputation needed.
        """
        requested = BaselineStrategy(settings.baseline_strategy)

        if cached.strategy is None:
            # Legacy baseline without strategy info -> recompute
            return False

        if requested == BaselineStrategy.AUTO:
            # AUTO accepts any concrete resolved strategy
            return True

        # Explicit strategy must match resolved
        return cached.strategy.resolved == requested

    def compute_full_baseline(self, repo_path: Path, max_workers: int = 4) -> RepoBaseline | None:
        """Compute baseline from git history using strategy-based commit selection.

        The strategy determines which commits are selected for delta computation:
        - MERGE_ANCHORED: first-parent trunk history (merge commits as quality checkpoints)
        - TIME_SAMPLED: regular time-interval samples within a bounded lookback window
        - AUTO: detects merge ratio and picks the best strategy

        Delta computation, stats aggregation, and QPE calculation are unchanged --
        only commit selection varies.
        """
        repo_path = repo_path.resolve()

        strategy = self._resolve_strategy(repo_path)

        match strategy.resolved:
            case BaselineStrategy.MERGE_ANCHORED:
                commits = self._get_merge_anchored_commits(repo_path)
            case BaselineStrategy.TIME_SAMPLED:
                commits = self._get_time_sampled_commits(repo_path)
            case _:
                raise ValueError(f"Unexpected resolved strategy: {strategy.resolved}")

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
        qpe_deltas = [d.qpe_delta for d in deltas]

        newest_commit_date = commits[0].timestamp
        oldest_commit_date = commits[-1].timestamp

        oldest_commit_tokens = self._get_commit_token_count(repo_path, commits[-1].sha, analyzer)

        current_qpe = calculate_qpe(current_metrics)

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
            qpe_stats=self._compute_stats("qpe_delta", qpe_deltas),
            current_qpe=current_qpe,
            strategy=strategy,
        )

    def _resolve_strategy(self, repo_path: Path) -> ResolvedBaselineStrategy:
        """Resolve the baseline strategy, auto-detecting if needed."""
        requested = BaselineStrategy(settings.baseline_strategy)

        if requested != BaselineStrategy.AUTO:
            return ResolvedBaselineStrategy(
                requested=requested,
                resolved=requested,
                merge_ratio=0.0,
                total_commits_sampled=0,
            )

        return self._detect_strategy(repo_path)

    def _detect_strategy(self, repo_path: Path) -> ResolvedBaselineStrategy:
        """Auto-detect the best baseline strategy by examining merge commit ratio.

        Uses fast git rev-list --count operations to determine merge frequency.
        If merge_ratio > threshold, use MERGE_ANCHORED. Otherwise TIME_SAMPLED.
        """
        sample_size = settings.baseline_detection_sample_size

        # Count total commits (capped at sample_size)
        total_result = subprocess.run(
            ["git", "rev-list", "--count", f"--max-count={sample_size}", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        total_commits = int(total_result.stdout.strip()) if total_result.returncode == 0 else 0

        if total_commits == 0:
            return ResolvedBaselineStrategy(
                requested=BaselineStrategy.AUTO,
                resolved=BaselineStrategy.TIME_SAMPLED,
                merge_ratio=0.0,
                total_commits_sampled=0,
            )

        # Count merge commits in the same range
        merge_result = subprocess.run(
            ["git", "rev-list", "--merges", "--count", f"--max-count={sample_size}", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        merge_commits = int(merge_result.stdout.strip()) if merge_result.returncode == 0 else 0

        merge_ratio = merge_commits / total_commits if total_commits > 0 else 0.0

        resolved = (
            BaselineStrategy.MERGE_ANCHORED
            if merge_ratio > settings.baseline_merge_ratio_threshold
            else BaselineStrategy.TIME_SAMPLED
        )

        return ResolvedBaselineStrategy(
            requested=BaselineStrategy.AUTO,
            resolved=resolved,
            merge_ratio=merge_ratio,
            total_commits_sampled=total_commits,
        )

    def _get_merge_anchored_commits(self, repo_path: Path) -> list[CommitInfo]:
        """Get commits following first-parent (trunk) history.

        --first-parent follows only the trunk line, naturally landing on merge commits.
        Each delta between consecutive first-parent commits captures the net effect
        of one accepted merge/PR, filtering out intermediate WIP commits on feature branches.
        """
        result = subprocess.run(
            ["git", "log", "--first-parent", "--format=%H %ct", "--topo-order", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return []

        commits = _parse_commit_log(result.stdout)
        return commits[: settings.baseline_max_commits]

    def _get_time_sampled_commits(self, repo_path: Path) -> list[CommitInfo]:
        """Get commits sampled at regular time intervals within a bounded lookback window.

        1. Fetches all commits within the lookback window (--after flag)
        2. Samples one commit per interval (baseline_time_sample_interval_days)
        3. Always includes newest and oldest commits in window
        4. Falls back to evenly-spaced if interval produces too few commits
        """
        lookback_date = datetime.now() - timedelta(days=settings.baseline_lookback_months * 30)
        after_str = lookback_date.strftime("%Y-%m-%d")

        result = subprocess.run(
            ["git", "log", f"--after={after_str}", "--format=%H %ct", "--topo-order", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return []

        all_commits = _parse_commit_log(result.stdout)

        if len(all_commits) < 2:
            return all_commits

        # Sample at regular time intervals
        interval = timedelta(days=settings.baseline_time_sample_interval_days)
        sampled = [all_commits[0]]  # Always include newest
        last_sampled_time = all_commits[0].timestamp

        for commit in all_commits[1:-1]:
            if abs((last_sampled_time - commit.timestamp).total_seconds()) >= interval.total_seconds():
                sampled.append(commit)
                last_sampled_time = commit.timestamp

        # Always include oldest
        if all_commits[-1].sha != sampled[-1].sha:
            sampled.append(all_commits[-1])

        # Fall back to evenly-spaced if we got too few
        min_commits = settings.baseline_time_sample_min_commits
        if len(sampled) < min_commits and len(all_commits) >= min_commits:
            step = max(1, len(all_commits) // min_commits)
            sampled = all_commits[::step]
            # Ensure last commit is included
            if sampled[-1].sha != all_commits[-1].sha:
                sampled.append(all_commits[-1])

        return sampled[: settings.baseline_max_commits]

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
                except Exception as e:
                    logger.debug(f"Skipping failed baseline delta analysis: {e}")

        return deltas

    def _get_commit_token_count(self, repo_path: Path, commit_sha: str, analyzer: ComplexityAnalyzer) -> int | None:
        """Get total token count for a specific commit."""
        git_tracker = GitTracker(repo_path)
        commit_dir = None

        try:
            commit_dir = git_tracker.extract_files_from_commit(commit_sha)

            if not commit_dir:
                return None

            metrics = analyzer.analyze_extended_complexity(commit_dir)
            return metrics.total_tokens
        except GitOperationError as e:
            logger.warning(f"Git operation failed for commit {commit_sha}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Failed to analyze token count for commit {commit_sha}: {e}")
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

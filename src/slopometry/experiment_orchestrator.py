"""Orchestrates parallel experiment runs for code complexity reproduction."""

import os
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from slopometry.cli_calculator import CLICalculator
from slopometry.complexity_analyzer import ComplexityAnalyzer
from slopometry.database import EventDatabase
from slopometry.git_tracker import GitTracker
from slopometry.models import ExperimentProgress, ExperimentRun, ExperimentStatus, ExtendedComplexityMetrics
from slopometry.worktree_manager import WorktreeManager


class ExperimentOrchestrator:
    """Coordinates parallel experiments for complexity tracking."""

    def __init__(self, repo_path: Path, db_path: Path | None = None):
        self.repo_path = repo_path.resolve()
        self.db = EventDatabase(db_path)
        self.worktree_manager = WorktreeManager(self.repo_path)
        self.git_tracker = GitTracker(self.repo_path)
        self.cli_calculator = CLICalculator()

    def run_parallel_experiments(
        self, commit_pairs: list[tuple[str, str]], max_workers: int = 4
    ) -> dict[str, ExperimentRun]:
        """Run multiple experiments in parallel.

        Args:
            commit_pairs: List of (start_commit, target_commit) tuples
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary mapping experiment_id to ExperimentRun
        """
        experiments = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_experiment: dict[Future[None], ExperimentRun] = {}

            # Submit all experiments
            for start_commit, target_commit in commit_pairs:
                experiment = ExperimentRun(
                    repository_path=self.repo_path,
                    start_commit=start_commit,
                    target_commit=target_commit,
                    process_id=os.getpid(),
                )
                experiments[experiment.id] = experiment

                # Save initial experiment state
                self.db.save_experiment_run(experiment)

                future = executor.submit(
                    self._run_single_experiment,
                    experiment.id,
                    start_commit,
                    target_commit,
                )
                future_to_experiment[future] = experiment

            # Monitor progress while experiments run
            while future_to_experiment:
                self.display_aggregate_progress()

                done_futures = []
                for future in future_to_experiment:
                    if future.done():
                        done_futures.append(future)

                for future in done_futures:
                    experiment = future_to_experiment.pop(future)
                    try:
                        future.result()
                        experiment.status = ExperimentStatus.COMPLETED
                        experiment.end_time = datetime.now()
                    except Exception as e:
                        experiment.status = ExperimentStatus.FAILED
                        experiment.end_time = datetime.now()
                        print(f"Experiment {experiment.id} failed: {e}")

                    self.db.update_experiment_run(experiment)

                time.sleep(1)

        return experiments

    def _run_single_experiment(self, experiment_id: str, start_commit: str, target_commit: str) -> None:
        """Run a single experiment in an isolated worktree.

        Args:
            experiment_id: Unique experiment identifier
            start_commit: Starting commit SHA
            target_commit: Target commit SHA
        """
        worktree_path = None
        try:
            worktree_path = self.worktree_manager.create_experiment_worktree(experiment_id, start_commit)

            self.db.update_experiment_worktree(experiment_id, worktree_path)

            target_analyzer = ComplexityAnalyzer(self.repo_path)
            target_metrics = target_analyzer.analyze_extended_complexity()

            progress = ExperimentProgress(
                experiment_id=experiment_id,
                current_metrics=target_metrics,  # Starting from empty/minimal
                target_metrics=target_metrics,
                cli_score=0.0,
            )
            self.db.save_experiment_progress(progress)

            # TODO: This is where the agent would run
            # For now, simulate progress updates
            self._simulate_agent_progress(experiment_id, worktree_path, target_metrics)

        finally:
            # Clean up worktree
            if worktree_path:
                self.worktree_manager.cleanup_worktree(worktree_path)

    def _simulate_agent_progress(
        self, experiment_id: str, worktree_path: Path, target_metrics: ExtendedComplexityMetrics
    ) -> None:
        """Simulate agent making progress (placeholder for actual agent integration).

        Args:
            experiment_id: Experiment identifier
            worktree_path: Path to the experiment worktree
            target_metrics: Target complexity metrics to achieve
        """
        # This is a placeholder - in real implementation, this would:
        # 1. Launch the agent with the worktree as working directory
        # 2. Monitor agent's code changes
        # 3. Periodically analyze complexity and update progress

        analyzer = ComplexityAnalyzer(worktree_path)

        for i in range(10):
            time.sleep(0.5)

            current_metrics = analyzer.analyze_extended_complexity()

            cli_score, component_scores = self.cli_calculator.calculate_cli(current_metrics, target_metrics)

            progress = ExperimentProgress(
                experiment_id=experiment_id,
                current_metrics=current_metrics,
                target_metrics=target_metrics,
                cli_score=cli_score,
                complexity_score=component_scores["complexity"],
                halstead_score=component_scores["halstead"],
                maintainability_score=component_scores["maintainability"],
            )
            self.db.save_experiment_progress(progress)

    def display_aggregate_progress(self) -> None:
        """Display real-time progress across all running experiments."""
        running_experiments = self.db.get_running_experiments()

        if not running_experiments:
            return

        print("\n" + "=" * 80)
        print("EXPERIMENT PROGRESS")
        print("=" * 80)

        for experiment in running_experiments:
            latest_progress = self.db.get_latest_progress(experiment.id)
            if latest_progress:
                print(f"\nExperiment: {experiment.start_commit} → {experiment.target_commit}")
                print(f"  CLI Score: {latest_progress.cli_score:.3f}")
                print(f"  - Complexity: {latest_progress.complexity_score:.3f}")
                print(f"  - Halstead: {latest_progress.halstead_score:.3f}")
                print(f"  - Maintainability: {latest_progress.maintainability_score:.3f}")

        print("=" * 80)

    def analyze_commit_chain(self, base_commit: str, head_commit: str) -> None:
        """Analyze complexity evolution across a chain of commits.

        Args:
            base_commit: Starting commit (e.g., HEAD~10)
            head_commit: Ending commit (e.g., HEAD)
        """
        result = subprocess.run(
            ["git", "rev-list", "--reverse", f"{base_commit}..{head_commit}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        commits = result.stdout.strip().split("\n")
        if not commits:
            print(f"No commits found between {base_commit} and {head_commit}")
            return

        print(f"\nAnalyzing {len(commits)} commits from {base_commit} to {head_commit}")

        chain_id = self.db.create_commit_chain(str(self.repo_path), base_commit, head_commit, len(commits))

        cumulative_complexity = 0
        analyzer = ComplexityAnalyzer(self.repo_path)

        for i, commit_sha in enumerate(commits):
            print(f"Analyzing commit {i + 1}/{len(commits)}: {commit_sha[:8]}")

            temp_dir = self.git_tracker.extract_files_from_commit(commit_sha)
            if not temp_dir:
                continue

            try:
                metrics = analyzer.analyze_extended_complexity(temp_dir)
                incremental_complexity = metrics.total_complexity - cumulative_complexity
                cumulative_complexity = metrics.total_complexity

                self.db.save_complexity_evolution(
                    chain_id=chain_id,
                    commit_sha=commit_sha,
                    commit_order=i,
                    cumulative_complexity=cumulative_complexity,
                    incremental_complexity=incremental_complexity,
                    file_metrics=metrics.model_dump_json(),
                )

            finally:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"\nTotal complexity growth: {cumulative_complexity}")
        print(f"Average complexity per commit: {cumulative_complexity / len(commits):.2f}")

"""Orchestrates parallel experiment runs for code complexity reproduction."""

import os
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.coverage_analyzer import CoverageAnalyzer
from slopometry.core.database import EventDatabase
from slopometry.core.git_tracker import GitTracker
from slopometry.core.models import ExperimentProgress, ExperimentRun, ExperimentStatus, ExtendedComplexityMetrics
from slopometry.summoner.services.cli_calculator import CLICalculator
from slopometry.summoner.services.worktree_manager import WorktreeManager


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

            for start_commit, target_commit in commit_pairs:
                experiment = ExperimentRun(
                    repository_path=self.repo_path,
                    start_commit=start_commit,
                    target_commit=target_commit,
                    process_id=os.getpid(),
                )
                experiments[experiment.id] = experiment

                self.db.save_experiment_run(experiment)

                future = executor.submit(
                    self._run_single_experiment,
                    experiment.id,
                    start_commit,
                    target_commit,
                )
                future_to_experiment[future] = experiment

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

            # NOTE: Agent integration placeholder - currently simulates progress.
            # Future: integrate with Claude Code agent to drive actual code changes.
            self._simulate_agent_progress(experiment_id, worktree_path, target_metrics)

        finally:
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
        console = Console()

        result = subprocess.run(
            ["git", "rev-list", "--reverse", f"{base_commit}..{head_commit}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        commits = result.stdout.strip().split("\n")
        if not commits:
            console.print(f"[yellow]No commits found between {base_commit} and {head_commit}[/yellow]")
            return

        console.print(f"\n[bold]Analyzing {len(commits)} commits from {base_commit} to {head_commit}[/bold]")

        chain_id = self.db.create_commit_chain(str(self.repo_path), base_commit, head_commit, len(commits))

        analyzer = ComplexityAnalyzer(self.repo_path)
        previous_metrics = None
        previous_coverage: float | None = None

        cumulative_cc = 0
        cumulative_volume = 0.0
        cumulative_difficulty = 0.0
        cumulative_effort = 0.0
        cumulative_mi = 0.0
        cumulative_coverage = 0.0
        coverage_data_points = 0  # Track how many commits had coverage data

        for i, commit_sha in enumerate(commits):
            console.print(f"\n[cyan]Analyzing commit {i + 1}/{len(commits)}: {commit_sha[:8]}[/cyan]")

            temp_dir = self.git_tracker.extract_files_from_commit(commit_sha)
            if not temp_dir:
                continue

            try:
                metrics = analyzer.analyze_extended_complexity(temp_dir)

                # Parse coverage if coverage.xml exists in this commit
                coverage_percent: float | None = None
                coverage_xml_path = temp_dir / "coverage.xml"
                if coverage_xml_path.exists():
                    coverage_analyzer = CoverageAnalyzer(temp_dir)
                    coverage_result = coverage_analyzer.analyze_coverage()
                    if coverage_result.coverage_available:
                        coverage_percent = coverage_result.total_coverage_percent

                # Calculate deltas if we have previous metrics
                if previous_metrics:
                    delta_table = Table(title=f"Changes in {commit_sha[:8]}")
                    delta_table.add_column("Metric", style="cyan")
                    delta_table.add_column("Previous", justify="right")
                    delta_table.add_column("Current", justify="right")
                    delta_table.add_column("Change", justify="right")

                    # Cyclomatic Complexity
                    cc_change = metrics.total_complexity - previous_metrics.total_complexity
                    cc_color = "green" if cc_change < 0 else "red" if cc_change > 0 else "yellow"
                    delta_table.add_row(
                        "Cyclomatic Complexity",
                        str(previous_metrics.total_complexity),
                        str(metrics.total_complexity),
                        f"[{cc_color}]{cc_change:+d}[/{cc_color}]",
                    )

                    # Halstead Volume
                    vol_change = metrics.total_volume - previous_metrics.total_volume
                    vol_color = "green" if vol_change < 0 else "red" if vol_change > 0 else "yellow"
                    delta_table.add_row(
                        "Halstead Volume",
                        f"{previous_metrics.total_volume:.1f}",
                        f"{metrics.total_volume:.1f}",
                        f"[{vol_color}]{vol_change:+.1f}[/{vol_color}]",
                    )

                    # Halstead Difficulty
                    diff_change = metrics.total_difficulty - previous_metrics.total_difficulty
                    diff_color = "green" if diff_change < 0 else "red" if diff_change > 0 else "yellow"
                    delta_table.add_row(
                        "Halstead Difficulty",
                        f"{previous_metrics.total_difficulty:.1f}",
                        f"{metrics.total_difficulty:.1f}",
                        f"[{diff_color}]{diff_change:+.1f}[/{diff_color}]",
                    )

                    # Halstead Effort
                    effort_change = metrics.total_effort - previous_metrics.total_effort
                    effort_color = "green" if effort_change < 0 else "red" if effort_change > 0 else "yellow"
                    delta_table.add_row(
                        "Halstead Effort",
                        f"{previous_metrics.total_effort:.1f}",
                        f"{metrics.total_effort:.1f}",
                        f"[{effort_color}]{effort_change:+.1f}[/{effort_color}]",
                    )

                    # Maintainability Index (higher is better)
                    mi_change = metrics.average_mi - previous_metrics.average_mi
                    mi_color = "red" if mi_change < 0 else "green" if mi_change > 0 else "yellow"
                    delta_table.add_row(
                        "Avg Maintainability Index",
                        f"{previous_metrics.average_mi:.1f}",
                        f"{metrics.average_mi:.1f}",
                        f"[{mi_color}]{mi_change:+.1f}[/{mi_color}]",
                    )

                    # Files
                    files_change = metrics.total_files_analyzed - previous_metrics.total_files_analyzed
                    files_color = "green" if files_change < 0 else "red" if files_change > 0 else "yellow"
                    delta_table.add_row(
                        "Files Analyzed",
                        str(previous_metrics.total_files_analyzed),
                        str(metrics.total_files_analyzed),
                        f"[{files_color}]{files_change:+d}[/{files_color}]",
                    )

                    # Test Coverage (higher is better)
                    if coverage_percent is not None or previous_coverage is not None:
                        prev_cov_str = f"{previous_coverage:.1f}%" if previous_coverage is not None else "N/A"
                        curr_cov_str = f"{coverage_percent:.1f}%" if coverage_percent is not None else "N/A"
                        if coverage_percent is not None and previous_coverage is not None:
                            cov_change = coverage_percent - previous_coverage
                            cov_color = "green" if cov_change > 0 else "red" if cov_change < 0 else "yellow"
                            cov_change_str = f"[{cov_color}]{cov_change:+.1f}%[/{cov_color}]"
                        else:
                            cov_change_str = "[dim]N/A[/dim]"
                        delta_table.add_row("Test Coverage", prev_cov_str, curr_cov_str, cov_change_str)

                    console.print(delta_table)

                    cumulative_cc += cc_change
                    cumulative_volume += vol_change
                    cumulative_difficulty += diff_change
                    cumulative_effort += effort_change
                    cumulative_mi += mi_change
                    if coverage_percent is not None and previous_coverage is not None:
                        cumulative_coverage += coverage_percent - previous_coverage
                        coverage_data_points += 1
                else:
                    # First commit - show initial state
                    initial_table = Table(title=f"Initial State at {commit_sha[:8]}")
                    initial_table.add_column("Metric", style="cyan")
                    initial_table.add_column("Value", justify="right")

                    initial_table.add_row("Cyclomatic Complexity", str(metrics.total_complexity))
                    initial_table.add_row("Halstead Volume", f"{metrics.total_volume:.1f}")
                    initial_table.add_row("Halstead Difficulty", f"{metrics.total_difficulty:.1f}")
                    initial_table.add_row("Halstead Effort", f"{metrics.total_effort:.1f}")
                    initial_table.add_row("Avg Maintainability Index", f"{metrics.average_mi:.1f}")
                    initial_table.add_row("Files Analyzed", str(metrics.total_files_analyzed))
                    if coverage_percent is not None:
                        initial_table.add_row("Test Coverage", f"{coverage_percent:.1f}%")

                    console.print(initial_table)

                self.db.save_complexity_evolution(
                    chain_id=chain_id,
                    commit_sha=commit_sha,
                    commit_order=i,
                    cumulative_complexity=metrics.total_complexity,
                    incremental_complexity=metrics.total_complexity
                    - (previous_metrics.total_complexity if previous_metrics else 0),
                    file_metrics=metrics.model_dump_json(),
                    test_coverage_percent=coverage_percent,
                )

                previous_metrics = metrics
                previous_coverage = coverage_percent

            finally:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        # Show cumulative summary
        if len(commits) > 1:
            console.print("\n[bold]Cumulative Changes Summary[/bold]")
            summary_table = Table()
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Total Change", justify="right")
            summary_table.add_column("Average per Commit", justify="right")

            cc_color = "green" if cumulative_cc < 0 else "red" if cumulative_cc > 0 else "yellow"
            summary_table.add_row(
                "Cyclomatic Complexity",
                f"[{cc_color}]{cumulative_cc:+d}[/{cc_color}]",
                f"{cumulative_cc / len(commits):.1f}",
            )

            vol_color = "green" if cumulative_volume < 0 else "red" if cumulative_volume > 0 else "yellow"
            summary_table.add_row(
                "Halstead Volume",
                f"[{vol_color}]{cumulative_volume:+.1f}[/{vol_color}]",
                f"{cumulative_volume / len(commits):.1f}",
            )

            diff_color = "green" if cumulative_difficulty < 0 else "red" if cumulative_difficulty > 0 else "yellow"
            summary_table.add_row(
                "Halstead Difficulty",
                f"[{diff_color}]{cumulative_difficulty:+.1f}[/{diff_color}]",
                f"{cumulative_difficulty / len(commits):.1f}",
            )

            effort_color = "green" if cumulative_effort < 0 else "red" if cumulative_effort > 0 else "yellow"
            summary_table.add_row(
                "Halstead Effort",
                f"[{effort_color}]{cumulative_effort:+.1f}[/{effort_color}]",
                f"{cumulative_effort / len(commits):.1f}",
            )

            mi_color = "red" if cumulative_mi < 0 else "green" if cumulative_mi > 0 else "yellow"
            summary_table.add_row(
                "Avg Maintainability Index",
                f"[{mi_color}]{cumulative_mi:+.1f}[/{mi_color}]",
                f"{cumulative_mi / len(commits):.1f}",
            )

            # Test Coverage (only show if we had coverage data)
            if coverage_data_points > 0:
                cov_color = "green" if cumulative_coverage > 0 else "red" if cumulative_coverage < 0 else "yellow"
                summary_table.add_row(
                    "Test Coverage",
                    f"[{cov_color}]{cumulative_coverage:+.1f}%[/{cov_color}]",
                    f"{cumulative_coverage / coverage_data_points:.1f}%",
                )

            console.print(summary_table)

"""Experiment orchestration service for summoner features."""

from datetime import datetime
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models.display import ExperimentDisplayData
from slopometry.core.models.experiment import ProgressDisplayData


class ExperimentService:
    """Handles experiment orchestration and tracking for summoner users."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def run_parallel_experiments(self, repo_path: Path, commits: int, max_workers: int) -> dict:
        """Run parallel experiments across git commits."""
        from slopometry.summoner.services.experiment_orchestrator import ExperimentOrchestrator

        orchestrator = ExperimentOrchestrator(repo_path)

        commit_pairs = []
        for i in range(commits, 0, -1):
            start_commit = f"HEAD~{i}"
            target_commit = f"HEAD~{i - 1}" if i > 1 else "HEAD"
            commit_pairs.append((start_commit, target_commit))

        return orchestrator.run_parallel_experiments(commit_pairs, max_workers)

    def analyze_commit_chain(self, repo_path: Path, base_commit: str, head_commit: str) -> None:
        """Analyze complexity evolution across a chain of commits."""
        from slopometry.summoner.services.experiment_orchestrator import ExperimentOrchestrator

        orchestrator = ExperimentOrchestrator(repo_path)
        orchestrator.analyze_commit_chain(base_commit, head_commit)

    def list_experiments(self) -> list[ExperimentDisplayData]:
        """List all experiment runs with metadata."""
        try:
            with self.db._get_db_connection() as conn:
                rows = conn.execute("""
                    SELECT id, repository_path, start_commit, target_commit,
                           start_time, end_time, status
                    FROM experiment_runs
                    ORDER BY start_time DESC
                """).fetchall()

            experiments_data = []
            for row in rows:
                experiment_id, repo_path, start_commit, target_commit, start_time, end_time, status = row

                start_dt = datetime.fromisoformat(start_time)
                if end_time:
                    end_dt = datetime.fromisoformat(end_time)
                    duration = str(end_dt - start_dt)
                else:
                    duration = "Running..."

                experiments_data.append(
                    ExperimentDisplayData(
                        id=experiment_id,
                        repository_name=Path(repo_path).name,
                        commits_display=f"{start_commit} → {target_commit}",
                        start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        duration=duration,
                        status=status,
                    )
                )

            return experiments_data
        except Exception:
            return []

    def get_experiment_details(self, experiment_id: str) -> tuple | None:
        """Get detailed information about a specific experiment."""
        try:
            with self.db._get_db_connection() as conn:
                experiment_row = conn.execute(
                    "SELECT * FROM experiment_runs WHERE id LIKE ?", (f"{experiment_id}%",)
                ).fetchone()

                if not experiment_row:
                    return None

                progress_rows = conn.execute(
                    """
                    SELECT timestamp, cli_score, complexity_score, halstead_score, maintainability_score
                    FROM experiment_progress
                    WHERE experiment_id = ?
                    ORDER BY timestamp
                """,
                    (experiment_row[0],),
                ).fetchall()

                return experiment_row, progress_rows
        except Exception:
            return None

    def prepare_progress_data_for_display(self, progress_rows: list) -> list[ProgressDisplayData]:
        """Prepare experiment progress data for display formatting."""
        progress_data = []
        for row in progress_rows:
            timestamp, cli_score, complexity_score, halstead_score, maintainability_score = row
            dt = datetime.fromisoformat(timestamp)
            progress_data.append(
                ProgressDisplayData(
                    timestamp=dt.strftime("%H:%M:%S"),
                    cli_score=f"{cli_score:.3f}",
                    complexity_score=f"{complexity_score:.3f}",
                    halstead_score=f"{halstead_score:.3f}",
                    maintainability_score=f"{maintainability_score:.3f}",
                )
            )
        return progress_data

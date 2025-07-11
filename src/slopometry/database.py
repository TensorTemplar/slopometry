"""Database operations for storing hook events."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from slopometry.models import (
    ComplexityDelta,
    ExperimentProgress,
    ExperimentRun,
    ExperimentStatus,
    ExtendedComplexityMetrics,
    GitState,
    HookEvent,
    HookEventType,
    NextFeaturePrediction,
    PlanEvolution,
    Project,
    ProjectSource,
    SessionStatistics,
    ToolType,
    UserStory,
)
from slopometry.plan_analyzer import PlanAnalyzer
from slopometry.settings import settings


class EventDatabase:
    """Manages SQLite database for hook event storage."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = settings.resolved_database_path
        else:
            db_path = Path(db_path)

        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._plan_analyzers: dict[str, PlanAnalyzer] = {}
        self._create_tables()

    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Create database tables."""
        with self._get_db_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS hook_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    tool_name TEXT,
                    tool_type TEXT,
                    metadata TEXT DEFAULT '{}',
                    duration_ms INTEGER,
                    exit_code INTEGER,
                    error_message TEXT,
                    git_state TEXT,
                    working_directory TEXT,
                    project_name TEXT,
                    project_source TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hook_events_session_id 
                ON hook_events(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hook_events_timestamp 
                ON hook_events(timestamp)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id TEXT PRIMARY KEY,
                    repository_path TEXT NOT NULL,
                    start_commit TEXT NOT NULL,
                    target_commit TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    worktree_path TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS commit_complexity_snapshots (
                    commit_sha TEXT PRIMARY KEY,
                    repository_path TEXT NOT NULL,
                    commit_message TEXT,
                    timestamp TEXT NOT NULL,
                    parent_commit_sha TEXT,
                    complexity_metrics TEXT NOT NULL,
                    complexity_delta TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_metrics TEXT NOT NULL,
                    target_metrics TEXT NOT NULL,
                    cli_score REAL NOT NULL,
                    complexity_score REAL,
                    halstead_score REAL,
                    maintainability_score REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiment_runs(id)
                )
            """)

            # Store pre-computed complexity deltas for commit ranges
            conn.execute("""
                CREATE TABLE IF NOT EXISTS commit_complexity_chains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository_path TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    head_commit TEXT NOT NULL,
                    commit_count INTEGER NOT NULL,
                    total_complexity_growth INTEGER,
                    computed_at TEXT NOT NULL,
                    UNIQUE(repository_path, base_commit, head_commit)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS complexity_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_id INTEGER NOT NULL,
                    commit_sha TEXT NOT NULL,
                    commit_order INTEGER NOT NULL,
                    cumulative_complexity INTEGER NOT NULL,
                    incremental_complexity INTEGER NOT NULL,
                    file_metrics TEXT NOT NULL,
                    FOREIGN KEY (chain_id) REFERENCES commit_complexity_chains(id),
                    UNIQUE(chain_id, commit_sha)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS nfp_objectives (
                    id TEXT PRIMARY KEY,
                    target_commit TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_stories (
                    id TEXT PRIMARY KEY,
                    nfp_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    acceptance_criteria TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    estimated_complexity INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    FOREIGN KEY (nfp_id) REFERENCES nfp_objectives(id)
                )
            """)

            # Update experiment_runs to include NFP reference
            try:
                conn.execute("""
                    ALTER TABLE experiment_runs 
                    ADD COLUMN nfp_objective_id TEXT 
                    REFERENCES nfp_objectives(id)
                """)
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_runs_status ON experiment_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_progress_cli ON experiment_progress(experiment_id, cli_score)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_complexity_evolution_commit ON complexity_evolution(commit_sha)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chains_repo_commits ON commit_complexity_chains(repository_path, base_commit, head_commit)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nfp_commits ON nfp_objectives(repository_path, base_commit, target_commit)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_stories_nfp ON user_stories(nfp_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_stories_priority ON user_stories(priority)")

            conn.commit()

    def save_event(self, event: HookEvent) -> int:
        """Save a hook event to the database."""
        self._update_plan_evolution(event)

        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO hook_events (
                    session_id, event_type, timestamp, sequence_number,
                    tool_name, tool_type, metadata, duration_ms,
                    exit_code, error_message, git_state, working_directory,
                    project_name, project_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.session_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.sequence_number,
                    event.tool_name,
                    event.tool_type.value if event.tool_type else None,
                    json.dumps(event.metadata),
                    event.duration_ms,
                    event.exit_code,
                    event.error_message,
                    event.git_state.model_dump_json() if event.git_state else None,
                    event.working_directory,
                    event.project.name if event.project else None,
                    event.project.source.value if event.project else None,
                ),
            )
            return cursor.lastrowid or 0

    def _update_plan_evolution(self, event: HookEvent) -> None:
        """Update plan evolution tracking for a session."""
        session_id = event.session_id
        if session_id not in self._plan_analyzers:
            self._plan_analyzers[session_id] = PlanAnalyzer()
        analyzer = self._plan_analyzers[session_id]
        if event.tool_name == "TodoWrite" and event.event_type == HookEventType.POST_TOOL_USE:
            tool_input = event.metadata.get("tool_input", {})
            if tool_input:
                analyzer.analyze_todo_write_event(tool_input, event.timestamp)
        elif event.event_type == HookEventType.POST_TOOL_USE:
            analyzer.increment_event_count(event.tool_type)

    def get_session_events(self, session_id: str) -> list[HookEvent]:
        """Get all events for a session."""
        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hook_events WHERE session_id = ? ORDER BY sequence_number",
                (session_id,),
            ).fetchall()

            events = []
            for row in rows:
                row_keys = row.keys()
                git_state = None
                if "git_state" in row_keys and row["git_state"]:
                    try:
                        git_state_data = json.loads(row["git_state"])
                        git_state = GitState.model_validate(git_state_data)
                    except (json.JSONDecodeError, ValueError):
                        git_state = None

                working_directory = (
                    row["working_directory"]
                    if "working_directory" in row_keys and row["working_directory"]
                    else "Unknown"
                )

                project = None
                if "project_name" in row_keys and row["project_name"]:
                    project = Project(
                        name=row["project_name"],
                        source=ProjectSource(row["project_source"]),
                    )

                events.append(
                    HookEvent(
                        id=row["id"],
                        session_id=row["session_id"],
                        event_type=HookEventType(row["event_type"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        sequence_number=row["sequence_number"],
                        tool_name=row["tool_name"],
                        tool_type=ToolType(row["tool_type"]) if row["tool_type"] else None,
                        metadata=json.loads(row["metadata"]),
                        duration_ms=row["duration_ms"],
                        exit_code=row["exit_code"],
                        error_message=row["error_message"],
                        git_state=git_state,
                        working_directory=working_directory,
                        project=project,
                    )
                )
            return events

    def get_session_statistics(self, session_id: str) -> SessionStatistics | None:
        """Calculate statistics for a session."""
        events = self.get_session_events(session_id)
        if not events:
            return None

        working_directory = "Unknown"
        project = None
        for event in events:
            if event.working_directory and event.working_directory != "Unknown":
                working_directory = event.working_directory
            if event.project:
                project = event.project
            if working_directory != "Unknown" and project:
                break

        stats = SessionStatistics(
            session_id=session_id,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp if events else None,
            total_events=len(events),
            working_directory=working_directory,
            project=project,
        )

        for event in events:
            stats.events_by_type[event.event_type] = stats.events_by_type.get(event.event_type, 0) + 1
            if event.tool_type:
                stats.tool_usage[event.tool_type] = stats.tool_usage.get(event.tool_type, 0) + 1
            if event.error_message:
                stats.error_count += 1
            if event.duration_ms:
                stats.total_duration_ms += event.duration_ms

        tool_events_with_duration = [e for e in events if e.duration_ms]
        if tool_events_with_duration:
            stats.average_tool_duration_ms = stats.total_duration_ms / len(tool_events_with_duration)

        git_events = [e for e in events if e.git_state]
        if git_events:
            stats.initial_git_state = git_events[0].git_state
            stats.final_git_state = git_events[-1].git_state
            if stats.initial_git_state and stats.final_git_state:
                from slopometry.git_tracker import GitTracker

                git_tracker = GitTracker()
                stats.commits_made = git_tracker.calculate_commits_made(stats.initial_git_state, stats.final_git_state)

        stats.complexity_metrics, stats.complexity_delta = self._get_session_complexity_metrics(
            session_id, stats.working_directory
        )

        try:
            stats.plan_evolution = self._calculate_plan_evolution(events)
        except Exception:
            stats.plan_evolution = None

        return stats

    def _get_session_complexity_metrics(
        self, session_id: str, working_directory: str | None
    ) -> tuple[ExtendedComplexityMetrics | None, ComplexityDelta | None]:
        """Retrieve extended complexity metrics for a session."""
        if not working_directory:
            return None, None

        try:
            return self.calculate_extended_complexity_metrics(working_directory)
        except Exception:
            return None, None

    def _calculate_plan_evolution(self, events: list[HookEvent]) -> PlanEvolution:
        """Calculate plan evolution from session events."""
        analyzer = PlanAnalyzer()
        for event in events:
            if event.tool_name == "TodoWrite" and event.event_type == HookEventType.POST_TOOL_USE:
                tool_input = event.metadata.get("tool_input", {})
                if tool_input:
                    analyzer.analyze_todo_write_event(tool_input, event.timestamp)
            elif event.event_type == HookEventType.POST_TOOL_USE:
                analyzer.increment_event_count(event.tool_type)
        return analyzer.get_plan_evolution()

    def calculate_extended_complexity_metrics(
        self, working_directory: str
    ) -> tuple[ExtendedComplexityMetrics | None, ComplexityDelta | None]:
        """Calculate extended complexity metrics including Halstead and MI.

        This provides CC, Halstead metrics, and Maintainability Index.
        Calculates delta between HEAD and HEAD~1 commits for accuracy.
        """
        try:
            import shutil

            from slopometry.complexity_analyzer import ComplexityAnalyzer
            from slopometry.git_tracker import GitTracker

            git_tracker = GitTracker(Path(working_directory))
            analyzer = ComplexityAnalyzer(working_directory=Path(working_directory))

            complexity_delta = None
            current_basic = None

            current_extended = analyzer.analyze_extended_complexity()

            # For accurate session tracking: working directory vs HEAD
            if git_tracker.has_previous_commit():
                # Extract HEAD (where session started)
                baseline_dir = git_tracker.extract_files_from_commit("HEAD")

                if baseline_dir:
                    try:
                        baseline_extended = analyzer.analyze_extended_complexity(baseline_dir)  # HEAD

                        current_basic = analyzer.analyze_complexity()  # Working directory
                        baseline_basic = analyzer._analyze_directory(baseline_dir)  # HEAD
                        complexity_delta = analyzer._calculate_delta(baseline_basic, current_basic)

                        complexity_delta.total_volume_change = (
                            current_extended.total_volume - baseline_extended.total_volume
                        )
                        complexity_delta.avg_volume_change = (
                            current_extended.average_volume - baseline_extended.average_volume
                        )
                        complexity_delta.total_difficulty_change = (
                            current_extended.total_difficulty - baseline_extended.total_difficulty
                        )
                        complexity_delta.avg_difficulty_change = (
                            current_extended.average_difficulty - baseline_extended.average_difficulty
                        )
                        complexity_delta.total_effort_change = (
                            current_extended.total_effort - baseline_extended.total_effort
                        )
                        complexity_delta.total_mi_change = current_extended.total_mi - baseline_extended.total_mi
                        complexity_delta.avg_mi_change = current_extended.average_mi - baseline_extended.average_mi

                        # Ensure consistency: use same CC values that were used in delta calculation
                        current_extended.total_complexity = current_basic.total_complexity
                        current_extended.average_complexity = current_basic.average_complexity
                        current_extended.max_complexity = current_basic.max_complexity
                        current_extended.min_complexity = current_basic.min_complexity
                        current_extended.total_files_analyzed = current_basic.total_files_analyzed
                        current_extended.files_by_complexity = current_basic.files_by_complexity

                        shutil.rmtree(baseline_dir, ignore_errors=True)
                    except Exception:
                        if baseline_dir:
                            shutil.rmtree(baseline_dir, ignore_errors=True)
                        pass

            return current_extended, complexity_delta

        except Exception:
            return None, None

    def list_sessions(self) -> list[str]:
        """List all unique session IDs."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                "SELECT session_id, MIN(timestamp) as first_event FROM hook_events GROUP BY session_id ORDER BY first_event DESC"
            ).fetchall()
            return [row[0] for row in rows]

    def cleanup_old_data(self, days: int, dry_run: bool = False) -> tuple[int, int]:
        """Clean up old session data and associated files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        with self._get_db_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM hook_events WHERE timestamp < ?",
                (cutoff_date.isoformat(),),
            ).fetchall()
            session_ids_to_delete = [row[0] for row in rows]
            if not dry_run and session_ids_to_delete:
                conn.execute("DELETE FROM hook_events WHERE timestamp < ?", (cutoff_date.isoformat(),))

        files_deleted = 0
        state_dir = Path.home() / ".claude" / "slopometry"
        if state_dir.exists():
            for session_id in session_ids_to_delete:
                seq_file = state_dir / f"seq_{session_id}.txt"
                if seq_file.exists():
                    if not dry_run:
                        seq_file.unlink()
                    files_deleted += 1
        return len(session_ids_to_delete), files_deleted

    def cleanup_session(self, session_id: str) -> tuple[int, int]:
        """Clean up a specific session and its associated files."""
        with self._get_db_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM hook_events WHERE session_id = ?", (session_id,)).fetchone()
            events_count = result[0] if result else 0
            conn.execute("DELETE FROM hook_events WHERE session_id = ?", (session_id,))

        files_deleted = 0
        state_dir = Path.home() / ".claude" / "slopometry"
        seq_file = state_dir / f"seq_{session_id}.txt"
        if seq_file.exists():
            seq_file.unlink()
            files_deleted = 1
        return events_count, files_deleted

    def cleanup_all_sessions(self) -> tuple[int, int, int]:
        """Clean up all sessions and associated files."""
        sessions = self.list_sessions()
        with self._get_db_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM hook_events").fetchone()
            events_count = result[0] if result else 0
            conn.execute("DELETE FROM hook_events")

        files_deleted = 0
        state_dir = Path.home() / ".claude" / "slopometry"
        if state_dir.exists():
            for seq_file in state_dir.glob("seq_*.txt"):
                seq_file.unlink()
                files_deleted += 1
        return len(sessions), events_count, files_deleted

    # Experiment tracking methods

    def save_experiment_run(self, experiment: ExperimentRun) -> None:
        """Save an experiment run to the database."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_runs (
                    id, repository_path, start_commit, target_commit,
                    process_id, worktree_path, start_time, end_time, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.id,
                    str(experiment.repository_path),
                    experiment.start_commit,
                    experiment.target_commit,
                    experiment.process_id,
                    str(experiment.worktree_path) if experiment.worktree_path else None,
                    experiment.start_time.isoformat(),
                    experiment.end_time.isoformat() if experiment.end_time else None,
                    experiment.status.value,
                ),
            )
            conn.commit()

    def update_experiment_run(self, experiment: ExperimentRun) -> None:
        """Update an existing experiment run."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                UPDATE experiment_runs 
                SET status = ?, end_time = ?, worktree_path = ?
                WHERE id = ?
            """,
                (
                    experiment.status.value,
                    experiment.end_time.isoformat() if experiment.end_time else None,
                    str(experiment.worktree_path) if experiment.worktree_path else None,
                    experiment.id,
                ),
            )
            conn.commit()

    def update_experiment_worktree(self, experiment_id: str, worktree_path: Path) -> None:
        """Update experiment worktree path."""
        with self._get_db_connection() as conn:
            conn.execute(
                "UPDATE experiment_runs SET worktree_path = ? WHERE id = ?",
                (str(worktree_path), experiment_id),
            )
            conn.commit()

    def save_experiment_progress(self, progress: ExperimentProgress) -> None:
        """Save experiment progress to the database."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiment_progress (
                    experiment_id, timestamp, current_metrics, target_metrics,
                    cli_score, complexity_score, halstead_score, maintainability_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    progress.experiment_id,
                    progress.timestamp.isoformat(),
                    progress.current_metrics.model_dump_json(),
                    progress.target_metrics.model_dump_json(),
                    progress.cli_score,
                    progress.complexity_score,
                    progress.halstead_score,
                    progress.maintainability_score,
                ),
            )
            conn.commit()

    def get_running_experiments(self) -> list[ExperimentRun]:
        """Get all currently running experiments."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_runs WHERE status = ?", (ExperimentStatus.RUNNING.value,)
            ).fetchall()

            experiments = []
            for row in rows:
                experiment = ExperimentRun(
                    id=row[0],
                    repository_path=Path(row[1]),
                    start_commit=row[2],
                    target_commit=row[3],
                    process_id=row[4],
                    worktree_path=Path(row[5]) if row[5] else None,
                    start_time=datetime.fromisoformat(row[6]),
                    end_time=datetime.fromisoformat(row[7]) if row[7] else None,
                    status=ExperimentStatus(row[8]),
                )
                experiments.append(experiment)

            return experiments

    def get_latest_progress(self, experiment_id: str) -> ExperimentProgress | None:
        """Get the latest progress for an experiment."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM experiment_progress 
                WHERE experiment_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """,
                (experiment_id,),
            ).fetchone()

            if not row:
                return None

            from slopometry.models import ExtendedComplexityMetrics

            return ExperimentProgress(
                experiment_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                current_metrics=ExtendedComplexityMetrics.model_validate_json(row[3]),
                target_metrics=ExtendedComplexityMetrics.model_validate_json(row[4]),
                cli_score=row[5],
                complexity_score=row[6],
                halstead_score=row[7],
                maintainability_score=row[8],
            )

    def create_commit_chain(self, repository_path: str, base_commit: str, head_commit: str, commit_count: int) -> int:
        """Create a commit complexity chain record."""
        with self._get_db_connection() as conn:
            # Check if chain already exists
            existing = conn.execute(
                "SELECT id FROM commit_complexity_chains WHERE repository_path = ? AND base_commit = ? AND head_commit = ?",
                (repository_path, base_commit, head_commit),
            ).fetchone()

            if existing:
                # Update existing record
                conn.execute(
                    """
                    UPDATE commit_complexity_chains 
                    SET commit_count = ?, computed_at = ?
                    WHERE id = ?
                """,
                    (commit_count, datetime.now().isoformat(), existing[0]),
                )
                conn.commit()
                return int(existing[0])
            else:
                # Create new record
                cursor = conn.execute(
                    """
                    INSERT INTO commit_complexity_chains (
                        repository_path, base_commit, head_commit, commit_count, computed_at
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (repository_path, base_commit, head_commit, commit_count, datetime.now().isoformat()),
                )
                conn.commit()
                return cursor.lastrowid or 0

    def save_complexity_evolution(
        self,
        chain_id: int,
        commit_sha: str,
        commit_order: int,
        cumulative_complexity: int,
        incremental_complexity: int,
        file_metrics: str,
    ) -> None:
        """Save complexity evolution data for a commit."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO complexity_evolution (
                    chain_id, commit_sha, commit_order, cumulative_complexity,
                    incremental_complexity, file_metrics
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (chain_id, commit_sha, commit_order, cumulative_complexity, incremental_complexity, file_metrics),
            )
            conn.commit()

    # NFP (Next Feature Prediction) management methods

    def save_nfp_objective(self, nfp: NextFeaturePrediction) -> None:
        """Save an NFP objective with its user stories."""
        with self._get_db_connection() as conn:
            # Save NFP objective
            conn.execute(
                """
                INSERT OR REPLACE INTO nfp_objectives (
                    id, target_commit, base_commit, repository_path,
                    title, description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    nfp.id,
                    nfp.target_commit,
                    nfp.base_commit,
                    str(nfp.repository_path),
                    nfp.title,
                    nfp.description,
                    nfp.created_at.isoformat(),
                    nfp.updated_at.isoformat(),
                ),
            )

            # Delete existing user stories for this NFP
            conn.execute("DELETE FROM user_stories WHERE nfp_id = ?", (nfp.id,))

            # Save user stories
            for story in nfp.user_stories:
                conn.execute(
                    """
                    INSERT INTO user_stories (
                        id, nfp_id, title, description, acceptance_criteria,
                        priority, estimated_complexity, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        story.id,
                        nfp.id,
                        story.title,
                        story.description,
                        json.dumps(story.acceptance_criteria),
                        story.priority,
                        story.estimated_complexity,
                        json.dumps(story.tags),
                    ),
                )

            conn.commit()

    def get_nfp_objective(self, nfp_id: str) -> NextFeaturePrediction | None:
        """Get an NFP objective by ID."""
        with self._get_db_connection() as conn:
            # Get NFP objective
            nfp_row = conn.execute("SELECT * FROM nfp_objectives WHERE id = ?", (nfp_id,)).fetchone()

            if not nfp_row:
                return None

            # Get user stories
            story_rows = conn.execute(
                "SELECT * FROM user_stories WHERE nfp_id = ? ORDER BY priority, title",
                (nfp_id,),
            ).fetchall()

            user_stories = []
            for story_row in story_rows:
                story = UserStory(
                    id=story_row[0],
                    title=story_row[2],
                    description=story_row[3],
                    acceptance_criteria=json.loads(story_row[4]),
                    priority=story_row[5],
                    estimated_complexity=story_row[6],
                    tags=json.loads(story_row[7]),
                )
                user_stories.append(story)

            nfp = NextFeaturePrediction(
                id=nfp_row[0],
                target_commit=nfp_row[1],
                base_commit=nfp_row[2],
                repository_path=Path(nfp_row[3]),
                title=nfp_row[4],
                description=nfp_row[5],
                created_at=datetime.fromisoformat(nfp_row[6]),
                updated_at=datetime.fromisoformat(nfp_row[7]),
                user_stories=user_stories,
            )

            return nfp

    def get_nfp_by_commits(
        self, repository_path: str, base_commit: str, target_commit: str
    ) -> NextFeaturePrediction | None:
        """Get NFP objective by commit range."""
        with self._get_db_connection() as conn:
            nfp_row = conn.execute(
                """
                SELECT id FROM nfp_objectives 
                WHERE repository_path = ? AND base_commit = ? AND target_commit = ?
            """,
                (repository_path, base_commit, target_commit),
            ).fetchone()

            if nfp_row:
                return self.get_nfp_objective(nfp_row[0])
            return None

    def list_nfp_objectives(self, repository_path: str | None = None) -> list[NextFeaturePrediction]:
        """List all NFP objectives, optionally filtered by repository."""
        with self._get_db_connection() as conn:
            if repository_path:
                rows = conn.execute(
                    """
                    SELECT id FROM nfp_objectives 
                    WHERE repository_path = ?
                    ORDER BY created_at DESC
                """,
                    (repository_path,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT id FROM nfp_objectives ORDER BY created_at DESC").fetchall()

            objectives = []
            for row in rows:
                nfp = self.get_nfp_objective(row[0])
                if nfp:
                    objectives.append(nfp)

            return objectives

    def delete_nfp_objective(self, nfp_id: str) -> bool:
        """Delete an NFP objective and its user stories."""
        with self._get_db_connection() as conn:
            # Delete user stories first
            conn.execute("DELETE FROM user_stories WHERE nfp_id = ?", (nfp_id,))

            # Delete NFP objective
            cursor = conn.execute("DELETE FROM nfp_objectives WHERE id = ?", (nfp_id,))
            conn.commit()

            return bool(cursor.rowcount and cursor.rowcount > 0)


class SessionManager:
    """Manages sequence numbering for Claude Code sessions."""

    def __init__(self):
        self.state_dir = Path.home() / ".claude" / "slopometry"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def get_next_sequence_number(self, session_id: str) -> int:
        """Get the next sequence number for a session."""
        seq_file = self.state_dir / f"seq_{session_id}.txt"
        if seq_file.exists():
            try:
                current_seq = int(seq_file.read_text().strip())
                next_seq = current_seq + 1
            except (ValueError, FileNotFoundError):
                next_seq = 1
        else:
            next_seq = 1
        seq_file.write_text(str(next_seq))
        return next_seq

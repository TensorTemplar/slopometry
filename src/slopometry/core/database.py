"""Database operations for storing hook events."""

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from slopometry.core.migrations import MigrationRunner

logger = logging.getLogger(__name__)
from slopometry.core.models import (
    ComplexityDelta,
    ContextCoverage,
    ExperimentProgress,
    ExperimentRun,
    ExperimentStatus,
    ExtendedComplexityMetrics,
    FeatureBoundary,
    GitState,
    HistoricalMetricStats,
    HookEvent,
    HookEventType,
    LeaderboardEntry,
    NextFeaturePrediction,
    PlanEvolution,
    Project,
    ProjectSource,
    RepoBaseline,
    SessionStatistics,
    ToolType,
    UserStory,
    UserStoryEntry,
)
from slopometry.core.plan_analyzer import PlanAnalyzer
from slopometry.core.settings import settings


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
        self._run_migrations()

    @contextmanager
    def _get_db_connection(self) -> Generator[sqlite3.Connection]:
        """Context manager that ensures database connections are properly closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _run_migrations(self) -> None:
        """Run any pending database migrations."""
        migration_runner = MigrationRunner(self.db_path)
        applied_migrations = migration_runner.run_migrations()

        if applied_migrations and settings.debug_mode:
            import sys

            print(f"Slopometry applied migrations: {applied_migrations}", file=sys.stderr)

    def _create_tables(self) -> None:
        """Create database tables."""
        with self._get_db_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys = ON")

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
                    project_source TEXT,
                    transcript_path TEXT
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
                CREATE INDEX IF NOT EXISTS idx_hook_events_session_timestamp 
                ON hook_events(session_id, timestamp)
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
                    test_coverage_percent REAL,
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

            conn.execute("""
                CREATE TABLE IF NOT EXISTS diff_user_story_dataset (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    head_commit TEXT NOT NULL,
                    diff_content TEXT NOT NULL,
                    user_stories TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    guidelines_for_improving TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    repository_path TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_boundaries (
                    id TEXT PRIMARY KEY,
                    base_commit TEXT NOT NULL,
                    head_commit TEXT NOT NULL,
                    merge_commit TEXT NOT NULL,
                    merge_message TEXT NOT NULL,
                    feature_message TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            try:
                conn.execute("""
                    ALTER TABLE experiment_runs
                    ADD COLUMN nfp_objective_id TEXT
                    REFERENCES nfp_objectives(id)
                """)
            except sqlite3.OperationalError:
                pass

            try:
                conn.execute("""
                    ALTER TABLE complexity_evolution
                    ADD COLUMN test_coverage_percent REAL
                """)
            except sqlite3.OperationalError:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset_commits ON diff_user_story_dataset(base_commit, head_commit)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_rating ON diff_user_story_dataset(rating)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_boundaries_repo ON feature_boundaries(repository_path)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_boundaries_commits ON feature_boundaries(base_commit, head_commit)"
            )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_quality_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    repository_path TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    calculated_at TEXT NOT NULL,
                    complexity_metrics_json TEXT NOT NULL,
                    complexity_delta_json TEXT,
                    working_tree_hash TEXT,
                    calculator_version TEXT,
                    UNIQUE(session_id, repository_path, commit_sha, working_tree_hash, calculator_version)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_quality_cache_repo_commit ON code_quality_cache(repository_path, commit_sha)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_code_quality_cache_session ON code_quality_cache(session_id)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS repo_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository_path TEXT NOT NULL,
                    head_commit_sha TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    total_commits_analyzed INTEGER NOT NULL,

                    cc_delta_mean REAL NOT NULL,
                    cc_delta_std REAL NOT NULL,
                    cc_delta_median REAL NOT NULL,
                    cc_delta_min REAL NOT NULL,
                    cc_delta_max REAL NOT NULL,
                    cc_delta_trend REAL NOT NULL,

                    effort_delta_mean REAL NOT NULL,
                    effort_delta_std REAL NOT NULL,
                    effort_delta_median REAL NOT NULL,
                    effort_delta_min REAL NOT NULL,
                    effort_delta_max REAL NOT NULL,
                    effort_delta_trend REAL NOT NULL,

                    mi_delta_mean REAL NOT NULL,
                    mi_delta_std REAL NOT NULL,
                    mi_delta_median REAL NOT NULL,
                    mi_delta_min REAL NOT NULL,
                    mi_delta_max REAL NOT NULL,
                    mi_delta_trend REAL NOT NULL,

                    current_metrics_json TEXT NOT NULL,

                    UNIQUE(repository_path, head_commit_sha)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_repo_baselines_repo_head ON repo_baselines(repository_path, head_commit_sha)"
            )

            conn.commit()

    def save_event(self, event: HookEvent) -> int:
        """Save a hook event to the database."""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO hook_events (
                    session_id, event_type, timestamp, sequence_number,
                    tool_name, tool_type, metadata, duration_ms,
                    exit_code, error_message, git_state, working_directory,
                    project_name, project_source, transcript_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    event.transcript_path,
                ),
            )
            return cursor.lastrowid or 0

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
                        transcript_path=row["transcript_path"],
                    )
                )
            return events

    def get_session_basic_info(self, session_id: str) -> tuple[datetime, int] | None:
        """Get minimal session info (start_time, total_events) without expensive computations.

        Use this for operations that only need to verify a session exists and show basic info,
        like cleanup confirmations.

        Returns:
            Tuple of (start_time, total_events) or None if session not found.
        """
        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT MIN(timestamp) as start_time, COUNT(*) as total_events
                FROM hook_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if not row or row["total_events"] == 0:
                return None

            return datetime.fromisoformat(row["start_time"]), row["total_events"]

    def get_session_statistics(self, session_id: str) -> SessionStatistics | None:
        """Calculate statistics for a session using optimized SQL aggregations.

        This method avoids loading all events into memory by using SQL aggregations
        and only loading specific events when needed (e.g., for plan evolution).
        """
        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row

            stats_row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_events,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    SUM(CASE WHEN error_message IS NOT NULL AND error_message != '' THEN 1 ELSE 0 END) as error_count,
                    COALESCE(SUM(duration_ms), 0) as total_duration_ms,
                    COUNT(CASE WHEN duration_ms IS NOT NULL THEN 1 ELSE 0 END) as events_with_duration
                FROM hook_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if not stats_row or stats_row["total_events"] == 0:
                return None

            first_event_row = conn.execute(
                """
                SELECT working_directory, project_name, project_source, transcript_path
                FROM hook_events
                WHERE session_id = ?
                ORDER BY sequence_number ASC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            working_directory = first_event_row["working_directory"] or "Unknown"
            project = None
            if first_event_row["project_name"]:
                project = Project(
                    name=first_event_row["project_name"],
                    source=ProjectSource(first_event_row["project_source"]),
                )
            transcript_path = first_event_row["transcript_path"]

            event_type_rows = conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM hook_events
                WHERE session_id = ?
                GROUP BY event_type
                """,
                (session_id,),
            ).fetchall()

            events_by_type = {HookEventType(row["event_type"]): row["count"] for row in event_type_rows}

            tool_usage_rows = conn.execute(
                """
                SELECT tool_type, COUNT(*) as count
                FROM hook_events
                WHERE session_id = ? AND tool_type IS NOT NULL
                GROUP BY tool_type
                """,
                (session_id,),
            ).fetchall()

            tool_usage = {ToolType(row["tool_type"]): row["count"] for row in tool_usage_rows}

            first_git_row = conn.execute(
                """
                SELECT git_state
                FROM hook_events
                WHERE session_id = ? AND git_state IS NOT NULL
                ORDER BY sequence_number ASC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            last_git_row = conn.execute(
                """
                SELECT git_state
                FROM hook_events
                WHERE session_id = ? AND git_state IS NOT NULL
                ORDER BY sequence_number DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            initial_git_state = None
            final_git_state = None

            if first_git_row and first_git_row["git_state"]:
                git_state_data = json.loads(first_git_row["git_state"])
                initial_git_state = GitState.model_validate(git_state_data)

            if last_git_row and last_git_row["git_state"]:
                git_state_data = json.loads(last_git_row["git_state"])
                final_git_state = GitState.model_validate(git_state_data)

            commits_made = 0
            if initial_git_state and final_git_state:
                from slopometry.core.git_tracker import GitTracker

                git_tracker = GitTracker()
                commits_made = git_tracker.calculate_commits_made(initial_git_state, final_git_state)

            stats = SessionStatistics(
                session_id=session_id,
                start_time=datetime.fromisoformat(stats_row["start_time"]),
                end_time=datetime.fromisoformat(stats_row["end_time"]) if stats_row["end_time"] else None,
                total_events=stats_row["total_events"],
                working_directory=working_directory,
                project=project,
                transcript_path=transcript_path,
                events_by_type=events_by_type,
                tool_usage=tool_usage,
                error_count=stats_row["error_count"],
                total_duration_ms=stats_row["total_duration_ms"],
                initial_git_state=initial_git_state,
                final_git_state=final_git_state,
                commits_made=commits_made,
            )

            if stats_row["events_with_duration"] > 0:
                stats.average_tool_duration_ms = stats_row["total_duration_ms"] / stats_row["events_with_duration"]

        stats.complexity_metrics, stats.complexity_delta = self._get_session_complexity_metrics(
            session_id, stats.working_directory, stats.initial_git_state
        )

        try:
            stats.plan_evolution = self._calculate_plan_evolution(session_id)
            if stats.plan_evolution and stats.transcript_path:
                try:
                    from slopometry.core.transcript_token_analyzer import analyze_transcript_tokens

                    transcript_path = Path(stats.transcript_path)
                    if transcript_path.exists():
                        stats.plan_evolution.token_usage = analyze_transcript_tokens(transcript_path)
                except Exception as e:
                    logger.debug(f"Failed to analyze transcript tokens for session {session_id}: {e}")
        except Exception as e:
            logger.debug(f"Failed to calculate plan evolution for session {session_id}: {e}")
            stats.plan_evolution = None

        if stats.transcript_path:
            try:
                from slopometry.core.compact_analyzer import analyze_transcript_compacts

                transcript_path = Path(stats.transcript_path)
                if transcript_path.exists():
                    stats.compact_events = analyze_transcript_compacts(transcript_path)
            except Exception as e:
                logger.debug(f"Failed to analyze compact events for session {session_id}: {e}")

        try:
            stats.context_coverage = self._calculate_context_coverage(stats.transcript_path, stats.working_directory)
        except Exception as e:
            logger.debug(f"Failed to calculate context coverage for session {session_id}: {e}")
            stats.context_coverage = None

        return stats

    def _get_session_complexity_metrics(
        self, session_id: str, working_directory: str | None, initial_git_state: GitState | None
    ) -> tuple[ExtendedComplexityMetrics | None, ComplexityDelta | None]:
        """Retrieve extended complexity metrics for a session with intelligent caching."""
        if not working_directory:
            return None, None

        try:
            from slopometry.core.code_quality_cache import CodeQualityCacheManager
            from slopometry.core.working_tree_state import WorkingTreeStateCalculator

            wt_calculator = WorkingTreeStateCalculator(working_directory)
            repository_path = str(Path(working_directory).resolve())

            commit_sha = wt_calculator.get_current_commit_sha()
            has_uncommitted_changes = wt_calculator.has_uncommitted_changes()

            baseline_commit_sha = initial_git_state.commit_sha if initial_git_state else None

            if commit_sha is None:
                return self.calculate_extended_complexity_metrics(working_directory, baseline_commit_sha)

            working_tree_hash = None
            if has_uncommitted_changes:
                working_tree_hash = wt_calculator.calculate_working_tree_hash(commit_sha)

            with self._get_db_connection() as conn:
                cache_manager = CodeQualityCacheManager(conn)

                cached_metrics, cached_delta = cache_manager.get_cached_metrics(
                    session_id, repository_path, commit_sha, working_tree_hash
                )

                if cached_metrics is not None:
                    return cached_metrics, cached_delta

                fresh_metrics, fresh_delta = self.calculate_extended_complexity_metrics(
                    working_directory, baseline_commit_sha
                )

                if fresh_metrics is not None:
                    cache_manager.save_metrics_to_cache(
                        session_id, repository_path, commit_sha, fresh_metrics, fresh_delta, working_tree_hash
                    )

                return fresh_metrics, fresh_delta

        except Exception:
            try:
                baseline_commit_sha = initial_git_state.commit_sha if initial_git_state else None
                return self.calculate_extended_complexity_metrics(working_directory, baseline_commit_sha)
            except Exception as e2:
                logger.debug(f"Failed to compute session complexity metrics (fallback also failed): {e2}")
                return None, None

    def _calculate_plan_evolution(self, session_id: str) -> PlanEvolution:
        """Calculate plan evolution using optimized SQL queries.

        Only loads POST_TOOL_USE events needed for plan analysis, avoiding
        loading all events into memory.
        """
        analyzer = PlanAnalyzer()

        with self._get_db_connection() as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT timestamp, tool_name, tool_type, metadata
                FROM hook_events
                WHERE session_id = ? AND event_type = ?
                ORDER BY sequence_number
                """,
                (session_id, HookEventType.POST_TOOL_USE.value),
            ).fetchall()

            for row in rows:
                timestamp = datetime.fromisoformat(row["timestamp"])
                tool_name = row["tool_name"]
                tool_type = ToolType(row["tool_type"]) if row["tool_type"] else None

                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                tool_input = metadata.get("tool_input", {})

                if tool_name == "TodoWrite":
                    if tool_input:
                        analyzer.analyze_todo_write_event(tool_input, timestamp)
                elif tool_name == "Write":
                    # Track Write events for plan files (in addition to counting as implementation)
                    if tool_input:
                        analyzer.analyze_write_event(tool_input)
                    analyzer.increment_event_count(tool_type, tool_input)
                else:
                    analyzer.increment_event_count(tool_type, tool_input)

        return analyzer.get_plan_evolution()

    def _calculate_context_coverage(
        self, transcript_path: str | None, working_directory: str | None
    ) -> ContextCoverage | None:
        """Calculate context coverage from session transcript.

        Args:
            transcript_path: Path to the Claude Code transcript JSONL
            working_directory: Working directory for the session

        Returns:
            ContextCoverage with metrics, or None if unavailable
        """
        if not transcript_path or not working_directory:
            return None

        transcript_file = Path(transcript_path)
        if not transcript_file.exists():
            return None

        working_dir = Path(working_directory)
        if not working_dir.exists():
            return None

        from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

        analyzer = ContextCoverageAnalyzer(working_dir)
        return analyzer.analyze_transcript(transcript_file)

    def calculate_extended_complexity_metrics(
        self, working_directory: str, baseline_commit_sha: str | None = None
    ) -> tuple[ExtendedComplexityMetrics | None, ComplexityDelta | None]:
        """Calculate extended complexity metrics including Halstead and MI.

        This provides CC, Halstead metrics, and Maintainability Index.
        Calculates delta between current state and the baseline commit SHA.

        Args:
            working_directory: Directory to analyze
            baseline_commit_sha: Commit SHA to use as baseline. If None, uses fallback logic:
                - Try merge-base with main/master
                - Fall back to HEAD

        Returns:
            Tuple of (current_metrics, complexity_delta)
        """
        try:
            from slopometry.core.complexity_analyzer import ComplexityAnalyzer
            from slopometry.core.git_tracker import GitTracker

            git_tracker = GitTracker(Path(working_directory))
            analyzer = ComplexityAnalyzer(working_directory=Path(working_directory))

            complexity_delta = None
            current_basic = None

            current_extended = analyzer.analyze_extended_complexity()

            if git_tracker.has_previous_commit():
                if baseline_commit_sha:
                    baseline_ref = baseline_commit_sha
                else:
                    baseline_ref = git_tracker.get_merge_base_with_main()
                    if baseline_ref is None:
                        baseline_ref = "HEAD"

                with git_tracker.extract_files_from_commit_ctx(baseline_ref) as baseline_dir:
                    if baseline_dir:
                        baseline_extended = analyzer.analyze_extended_complexity(baseline_dir)

                        current_basic = analyzer.analyze_complexity()
                        baseline_basic = analyzer._analyze_directory(baseline_dir)
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

                        current_extended.total_complexity = current_basic.total_complexity
                        current_extended.average_complexity = current_basic.average_complexity
                        current_extended.max_complexity = current_basic.max_complexity
                        current_extended.min_complexity = current_basic.min_complexity
                        current_extended.total_files_analyzed = current_basic.total_files_analyzed
                        current_extended.files_by_complexity = current_basic.files_by_complexity

                        complexity_delta.orphan_comment_change = (
                            current_extended.orphan_comment_count - baseline_extended.orphan_comment_count
                        )
                        complexity_delta.untracked_todo_change = (
                            current_extended.untracked_todo_count - baseline_extended.untracked_todo_count
                        )
                        complexity_delta.inline_import_change = (
                            current_extended.inline_import_count - baseline_extended.inline_import_count
                        )
                        complexity_delta.dict_get_with_default_change = (
                            current_extended.dict_get_with_default_count - baseline_extended.dict_get_with_default_count
                        )
                        complexity_delta.hasattr_getattr_change = (
                            current_extended.hasattr_getattr_count - baseline_extended.hasattr_getattr_count
                        )
                        complexity_delta.nonempty_init_change = (
                            current_extended.nonempty_init_count - baseline_extended.nonempty_init_count
                        )

            return current_extended, complexity_delta

        except Exception as e:
            logger.debug(f"Failed to compute extended complexity metrics: {e}")
            return None, None

    def list_sessions(self, limit: int | None = None) -> list[str]:
        """List unique session IDs, optionally limited to recent sessions."""
        with self._get_db_connection() as conn:
            query = "SELECT session_id, MIN(timestamp) as first_event FROM hook_events GROUP BY session_id ORDER BY first_event DESC"
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()
            return [row[0] for row in rows]

    def list_sessions_by_repository(self, repository_path: Path, limit: int | None = None) -> list[str]:
        """List session IDs filtered by repository working directory.

        Sessions are identified by their first event's working_directory.

        Args:
            repository_path: The repository path to filter by
            limit: Optional limit on number of sessions to return

        Returns:
            List of session IDs that started in this repository, ordered by most recent first
        """
        with self._get_db_connection() as conn:
            normalized_path = str(repository_path.resolve())

            query = """
                SELECT session_id, MIN(timestamp) as first_event
                FROM hook_events
                WHERE working_directory = ?
                GROUP BY session_id
                ORDER BY first_event DESC
            """
            params: list = [normalized_path]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [row[0] for row in rows]

    def get_sessions_summary(self, limit: int | None = None) -> list[dict]:
        """Get lightweight session summaries for list display."""
        with self._get_db_connection() as conn:
            query = """
                SELECT 
                    session_id,
                    MIN(timestamp) as start_time,
                    COUNT(*) as total_events,
                    COUNT(DISTINCT tool_type) as tools_used,
                    project_name,
                    project_source
                FROM hook_events 
                WHERE session_id IS NOT NULL
                GROUP BY session_id, project_name, project_source
                ORDER BY MIN(timestamp) DESC
            """
            if limit:
                query += f" LIMIT {limit}"

            rows = conn.execute(query).fetchall()

            summaries = []
            for row in rows:
                summaries.append(
                    {
                        "session_id": row[0],
                        "start_time": row[1],
                        "total_events": row[2],
                        "tools_used": row[3] if row[3] is not None else 0,
                        "project_name": row[4],
                        "project_source": row[5],
                    }
                )

            return summaries

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
                    cli_score, complexity_score, halstead_score, maintainability_score,
                    qpe_score, smell_penalty
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    progress.qpe_score,
                    progress.smell_penalty,
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

            return ExperimentProgress(
                experiment_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                current_metrics=ExtendedComplexityMetrics.model_validate_json(row[3]),
                target_metrics=ExtendedComplexityMetrics.model_validate_json(row[4]),
                cli_score=row[5],
                complexity_score=row[6],
                halstead_score=row[7],
                maintainability_score=row[8],
                qpe_score=row[9] if len(row) > 9 else None,
                smell_penalty=row[10] if len(row) > 10 else None,
            )

    def create_commit_chain(self, repository_path: str, base_commit: str, head_commit: str, commit_count: int) -> int:
        """Create a commit complexity chain record."""
        with self._get_db_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM commit_complexity_chains WHERE repository_path = ? AND base_commit = ? AND head_commit = ?",
                (repository_path, base_commit, head_commit),
            ).fetchone()

            if existing:
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
        test_coverage_percent: float | None = None,
    ) -> None:
        """Save complexity evolution data for a commit."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO complexity_evolution (
                    chain_id, commit_sha, commit_order, cumulative_complexity,
                    incremental_complexity, file_metrics, test_coverage_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    chain_id,
                    commit_sha,
                    commit_order,
                    cumulative_complexity,
                    incremental_complexity,
                    file_metrics,
                    test_coverage_percent,
                ),
            )
            conn.commit()

    def save_nfp_objective(self, nfp: NextFeaturePrediction) -> None:
        """Save an NFP objective with its user stories."""
        with self._get_db_connection() as conn:
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

            conn.execute("DELETE FROM user_stories WHERE nfp_id = ?", (nfp.id,))

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
            nfp_row = conn.execute("SELECT * FROM nfp_objectives WHERE id = ?", (nfp_id,)).fetchone()

            if not nfp_row:
                return None

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
            conn.execute("DELETE FROM user_stories WHERE nfp_id = ?", (nfp_id,))
            cursor = conn.execute("DELETE FROM nfp_objectives WHERE id = ?", (nfp_id,))
            conn.commit()

            return bool(cursor.rowcount and cursor.rowcount > 0)

    def save_user_story_entry(self, entry: UserStoryEntry) -> None:
        """Save a user story entry."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO diff_user_story_dataset 
                (id, created_at, base_commit, head_commit, diff_content, user_stories,
                 rating, guidelines_for_improving, model_used, prompt_template, repository_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.created_at.isoformat(),
                    entry.base_commit,
                    entry.head_commit,
                    entry.diff_content,
                    entry.user_stories,
                    entry.rating,
                    entry.guidelines_for_improving,
                    entry.model_used,
                    entry.prompt_template,
                    entry.repository_path,
                ),
            )

    def get_user_story_entries(self, limit: int = 100) -> list[UserStoryEntry]:
        """Get user story entries ordered by creation date."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, base_commit, head_commit, diff_content, user_stories,
                       rating, guidelines_for_improving, model_used, prompt_template, repository_path
                FROM diff_user_story_dataset
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                UserStoryEntry(
                    id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    base_commit=row[2],
                    head_commit=row[3],
                    diff_content=row[4],
                    user_stories=row[5],
                    rating=row[6],
                    guidelines_for_improving=row[7],
                    model_used=row[8],
                    prompt_template=row[9],
                    repository_path=row[10],
                )
                for row in rows
            ]

    def get_user_story_stats(self) -> dict:
        """Get statistics about the user story entries."""
        with self._get_db_connection() as conn:
            stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(rating) as avg_rating,
                    COUNT(DISTINCT model_used) as unique_models,
                    COUNT(DISTINCT repository_path) as unique_repos
                FROM diff_user_story_dataset
                """
            ).fetchone()

            rating_distribution = conn.execute(
                """
                SELECT rating, COUNT(*) as count
                FROM diff_user_story_dataset
                GROUP BY rating
                ORDER BY rating
                """
            ).fetchall()

            return {
                "total_entries": stats[0] or 0,
                "avg_rating": round(stats[1] or 0, 2),
                "unique_models": stats[2] or 0,
                "unique_repos": stats[3] or 0,
                "rating_distribution": {str(row[0]): row[1] for row in rating_distribution},
            }

    def export_user_stories(self, output_path: Path) -> int:
        """Export user story entries to Parquet format.

        Args:
            output_path: Path for the output parquet file

        Returns:
            Number of records exported
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for export. Install with: pip install pandas pyarrow")

        entries = self.get_user_story_entries(limit=10000)  # Get all entries

        if not entries:
            return 0

        data = []
        for entry in entries:
            data.append(
                {
                    "id": entry.id,
                    "created_at": entry.created_at,
                    "base_commit": entry.base_commit,
                    "head_commit": entry.head_commit,
                    "diff_content": entry.diff_content,
                    "user_stories": entry.user_stories,
                    "rating": entry.rating,
                    "guidelines_for_improving": entry.guidelines_for_improving,
                    "model_used": entry.model_used,
                    "prompt_template": entry.prompt_template,
                    "repository_path": entry.repository_path,
                }
            )

        df = pd.DataFrame(data)
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")

        return len(data)

    def get_user_story_entry_by_id(self, entry_id: str) -> UserStoryEntry | None:
        """Get a specific user story entry by ID."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, base_commit, head_commit, diff_content, user_stories,
                       rating, guidelines_for_improving, model_used, prompt_template, repository_path
                FROM diff_user_story_dataset
                WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()

            if row:
                return UserStoryEntry(
                    id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    base_commit=row[2],
                    head_commit=row[3],
                    diff_content=row[4],
                    user_stories=row[5],
                    rating=row[6],
                    guidelines_for_improving=row[7],
                    model_used=row[8],
                    prompt_template=row[9],
                    repository_path=row[10],
                )
            return None

    def resolve_user_story_entry_id(self, short_id: str) -> str | None:
        """Resolve a short user story entry ID (first 8 chars) to the full ID."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM diff_user_story_dataset
                WHERE id LIKE ?
                """,
                (f"{short_id}%",),
            ).fetchall()

            if len(rows) == 1:
                return rows[0][0]
            elif len(rows) > 1:
                return None
            else:
                return None

    def get_user_story_entry_ids_for_completion(self) -> list[str]:
        """Get all user story entry IDs for tab completion."""
        with self._get_db_connection() as conn:
            rows = conn.execute("SELECT id FROM diff_user_story_dataset").fetchall()
            return [row[0] for row in rows]

    def save_feature_boundaries(self, features: list[FeatureBoundary], repository_path: Path) -> None:
        """Save feature boundaries to the database."""
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM feature_boundaries WHERE repository_path = ?", (repository_path.as_posix(),))

            for feature in features:
                conn.execute(
                    """
                    INSERT INTO feature_boundaries (
                        id, base_commit, head_commit, merge_commit, merge_message,
                        feature_message, repository_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feature.id,
                        feature.base_commit,
                        feature.head_commit,
                        feature.merge_commit,
                        feature.merge_message,
                        feature.feature_message,
                        repository_path.as_posix(),
                        datetime.now().isoformat(),
                    ),
                )
            conn.commit()

    def get_feature_boundaries(self, repository_path: Path) -> list[FeatureBoundary]:
        """Get all feature boundaries for a repository."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, base_commit, head_commit, merge_commit, merge_message, feature_message
                FROM feature_boundaries
                WHERE repository_path = ?
                ORDER BY created_at DESC
                """,
                (repository_path.as_posix(),),
            ).fetchall()

            return [
                FeatureBoundary(
                    id=row[0],
                    base_commit=row[1],
                    head_commit=row[2],
                    merge_commit=row[3],
                    merge_message=row[4],
                    feature_message=row[5],
                    repository_path=repository_path,
                )
                for row in rows
            ]

    def get_feature_boundary_by_id(self, feature_id: str, repository_path: Path) -> FeatureBoundary | None:
        """Get a specific feature boundary by ID."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT id, base_commit, head_commit, merge_commit, merge_message, feature_message
                FROM feature_boundaries
                WHERE id = ? AND repository_path = ?
                """,
                (feature_id, repository_path.as_posix()),
            ).fetchone()

            if row:
                return FeatureBoundary(
                    id=row[0],
                    base_commit=row[1],
                    head_commit=row[2],
                    merge_commit=row[3],
                    merge_message=row[4],
                    feature_message=row[5],
                    repository_path=repository_path,
                )
            return None

    def resolve_feature_id(self, short_id: str, repository_path: Path) -> str | None:
        """Resolve a short feature ID (first 8 chars) to the full ID."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM feature_boundaries
                WHERE id LIKE ? AND repository_path = ?
                """,
                (f"{short_id}%", repository_path.as_posix()),
            ).fetchall()

            if len(rows) == 1:
                return rows[0][0]
            elif len(rows) > 1:
                return None
            else:
                return None

    def get_feature_ids_for_completion(self, repository_path: Path) -> list[str]:
        """Get all feature IDs for tab completion."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM feature_boundaries WHERE repository_path = ?",
                (repository_path.as_posix(),),
            ).fetchall()

            return [row[0] for row in rows]

    def get_best_user_story_entry_for_feature(self, feature: FeatureBoundary) -> str | None:
        """Get the best user story entry ID for a feature based on commit matching.

        Returns the entry with highest rating, or if tied, the newest by created_at.
        """
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, rating, created_at
                FROM diff_user_story_dataset
                WHERE base_commit = ? AND head_commit = ?
                ORDER BY rating DESC, created_at DESC
                LIMIT 1
                """,
                (feature.base_commit, feature.head_commit),
            ).fetchall()

            if rows:
                return rows[0][0]
            return None

    def get_cached_baseline(self, repository_path: str, head_commit_sha: str) -> RepoBaseline | None:
        """Retrieve cached baseline if HEAD matches."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT repository_path, head_commit_sha, computed_at, total_commits_analyzed,
                       cc_delta_mean, cc_delta_std, cc_delta_median, cc_delta_min, cc_delta_max, cc_delta_trend,
                       effort_delta_mean, effort_delta_std, effort_delta_median, effort_delta_min, effort_delta_max, effort_delta_trend,
                       mi_delta_mean, mi_delta_std, mi_delta_median, mi_delta_min, mi_delta_max, mi_delta_trend,
                       current_metrics_json,
                       oldest_commit_date, newest_commit_date, oldest_commit_tokens
                FROM repo_baselines
                WHERE repository_path = ? AND head_commit_sha = ?
                """,
                (repository_path, head_commit_sha),
            ).fetchone()

            if not row:
                return None

            return RepoBaseline(
                repository_path=row[0],
                head_commit_sha=row[1],
                computed_at=datetime.fromisoformat(row[2]),
                total_commits_analyzed=row[3],
                cc_delta_stats=HistoricalMetricStats(
                    metric_name="cc_delta",
                    mean=row[4],
                    std_dev=row[5],
                    median=row[6],
                    min_value=row[7],
                    max_value=row[8],
                    sample_count=row[3],
                    trend_coefficient=row[9],
                ),
                effort_delta_stats=HistoricalMetricStats(
                    metric_name="effort_delta",
                    mean=row[10],
                    std_dev=row[11],
                    median=row[12],
                    min_value=row[13],
                    max_value=row[14],
                    sample_count=row[3],
                    trend_coefficient=row[15],
                ),
                mi_delta_stats=HistoricalMetricStats(
                    metric_name="mi_delta",
                    mean=row[16],
                    std_dev=row[17],
                    median=row[18],
                    min_value=row[19],
                    max_value=row[20],
                    sample_count=row[3],
                    trend_coefficient=row[21],
                ),
                current_metrics=ExtendedComplexityMetrics.model_validate_json(row[22]),
                oldest_commit_date=datetime.fromisoformat(row[23]) if row[23] else None,
                newest_commit_date=datetime.fromisoformat(row[24]) if row[24] else None,
                oldest_commit_tokens=row[25],
            )

    def save_baseline(self, baseline: RepoBaseline) -> None:
        """Save computed baseline to cache."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO repo_baselines (
                    repository_path, head_commit_sha, computed_at, total_commits_analyzed,
                    cc_delta_mean, cc_delta_std, cc_delta_median, cc_delta_min, cc_delta_max, cc_delta_trend,
                    effort_delta_mean, effort_delta_std, effort_delta_median, effort_delta_min, effort_delta_max, effort_delta_trend,
                    mi_delta_mean, mi_delta_std, mi_delta_median, mi_delta_min, mi_delta_max, mi_delta_trend,
                    current_metrics_json,
                    oldest_commit_date, newest_commit_date, oldest_commit_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline.repository_path,
                    baseline.head_commit_sha,
                    baseline.computed_at.isoformat(),
                    baseline.total_commits_analyzed,
                    baseline.cc_delta_stats.mean,
                    baseline.cc_delta_stats.std_dev,
                    baseline.cc_delta_stats.median,
                    baseline.cc_delta_stats.min_value,
                    baseline.cc_delta_stats.max_value,
                    baseline.cc_delta_stats.trend_coefficient,
                    baseline.effort_delta_stats.mean,
                    baseline.effort_delta_stats.std_dev,
                    baseline.effort_delta_stats.median,
                    baseline.effort_delta_stats.min_value,
                    baseline.effort_delta_stats.max_value,
                    baseline.effort_delta_stats.trend_coefficient,
                    baseline.mi_delta_stats.mean,
                    baseline.mi_delta_stats.std_dev,
                    baseline.mi_delta_stats.median,
                    baseline.mi_delta_stats.min_value,
                    baseline.mi_delta_stats.max_value,
                    baseline.mi_delta_stats.trend_coefficient,
                    baseline.current_metrics.model_dump_json(),
                    baseline.oldest_commit_date.isoformat() if baseline.oldest_commit_date else None,
                    baseline.newest_commit_date.isoformat() if baseline.newest_commit_date else None,
                    baseline.oldest_commit_tokens,
                ),
            )
            conn.commit()

    def save_leaderboard_entry(self, entry: LeaderboardEntry) -> None:
        """Save or update a leaderboard entry.

        Uses UPSERT semantics - if an entry for this project_path exists,
        it will be updated with the new values (including new commit info).
        """
        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO qpe_leaderboard (
                    project_name, project_path, commit_sha_short, commit_sha_full,
                    measured_at, qpe_score, mi_normalized, smell_penalty,
                    adjusted_quality, effort_factor, total_effort, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_path) DO UPDATE SET
                    project_name = excluded.project_name,
                    commit_sha_short = excluded.commit_sha_short,
                    commit_sha_full = excluded.commit_sha_full,
                    measured_at = excluded.measured_at,
                    qpe_score = excluded.qpe_score,
                    mi_normalized = excluded.mi_normalized,
                    smell_penalty = excluded.smell_penalty,
                    adjusted_quality = excluded.adjusted_quality,
                    effort_factor = excluded.effort_factor,
                    total_effort = excluded.total_effort,
                    metrics_json = excluded.metrics_json
                """,
                (
                    entry.project_name,
                    entry.project_path,
                    entry.commit_sha_short,
                    entry.commit_sha_full,
                    entry.measured_at.isoformat(),
                    entry.qpe_score,
                    entry.mi_normalized,
                    entry.smell_penalty,
                    entry.adjusted_quality,
                    entry.effort_factor,
                    entry.total_effort,
                    entry.metrics_json,
                ),
            )
            conn.commit()

    def get_leaderboard(self) -> list[LeaderboardEntry]:
        """Get all leaderboard entries, sorted by QPE score (highest first)."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, project_name, project_path, commit_sha_short, commit_sha_full,
                       measured_at, qpe_score, mi_normalized, smell_penalty,
                       adjusted_quality, effort_factor, total_effort, metrics_json
                FROM qpe_leaderboard
                ORDER BY qpe_score DESC
                """
            ).fetchall()

            return [
                LeaderboardEntry(
                    id=row[0],
                    project_name=row[1],
                    project_path=row[2],
                    commit_sha_short=row[3],
                    commit_sha_full=row[4],
                    measured_at=datetime.fromisoformat(row[5]),
                    qpe_score=row[6],
                    mi_normalized=row[7],
                    smell_penalty=row[8],
                    adjusted_quality=row[9],
                    effort_factor=row[10],
                    total_effort=row[11],
                    metrics_json=row[12],
                )
                for row in rows
            ]

    def get_project_history(self, project_path: str) -> list[LeaderboardEntry]:
        """Get all leaderboard entries for a specific project, ordered by date."""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, project_name, project_path, commit_sha_short, commit_sha_full,
                       measured_at, qpe_score, mi_normalized, smell_penalty,
                       adjusted_quality, effort_factor, total_effort, metrics_json
                FROM qpe_leaderboard
                WHERE project_path = ?
                ORDER BY measured_at DESC
                """,
                (project_path,),
            ).fetchall()

            return [
                LeaderboardEntry(
                    id=row[0],
                    project_name=row[1],
                    project_path=row[2],
                    commit_sha_short=row[3],
                    commit_sha_full=row[4],
                    measured_at=datetime.fromisoformat(row[5]),
                    qpe_score=row[6],
                    mi_normalized=row[7],
                    smell_penalty=row[8],
                    adjusted_quality=row[9],
                    effort_factor=row[10],
                    total_effort=row[11],
                    metrics_json=row[12],
                )
                for row in rows
            ]

    def clear_leaderboard(self) -> int:
        """Clear all leaderboard entries.

        Returns:
            Number of entries deleted
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM qpe_leaderboard")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM qpe_leaderboard")
            return count


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

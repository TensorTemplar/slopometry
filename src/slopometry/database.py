"""Database operations for storing hook events."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from slopometry.models import (
    ComplexityDelta,
    ComplexityMetrics,
    GitState,
    HookEvent,
    HookEventType,
    PlanEvolution,
    Project,
    ProjectSource,
    SessionStatistics,
    ToolType,
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

        # Complexity analysis is now handled only in Stop hook events
        stats.complexity_metrics = None
        stats.complexity_delta = None

        try:
            stats.plan_evolution = self._calculate_plan_evolution(events)
        except Exception:
            stats.plan_evolution = None

        return stats

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

    def calculate_complexity_metrics(
        self, working_directory: str
    ) -> tuple[ComplexityMetrics | None, ComplexityDelta | None]:
        """Calculate complexity metrics for a specific working directory.

        This method is specifically for Stop hook events to avoid running
        complexity analysis during regular CLI operations.
        """
        try:
            from slopometry.complexity_analyzer import ComplexityAnalyzer
            from slopometry.git_tracker import GitTracker

            analyzer = ComplexityAnalyzer(working_directory=Path(working_directory))
            git_tracker = GitTracker()
            if git_tracker.has_previous_commit():
                baseline_dir = git_tracker.extract_files_from_commit()
                if baseline_dir:
                    try:
                        current_metrics, complexity_delta = analyzer.analyze_complexity_with_baseline(baseline_dir)
                        import shutil

                        shutil.rmtree(baseline_dir, ignore_errors=True)
                        return current_metrics, complexity_delta
                    except Exception:
                        current_metrics = analyzer.analyze_complexity()
                        return current_metrics, None
                else:
                    current_metrics = analyzer.analyze_complexity()
                    return current_metrics, None
            else:
                current_metrics = analyzer.analyze_complexity()
                return current_metrics, None
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

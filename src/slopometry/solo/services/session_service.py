"""Session management service for solo-leveler features."""

from slopometry.core.database import EventDatabase
from slopometry.core.models import SessionStatistics


class SessionService:
    """Handles basic session tracking and management for solo users."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def list_sessions(self, limit: int | None = None) -> list[str]:
        """List recent sessions, optionally limited."""
        all_sessions = self.db.list_sessions()
        return all_sessions[:limit] if limit else all_sessions

    def get_session_statistics(self, session_id: str) -> SessionStatistics | None:
        """Get detailed statistics for a session."""
        return self.db.get_session_statistics(session_id)

    def get_most_recent_session(self) -> str | None:
        """Get the ID of the most recent session."""
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None

    def cleanup_session(self, session_id: str) -> tuple[int, int]:
        """Clean up data for a specific session."""
        return self.db.cleanup_session(session_id)

    def cleanup_all_sessions(self) -> tuple[int, int, int]:
        """Clean up all session data."""
        return self.db.cleanup_all_sessions()

    def prepare_sessions_data_for_display(self, sessions: list[str]) -> list[dict]:
        """Prepare session data for display formatting."""
        sessions_data = []
        for session_id in sessions:
            stats = self.get_session_statistics(session_id)
            if stats:
                sessions_data.append(
                    {
                        "session_id": session_id,
                        "project_name": stats.project.name if stats.project else None,
                        "project_source": stats.project.source.value if stats.project else None,
                        "start_time": stats.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_events": stats.total_events,
                        "tools_used": len(stats.tool_usage),
                    }
                )
        return sessions_data

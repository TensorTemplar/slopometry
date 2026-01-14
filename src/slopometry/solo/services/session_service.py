"""Session management service for solo-leveler features."""

from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models import SessionStatistics


class SessionService:
    """Handles basic session tracking and management for solo users."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def list_sessions(self, limit: int | None = None) -> list[str]:
        """List recent sessions, optionally limited."""
        return self.db.list_sessions(limit=limit)

    def list_sessions_by_repository(self, repository_path: Path, limit: int | None = None) -> list[str]:
        """List sessions that occurred in a specific repository.

        Args:
            repository_path: The repository path to filter by
            limit: Optional limit on number of sessions to return

        Returns:
            List of session IDs that started in this repository
        """
        return self.db.list_sessions_by_repository(repository_path, limit=limit)

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

    def get_sessions_for_display(self, limit: int | None = None) -> list[dict]:
        """Get session summaries formatted for display."""
        summaries = self.db.get_sessions_summary(limit=limit)

        sessions_data = []
        for summary in summaries:
            from datetime import datetime

            try:
                start_time = datetime.fromisoformat(summary["start_time"])
                formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                formatted_time = summary["start_time"] or "Unknown"

            sessions_data.append(
                {
                    "session_id": summary["session_id"],
                    "project_name": summary["project_name"],
                    "project_source": summary["project_source"],
                    "start_time": formatted_time,
                    "total_events": summary["total_events"],
                    "tools_used": summary["tools_used"],
                }
            )

        return sessions_data

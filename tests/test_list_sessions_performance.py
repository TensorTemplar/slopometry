"""Tests for list_sessions performance optimizations."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from slopometry.core.database import EventDatabase
from slopometry.core.models.hook import HookEvent, HookEventType, Project, ProjectSource


class TestListSessionsPerformance:
    """Test list_sessions database query optimizations."""

    def test_list_sessions__respects_limit_parameter(self):
        """Test that list_sessions limit parameter works correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            base_time = datetime.now()
            sessions = []
            for i in range(10):
                session_id = f"session-{i:03d}"
                sessions.append(session_id)

                timestamp = base_time + timedelta(minutes=i)

                event = HookEvent(
                    session_id=session_id,
                    event_type=HookEventType.PRE_TOOL_USE,
                    timestamp=timestamp,
                    sequence_number=1,
                    working_directory="/test",
                    project=Project(name="test", source=ProjectSource.GIT),
                )
                db.save_event(event)

            all_sessions = db.list_sessions()

            limited_sessions = db.list_sessions(limit=3)

            assert len(all_sessions) == 10
            assert len(limited_sessions) == 3

            # Sessions should be ordered by timestamp DESC (newest first)
            assert limited_sessions == ["session-009", "session-008", "session-007"]

            # All sessions should be in correct order
            expected_order = [f"session-{i:03d}" for i in range(9, -1, -1)]
            assert all_sessions == expected_order

    def test_list_sessions__handles_no_limit_parameter(self):
        """Test that list_sessions works without limit parameter."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            for i in range(3):
                session_id = f"session-{i}"
                event = HookEvent(
                    session_id=session_id,
                    event_type=HookEventType.PRE_TOOL_USE,
                    timestamp=datetime.now() + timedelta(minutes=i),
                    sequence_number=1,
                    working_directory="/test",
                )
                db.save_event(event)

            sessions = db.list_sessions()

            assert len(sessions) == 3
            assert sessions == ["session-2", "session-1", "session-0"]  # Newest first

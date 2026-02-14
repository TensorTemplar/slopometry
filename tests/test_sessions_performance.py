"""Tests for session listing performance optimizations."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from slopometry.core.database import EventDatabase
from slopometry.core.models.hook import HookEvent, HookEventType, Project, ProjectSource, ToolType
from slopometry.solo.services.session_service import SessionService


class TestSessionsPerformance:
    """Test session listing performance optimizations."""

    def test_get_sessions_for_display__uses_single_query(self):
        """Test that get_sessions_for_display uses efficient single query."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            base_time = datetime.now()

            for i in range(5):
                event = HookEvent(
                    session_id="session-001",
                    event_type=HookEventType.PRE_TOOL_USE,
                    timestamp=base_time + timedelta(minutes=i),
                    sequence_number=i + 1,
                    working_directory="/test",
                    project=Project(name="project-a", source=ProjectSource.GIT),
                    tool_name="bash" if i < 3 else ("read" if i == 3 else "write"),
                    tool_type=ToolType.BASH if i < 3 else (ToolType.READ if i == 3 else ToolType.WRITE),
                )
                db.save_event(event)

            for i in range(3):
                event = HookEvent(
                    session_id="session-002",
                    event_type=HookEventType.PRE_TOOL_USE,
                    timestamp=base_time + timedelta(hours=1, minutes=i),
                    sequence_number=i + 1,
                    working_directory="/test2",
                    project=Project(name="project-b", source=ProjectSource.PYPROJECT),
                    tool_name="grep" if i < 2 else "ls",
                    tool_type=ToolType.GREP if i < 2 else ToolType.LS,
                )
                db.save_event(event)

            service = SessionService(db)

            sessions_data = service.get_sessions_for_display(limit=10)

            assert len(sessions_data) == 2

            assert sessions_data[0]["session_id"] == "session-002"  # Started later
            assert sessions_data[1]["session_id"] == "session-001"  # Started earlier

            session1 = sessions_data[1]
            assert session1["session_id"] == "session-001"
            assert session1["project_name"] == "project-a"
            assert session1["project_source"] == "git"
            assert session1["total_events"] == 5
            assert session1["tools_used"] == 3  # bash, read, write

            session2 = sessions_data[0]
            assert session2["session_id"] == "session-002"
            assert session2["project_name"] == "project-b"
            assert session2["project_source"] == "pyproject"
            assert session2["total_events"] == 3
            assert session2["tools_used"] == 2  # grep, ls

    def test_get_sessions_for_display__respects_limit(self):
        """Test that limit parameter works correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            base_time = datetime.now()
            for session_num in range(5):
                event = HookEvent(
                    session_id=f"session-{session_num:03d}",
                    event_type=HookEventType.PRE_TOOL_USE,
                    timestamp=base_time + timedelta(minutes=session_num),
                    sequence_number=1,
                    working_directory="/test",
                    tool_name="bash",
                    tool_type=ToolType.BASH,
                )
                db.save_event(event)

            service = SessionService(db)

            limited_sessions = service.get_sessions_for_display(limit=3)
            all_sessions = service.get_sessions_for_display()

            assert len(all_sessions) == 5
            assert len(limited_sessions) == 3

            expected_sessions = ["session-004", "session-003", "session-002"]
            actual_sessions = [s["session_id"] for s in limited_sessions]
            assert actual_sessions == expected_sessions

    def test_get_sessions_for_display__handles_empty_database(self):
        """Test that method handles empty database gracefully."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)
            service = SessionService(db)

            sessions_data = service.get_sessions_for_display()

            assert sessions_data == []

    def test_get_sessions_for_display__handles_null_tool_types(self):
        """Test that method handles NULL tool_type values correctly."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = EventDatabase(db_path)

            event = HookEvent(
                session_id="session-001",
                event_type=HookEventType.NOTIFICATION,
                timestamp=datetime.now(),
                sequence_number=1,
                working_directory="/test",
                tool_name=None,
                tool_type=None,
            )
            db.save_event(event)

            service = SessionService(db)

            sessions_data = service.get_sessions_for_display()

            assert len(sessions_data) == 1
            assert sessions_data[0]["tools_used"] == 0

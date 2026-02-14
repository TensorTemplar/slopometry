"""Test database functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models.display import LeaderboardEntry
from slopometry.core.models.hook import HookEvent, HookEventType, ToolType
from slopometry.core.models.user_story import UserStoryEntry


def test_user_story_export_functionality() -> None:
    """Test exporting user stories with existing or minimal test data."""
    db = EventDatabase()

    stats = db.get_user_story_stats()

    if stats["total_entries"] == 0:
        test_entry = UserStoryEntry(
            base_commit="test-base",
            head_commit="test-head",
            diff_content="diff --git a/test.py b/test.py\n+def hello():\n+    print('world')",
            user_stories="## Test User Stories\n\n1. As a tester, I want test functionality...",
            rating=3,
            guidelines_for_improving="Test guidelines",
            model_used="test-model",
            prompt_template="Test prompt template",
            repository_path=str(Path.cwd()),
        )
        db.save_user_story_entry(test_entry)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        count = db.export_user_stories(output_path)

        assert count >= 1, f"Expected at least 1 entry, got {count}"
        assert output_path.exists(), "Export file was not created"
        assert output_path.stat().st_size > 0, "Export file is empty"

        import pandas as pd

        df = pd.read_parquet(output_path)

        expected_columns = [
            "id",
            "created_at",
            "base_commit",
            "head_commit",
            "diff_content",
            "user_stories",
            "rating",
            "guidelines_for_improving",
            "model_used",
            "prompt_template",
            "repository_path",
        ]
        assert all(col in df.columns for col in expected_columns)
        assert len(df) >= 1

    finally:
        if output_path.exists():
            output_path.unlink()


def test_user_story_stats() -> None:
    """Test user story statistics calculation."""
    db = EventDatabase()

    stats = db.get_user_story_stats()

    assert stats["total_entries"] >= 0
    assert "avg_rating" in stats
    assert "unique_models" in stats
    assert "unique_repos" in stats
    assert "rating_distribution" in stats


def test_user_story_generation_cli_integration() -> None:
    """Test that the CLI command for generating user story entries works.

    Note: Does not run the actual command as it requires LLM access.
    """
    from click.testing import CliRunner

    from slopometry.cli import cli

    runner = CliRunner()

    result = runner.invoke(cli, ["summoner", "userstorify", "--help"])
    assert result.exit_code == 0
    assert "Generate user stories from commits using configured AI agents" in result.output
    assert "--base-commit" in result.output
    assert "--head-commit" in result.output


def test_leaderboard_upsert__updates_existing_project_on_new_commit() -> None:
    """Test that saving a leaderboard entry with same project_path but different commit updates the entry."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        project_path = "/test/project"
        entry_v1 = LeaderboardEntry(
            project_name="test-project",
            project_path=project_path,
            commit_sha_short="abc1234",
            commit_sha_full="abc1234567890",
            measured_at=datetime(2023, 1, 1),
            qpe_score=0.5,
            mi_normalized=0.6,
            smell_penalty=0.1,
            adjusted_quality=0.7,
            effort_factor=1.2,
            total_effort=1000.0,
            metrics_json="{}",
        )
        db.save_leaderboard_entry(entry_v1)

        leaderboard = db.get_leaderboard()
        assert len(leaderboard) == 1
        assert leaderboard[0].commit_sha_short == "abc1234"
        assert leaderboard[0].qpe_score == 0.5

        entry_v2 = LeaderboardEntry(
            project_name="test-project",
            project_path=project_path,
            commit_sha_short="def5678",
            commit_sha_full="def5678901234",
            measured_at=datetime(2024, 6, 1),
            qpe_score=0.8,
            mi_normalized=0.7,
            smell_penalty=0.05,
            adjusted_quality=0.85,
            effort_factor=1.1,
            total_effort=1200.0,
            metrics_json='{"updated": true}',
        )
        db.save_leaderboard_entry(entry_v2)

        leaderboard = db.get_leaderboard()
        assert len(leaderboard) == 1, "Should update existing entry, not create duplicate"
        assert leaderboard[0].commit_sha_short == "def5678"
        assert leaderboard[0].commit_sha_full == "def5678901234"
        assert leaderboard[0].qpe_score == 0.8
        assert leaderboard[0].measured_at == datetime(2024, 6, 1)


def test_leaderboard_save__round_trips_qpe_weight_version() -> None:
    """Test that qpe_weight_version is persisted and retrieved from the leaderboard."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        entry_with_version = LeaderboardEntry(
            project_name="versioned-project",
            project_path="/test/versioned",
            commit_sha_short="aaa1111",
            commit_sha_full="aaa1111222233334444",
            measured_at=datetime(2024, 1, 1),
            qpe_score=0.6,
            mi_normalized=0.7,
            smell_penalty=0.05,
            adjusted_quality=0.65,
            effort_factor=1.0,
            total_effort=500.0,
            metrics_json="{}",
            qpe_weight_version="2",
        )
        entry_without_version = LeaderboardEntry(
            project_name="legacy-project",
            project_path="/test/legacy",
            commit_sha_short="bbb2222",
            commit_sha_full="bbb2222333344445555",
            measured_at=datetime(2024, 1, 1),
            qpe_score=0.5,
            mi_normalized=0.6,
            smell_penalty=0.1,
            adjusted_quality=0.54,
            effort_factor=1.0,
            total_effort=600.0,
            metrics_json="{}",
        )
        db.save_leaderboard_entry(entry_with_version)
        db.save_leaderboard_entry(entry_without_version)

        leaderboard = db.get_leaderboard()
        by_name = {e.project_name: e for e in leaderboard}

        assert by_name["versioned-project"].qpe_weight_version == "2"
        assert by_name["legacy-project"].qpe_weight_version is None


def test_clear_leaderboard__removes_all_entries() -> None:
    """Test that clear_leaderboard removes all entries and returns count."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        entry1 = LeaderboardEntry(
            project_name="project1",
            project_path="/path/to/project1",
            commit_sha_short="abc1234",
            commit_sha_full="abc1234567890",
            measured_at=datetime(2024, 1, 1),
            qpe_score=0.5,
            mi_normalized=0.7,
            smell_penalty=0.1,
            adjusted_quality=0.63,
            effort_factor=10.0,
            total_effort=20000.0,
            metrics_json="{}",
        )
        entry2 = LeaderboardEntry(
            project_name="project2",
            project_path="/path/to/project2",
            commit_sha_short="def5678",
            commit_sha_full="def5678901234",
            measured_at=datetime(2024, 2, 1),
            qpe_score=0.6,
            mi_normalized=0.8,
            smell_penalty=0.05,
            adjusted_quality=0.76,
            effort_factor=12.0,
            total_effort=25000.0,
            metrics_json="{}",
        )
        db.save_leaderboard_entry(entry1)
        db.save_leaderboard_entry(entry2)

        assert len(db.get_leaderboard()) == 2

        deleted_count = db.clear_leaderboard()

        assert deleted_count == 2
        assert len(db.get_leaderboard()) == 0


def test_list_sessions_by_repository__filters_correctly() -> None:
    """Sessions should be filtered by working directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        # Session 1 - in repo A
        db.save_event(
            HookEvent(
                session_id="session-repo-a",
                event_type=HookEventType.PRE_TOOL_USE,
                sequence_number=1,
                working_directory="/path/to/repo-a",
                tool_name="Read",
                tool_type=ToolType.READ,
            )
        )

        # Session 2 - in repo B
        db.save_event(
            HookEvent(
                session_id="session-repo-b",
                event_type=HookEventType.PRE_TOOL_USE,
                sequence_number=1,
                working_directory="/path/to/repo-b",
                tool_name="Read",
                tool_type=ToolType.READ,
            )
        )

        # Session 3 - also in repo A
        db.save_event(
            HookEvent(
                session_id="session-repo-a-2",
                event_type=HookEventType.PRE_TOOL_USE,
                sequence_number=1,
                working_directory="/path/to/repo-a",
                tool_name="Write",
                tool_type=ToolType.WRITE,
            )
        )

        sessions = db.list_sessions_by_repository(Path("/path/to/repo-a"))

        assert len(sessions) == 2
        assert "session-repo-a" in sessions
        assert "session-repo-a-2" in sessions
        assert "session-repo-b" not in sessions


def test_list_sessions_by_repository__returns_empty_for_unknown_repo() -> None:
    """Unknown repository should return empty list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        # Create a session in a known repo
        db.save_event(
            HookEvent(
                session_id="session-known",
                event_type=HookEventType.PRE_TOOL_USE,
                sequence_number=1,
                working_directory="/path/to/known-repo",
                tool_name="Read",
                tool_type=ToolType.READ,
            )
        )

        sessions = db.list_sessions_by_repository(Path("/path/to/unknown"))

        assert sessions == []


def test_list_sessions_by_repository__respects_limit() -> None:
    """Session list should respect the limit parameter."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        # Create 3 sessions in the same repo
        for i in range(3):
            db.save_event(
                HookEvent(
                    session_id=f"session-{i}",
                    event_type=HookEventType.PRE_TOOL_USE,
                    sequence_number=1,
                    working_directory="/path/to/repo",
                    tool_name="Read",
                    tool_type=ToolType.READ,
                )
            )

        sessions = db.list_sessions_by_repository(Path("/path/to/repo"), limit=2)

        assert len(sessions) == 2


def test_get_session_basic_info__returns_minimal_info() -> None:
    """get_session_basic_info returns just start_time and total_events without expensive computations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        db.save_event(
            HookEvent(
                session_id="test-session",
                event_type=HookEventType.PRE_TOOL_USE,
                sequence_number=1,
                working_directory="/path/to/repo",
                tool_name="Read",
                tool_type=ToolType.READ,
            )
        )
        db.save_event(
            HookEvent(
                session_id="test-session",
                event_type=HookEventType.POST_TOOL_USE,
                sequence_number=2,
                working_directory="/path/to/repo",
                tool_name="Write",
                tool_type=ToolType.WRITE,
            )
        )

        result = db.get_session_basic_info("test-session")

        assert result is not None
        start_time, total_events = result
        assert isinstance(start_time, datetime)
        assert total_events == 2


def test_get_session_basic_info__returns_none_for_unknown_session() -> None:
    """get_session_basic_info returns None for non-existent session."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        result = db.get_session_basic_info("nonexistent-session")

        assert result is None

"""Tests for the save-transcript command."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from slopometry.core.models import SessionStatistics
from slopometry.solo.cli.commands import (
    _find_plan_names_from_transcript,
    _find_session_todos,
    save_transcript,
)


class TestFindPlanNamesFromTranscript:
    """Tests for _find_plan_names_from_transcript helper."""

    def test_find_plan_names__extracts_plan_names_from_transcript(self, tmp_path):
        """Test extracting plan names from transcript content."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            '{"message": "Plan at plans/reactive-chasing-dawn.md"}\n'
            '{"message": "Another line with plans/elegant-leaping-panda.md"}\n'
        )

        result = _find_plan_names_from_transcript(transcript)

        assert set(result) == {"reactive-chasing-dawn.md", "elegant-leaping-panda.md"}

    def test_find_plan_names__returns_empty_for_no_plans(self, tmp_path):
        """Test returns empty list when no plans found."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text('{"message": "No plan references here"}\n')

        result = _find_plan_names_from_transcript(transcript)

        assert result == []

    def test_find_plan_names__handles_missing_file(self, tmp_path):
        """Test gracefully handles missing transcript file."""
        missing_path = tmp_path / "nonexistent.jsonl"

        result = _find_plan_names_from_transcript(missing_path)

        assert result == []

    def test_find_plan_names__deduplicates_plan_names(self, tmp_path):
        """Test that duplicate plan names are deduplicated."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text('{"message": "plans/same-plan.md"}\n{"message": "plans/same-plan.md again"}\n')

        result = _find_plan_names_from_transcript(transcript)

        assert result == ["same-plan.md"]


class TestFindSessionTodos:
    """Tests for _find_session_todos helper."""

    def test_find_session_todos__finds_matching_todos(self, tmp_path):
        """Test finding todos matching session ID pattern."""
        session_id = "abc123"
        todos_dir = tmp_path / ".claude" / "todos"
        todos_dir.mkdir(parents=True)

        # Create matching todo files
        (todos_dir / f"{session_id}-agent-{session_id}.json").write_text("[]")
        (todos_dir / f"{session_id}-agent-other.json").write_text("[]")
        # Non-matching file
        (todos_dir / "other-session-agent-xyz.json").write_text("[]")

        with patch.object(Path, "home", return_value=tmp_path):
            result = _find_session_todos(session_id)

        assert len(result) == 2
        assert all(session_id in str(p) for p in result)

    def test_find_session_todos__returns_empty_when_no_todos_dir(self, tmp_path):
        """Test returns empty list when todos directory doesn't exist."""
        with patch.object(Path, "home", return_value=tmp_path):
            result = _find_session_todos("any-session")

        assert result == []

    def test_find_session_todos__returns_empty_when_no_matches(self, tmp_path):
        """Test returns empty list when no matching todos found."""
        todos_dir = tmp_path / ".claude" / "todos"
        todos_dir.mkdir(parents=True)
        (todos_dir / "other-session-agent.json").write_text("[]")

        with patch.object(Path, "home", return_value=tmp_path):
            result = _find_session_todos("nonexistent-session")

        assert result == []


class TestSaveTranscript:
    """Test save-transcript command functionality."""

    def test_save_transcript__creates_session_directory_structure(self, tmp_path):
        """Test creating .slopometry/<session-id>/ directory structure."""
        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"test": "data"}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch("slopometry.solo.cli.commands._find_plan_names_from_transcript", return_value=[]),
            patch("slopometry.solo.cli.commands._find_session_todos", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Saved transcript to:" in result.output

        # Verify directory structure
        session_dir = output_dir / ".slopometry" / session_id
        assert session_dir.exists()
        assert (session_dir / "transcript.jsonl").exists()
        assert (session_dir / "transcript.jsonl").read_text() == '{"test": "data"}'

    def test_save_transcript__copies_plans_from_transcript_references(self, tmp_path):
        """Test copying plans referenced in transcript."""
        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"message": "plans/my-plan.md"}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock plan file
        plans_dir = tmp_path / ".claude" / "plans"
        plans_dir.mkdir(parents=True)
        (plans_dir / "my-plan.md").write_text("# My Plan")

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch.object(Path, "home", return_value=tmp_path),
            patch("slopometry.solo.cli.commands._find_session_todos", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Saved plan: my-plan.md" in result.output

        # Verify plan was copied
        copied_plan = output_dir / ".slopometry" / session_id / "plans" / "my-plan.md"
        assert copied_plan.exists()
        assert copied_plan.read_text() == "# My Plan"

    def test_save_transcript__copies_todos_matching_session_id(self, tmp_path):
        """Test copying todos that match session ID pattern."""
        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"test": "data"}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock todo files
        todos_dir = tmp_path / ".claude" / "todos"
        todos_dir.mkdir(parents=True)
        todo_file = todos_dir / f"{session_id}-agent-{session_id}.json"
        todo_file.write_text('[{"task": "test"}]')

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch.object(Path, "home", return_value=tmp_path),
            patch("slopometry.solo.cli.commands._find_plan_names_from_transcript", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        assert result.exit_code == 0
        assert f"Saved todo: {session_id}-agent-{session_id}.json" in result.output

        # Verify todo was copied
        copied_todo = output_dir / ".slopometry" / session_id / "todos" / f"{session_id}-agent-{session_id}.json"
        assert copied_todo.exists()
        assert copied_todo.read_text() == '[{"task": "test"}]'

    def test_save_transcript__handles_missing_plans_gracefully(self, tmp_path):
        """Test graceful handling when referenced plan doesn't exist."""
        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"message": "plans/nonexistent-plan.md"}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch.object(Path, "home", return_value=tmp_path),
            patch("slopometry.solo.cli.commands._find_session_todos", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        # Should succeed without error
        assert result.exit_code == 0
        assert "Saved transcript to:" in result.output
        # No plan saved message
        assert "Saved plan:" not in result.output

    def test_save_transcript__shows_error_when_session_not_found(self):
        """Test error handling when session doesn't exist."""
        session_id = "non-existent"

        with patch("slopometry.solo.services.session_service.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = None

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id])

        assert result.exit_code == 0
        assert "No data found for session" in result.output

    def test_save_transcript__shows_error_when_no_transcript_path(self):
        """Test error handling when session has no transcript path."""
        session_id = "test-session"
        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory="/tmp",
            transcript_path=None,
        )

        with patch("slopometry.solo.services.session_service.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id])

        assert result.exit_code == 0
        assert "No transcript path found" in result.output
        assert "older session" in result.output

    def test_save_transcript__uses_latest_session_when_no_id_provided(self, tmp_path):
        """Test using latest session when no session ID is provided."""
        session_id = "latest-session-456"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"latest": "data"}')

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
            total_events=42,
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch("slopometry.solo.cli.commands._find_plan_names_from_transcript", return_value=[]),
            patch("slopometry.solo.cli.commands._find_session_todos", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [], input="y\n")

        assert result.exit_code == 0
        assert f"Latest session: {session_id}" in result.output
        assert "Total events: 42" in result.output
        assert "Save transcript for this session?" in result.output
        assert "Saved transcript to:" in result.output

    def test_save_transcript__skips_confirmation_with_yes_flag(self, tmp_path):
        """Test skipping confirmation with --yes flag."""
        session_id = "latest-session-456"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"latest": "data"}')

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
            total_events=42,
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch("slopometry.solo.cli.commands._find_plan_names_from_transcript", return_value=[]),
            patch("slopometry.solo.cli.commands._find_session_todos", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, ["--yes"])

        assert result.exit_code == 0
        assert "Save transcript for this session?" not in result.output
        assert "Saved transcript to:" in result.output

    def test_save_transcript__shows_error_when_no_sessions_exist(self):
        """Test error handling when no sessions exist at all."""
        with patch("slopometry.solo.services.session_service.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = None

            runner = CliRunner()
            result = runner.invoke(save_transcript, [])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_save_transcript__cancels_when_user_declines_confirmation(self, tmp_path):
        """Test cancellation when user declines confirmation for latest session."""
        session_id = "latest-session-456"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"latest": "data"}')

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
            total_events=42,
        )

        with patch("slopometry.solo.services.session_service.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        assert "Saved transcript to:" not in result.output

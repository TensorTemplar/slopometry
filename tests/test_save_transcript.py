"""Tests for the save-transcript command."""

from datetime import datetime
from unittest.mock import Mock, patch

from click.testing import CliRunner

from slopometry.core.models import SessionStatistics
from slopometry.solo.cli.commands import save_transcript


class TestSaveTranscript:
    """Test save-transcript command functionality."""

    def test_save_transcript__saves_and_adds_to_git_when_transcript_exists(self, tmp_path):
        """Test saving transcript when transcript path exists."""
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

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

                runner = CliRunner()
                result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Saved transcript to:" in result.output
        assert "Added to git:" in result.output

        expected_file = output_dir / ".slopometry" / f"claude-transcript-{session_id}.jsonl"
        assert expected_file.exists()
        assert expected_file.read_text() == '{"test": "data"}'

    def test_save_transcript__shows_error_when_session_not_found(self):
        """Test error handling when session doesn't exist."""
        session_id = "non-existent"

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
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

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id])

        assert result.exit_code == 0
        assert "No transcript path found" in result.output
        assert "older session" in result.output

    def test_save_transcript__skips_git_when_no_git_add_flag(self, tmp_path):
        """Test that git add is skipped with --no-git-add flag."""
        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"test": "data"}')

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
        )

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            with patch("subprocess.run") as mock_run:
                runner = CliRunner()
                result = runner.invoke(save_transcript, [session_id, "--no-git-add"])

        assert result.exit_code == 0
        assert "Saved transcript to:" in result.output
        assert "Added to git:" not in result.output
        mock_run.assert_not_called()

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

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

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

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

                runner = CliRunner()
                result = runner.invoke(save_transcript, ["--yes"])

        assert result.exit_code == 0
        assert "Save transcript for this session?" not in result.output
        assert "Saved transcript to:" in result.output

    def test_save_transcript__shows_error_when_no_sessions_exist(self):
        """Test error handling when no sessions exist at all."""
        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
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

        with patch("slopometry.solo.cli.commands.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = session_id
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output
        assert "Saved transcript to:" not in result.output

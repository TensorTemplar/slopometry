"""Tests for the save-transcript command."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from slopometry.core.models.hook import AgentTool
from slopometry.core.models.session import PlanEvolution, SessionMetadata, SessionStatistics, TodoItem, TokenUsage
from slopometry.core.transcript_token_analyzer import TranscriptMetadata, extract_transcript_metadata
from slopometry.solo.cli.commands import (
    _find_plan_names_from_transcript,
    save_transcript,
)


class TestFindPlanNamesFromTranscript:
    """Tests for _find_plan_names_from_transcript helper."""

    def test_find_plan_names__extracts_plan_names_from_transcript(self, tmp_path) -> None:
        """Test extracting plan names from transcript content."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text(
            '{"message": "Plan at plans/reactive-chasing-dawn.md"}\n'
            '{"message": "Another line with plans/elegant-leaping-panda.md"}\n'
        )

        result = _find_plan_names_from_transcript(transcript)

        assert set(result) == {"reactive-chasing-dawn.md", "elegant-leaping-panda.md"}

    def test_find_plan_names__returns_empty_for_no_plans(self, tmp_path) -> None:
        """Test returns empty list when no plans found."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text('{"message": "No plan references here"}\n')

        result = _find_plan_names_from_transcript(transcript)

        assert result == []

    def test_find_plan_names__handles_missing_file(self, tmp_path) -> None:
        """Test gracefully handles missing transcript file."""
        missing_path = tmp_path / "nonexistent.jsonl"

        result = _find_plan_names_from_transcript(missing_path)

        assert result == []

    def test_find_plan_names__deduplicates_plan_names(self, tmp_path) -> None:
        """Test that duplicate plan names are deduplicated."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text('{"message": "plans/same-plan.md"}\n{"message": "plans/same-plan.md again"}\n')

        result = _find_plan_names_from_transcript(transcript)

        assert result == ["same-plan.md"]


class TestSaveTranscript:
    """Test save-transcript command functionality."""

    def test_save_transcript__creates_session_directory_structure(self, tmp_path) -> None:
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
        assert (session_dir / "session_metadata.json").exists()

        metadata = json.loads((session_dir / "session_metadata.json").read_text())
        assert metadata["session_id"] == session_id
        assert metadata["agent_tool"] == "claude_code"

    def test_save_transcript__copies_plans_from_transcript_references(self, tmp_path) -> None:
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

    def test_save_transcript__saves_final_todos_from_plan_evolution(self, tmp_path) -> None:
        """Test saving final_todos.json from plan_evolution."""
        import json

        session_id = "test-session-123"
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text('{"test": "data"}')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock stats with plan_evolution containing final_todos
        mock_plan_evolution = PlanEvolution(
            final_todos=[
                TodoItem(content="Task 1", status="completed", activeForm="Completing task 1"),
                TodoItem(content="Task 2", status="in_progress", activeForm="Working on task 2"),
            ]
        )

        mock_stats = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now(),
            working_directory=str(tmp_path),
            transcript_path=str(transcript_path),
            plan_evolution=mock_plan_evolution,
        )

        with (
            patch("slopometry.solo.services.session_service.SessionService") as mock_service_class,
            patch("slopometry.solo.cli.commands._find_plan_names_from_transcript", return_value=[]),
        ):
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_session_statistics.return_value = mock_stats

            runner = CliRunner()
            result = runner.invoke(save_transcript, [session_id, "-o", str(output_dir)])

        assert result.exit_code == 0
        assert "Saved 2 todos to: final_todos.json" in result.output

        # Verify final_todos.json was created with correct content
        todos_file = output_dir / ".slopometry" / session_id / "final_todos.json"
        assert todos_file.exists()

        saved_todos = json.loads(todos_file.read_text())
        assert len(saved_todos) == 2
        assert saved_todos[0]["content"] == "Task 1"
        assert saved_todos[0]["status"] == "completed"
        assert saved_todos[1]["content"] == "Task 2"
        assert saved_todos[1]["status"] == "in_progress"

    def test_save_transcript__handles_missing_plans_gracefully(self, tmp_path) -> None:
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

    def test_save_transcript__shows_error_when_session_not_found(self) -> None:
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

    def test_save_transcript__shows_error_when_no_transcript_path(self) -> None:
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

    def test_save_transcript__uses_latest_session_when_no_id_provided(self, tmp_path) -> None:
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

    def test_save_transcript__skips_confirmation_with_yes_flag(self, tmp_path) -> None:
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

    def test_save_transcript__shows_error_when_no_sessions_exist(self) -> None:
        """Test error handling when no sessions exist at all."""
        with patch("slopometry.solo.services.session_service.SessionService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_most_recent_session.return_value = None

            runner = CliRunner()
            result = runner.invoke(save_transcript, [])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_save_transcript__cancels_when_user_declines_confirmation(self, tmp_path) -> None:
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


class TestExtractTranscriptMetadata:
    """Tests for the extract_transcript_metadata function."""

    @staticmethod
    def _fixture_transcript_path() -> Path:
        path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert path.exists(), f"transcript.jsonl fixture missing at {path}"
        return path

    def test_extract_transcript_metadata__extracts_version_and_model(self) -> None:
        """Real transcript fixture contains version, model, and branch."""
        meta = extract_transcript_metadata(self._fixture_transcript_path())

        assert meta.agent_version == "2.0.65"
        assert meta.model == "claude-opus-4-5-20251101"
        assert meta.git_branch == "opinionated-metrics"

    def test_extract_transcript_metadata__missing_fields_returns_none(self, tmp_path: Path) -> None:
        """Transcript with no version/model/branch fields returns None values."""
        transcript = tmp_path / "bare.jsonl"
        transcript.write_text(
            '{"type":"summary","summary":"test"}\n{"type":"user","message":{"role":"user","content":"hello"}}\n'
        )

        meta = extract_transcript_metadata(transcript)

        assert meta.agent_version is None
        assert meta.model is None
        assert meta.git_branch is None

    def test_extract_transcript_metadata__handles_missing_file(self) -> None:
        """Missing file returns all-None metadata without raising."""
        meta = extract_transcript_metadata(Path("/nonexistent/transcript.jsonl"))

        assert meta.agent_version is None
        assert meta.model is None
        assert meta.git_branch is None

    def test_extract_transcript_metadata__handles_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON lines are skipped without error."""
        transcript = tmp_path / "malformed.jsonl"
        transcript.write_text('not json at all\n{"version":"1.0.0","gitBranch":"main","type":"user"}\n')

        meta = extract_transcript_metadata(transcript)

        assert meta.agent_version == "1.0.0"
        assert meta.git_branch == "main"
        assert meta.model is None

    def test_extract_transcript_metadata__returns_pydantic_model(self, tmp_path: Path) -> None:
        """Return type is TranscriptMetadata, not a raw dict."""
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")

        meta = extract_transcript_metadata(transcript)

        assert isinstance(meta, TranscriptMetadata)


class TestSessionMetadata:
    """Tests for the SessionMetadata model."""

    def test_session_metadata__serializes_with_token_usage(self) -> None:
        """SessionMetadata JSON includes nested token_usage breakdown."""
        token_usage = TokenUsage(
            total_input_tokens=5000,
            total_output_tokens=2000,
            exploration_input_tokens=3000,
            exploration_output_tokens=1000,
            implementation_input_tokens=2000,
            implementation_output_tokens=1000,
            subagent_tokens=500,
        )
        metadata = SessionMetadata(
            session_id="test-session-123",
            agent_tool=AgentTool.CLAUDE_CODE,
            agent_version="2.0.65",
            model="claude-opus-4-5-20251101",
            start_time=datetime(2025, 12, 12, 13, 0, 0),
            end_time=datetime(2025, 12, 12, 14, 0, 0),
            total_events=42,
            working_directory="/home/user/project",
            git_branch="main",
            token_usage=token_usage,
        )

        data = json.loads(metadata.model_dump_json())

        assert data["token_usage"]["total_input_tokens"] == 5000
        assert data["token_usage"]["total_output_tokens"] == 2000
        assert data["token_usage"]["exploration_input_tokens"] == 3000
        assert data["token_usage"]["subagent_tokens"] == 500

    def test_session_metadata__agent_tool_discriminator(self) -> None:
        """AgentTool enum serializes as its string value."""
        metadata = SessionMetadata(
            session_id="test-123",
            agent_tool=AgentTool.CLAUDE_CODE,
            start_time=datetime(2025, 1, 1),
            working_directory="/tmp",
        )

        data = json.loads(metadata.model_dump_json())
        assert data["agent_tool"] == "claude_code"

        metadata_oc = SessionMetadata(
            session_id="test-456",
            agent_tool=AgentTool.OPENCODE,
            start_time=datetime(2025, 1, 1),
            working_directory="/tmp",
        )

        data_oc = json.loads(metadata_oc.model_dump_json())
        assert data_oc["agent_tool"] == "opencode"

    def test_session_metadata__optional_fields_default_to_none(self) -> None:
        """Optional fields default to None when not provided."""
        metadata = SessionMetadata(
            session_id="test-789",
            agent_tool=AgentTool.CLAUDE_CODE,
            start_time=datetime(2025, 1, 1),
            working_directory="/tmp",
        )

        assert metadata.agent_version is None
        assert metadata.model is None
        assert metadata.end_time is None
        assert metadata.git_branch is None
        assert metadata.token_usage is None
        assert metadata.total_events == 0

    def test_session_metadata__roundtrip_json(self) -> None:
        """SessionMetadata survives JSON serialization roundtrip."""
        original = SessionMetadata(
            session_id="roundtrip-test",
            agent_tool=AgentTool.CLAUDE_CODE,
            agent_version="2.0.65",
            model="claude-opus-4-5-20251101",
            start_time=datetime(2025, 12, 12, 13, 0, 0),
            total_events=10,
            working_directory="/home/user/project",
            git_branch="feature-branch",
        )

        json_str = original.model_dump_json()
        restored = SessionMetadata.model_validate_json(json_str)

        assert restored.session_id == original.session_id
        assert restored.agent_tool == original.agent_tool
        assert restored.agent_version == original.agent_version
        assert restored.model == original.model
        assert restored.total_events == original.total_events
        assert restored.git_branch == original.git_branch

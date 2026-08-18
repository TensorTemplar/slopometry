"""Tests for the canonical protocol events defined in `slopometry.core.protocol.events`.

Tests both happy-path validation (field defaults, type coercion) and the
`extra="forbid"` contract: downstream analyzers and storage must reject unknown
fields rather than silently ignore them, so wire-format drift is caught early.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from slopometry.core.models.hook import Project, ProjectSource
from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
    ToolCallPayload,
)


class TestAbstractEventSource:
    """Tests for the AbstractEventSource enum."""

    def test_event_source__claude_code_value(self):
        """Claude Code is the canonical first source."""
        assert AbstractEventSource.CLAUDE_CODE == "claude_code"

    def test_event_source__opencode_value(self):
        """OpenCode is the second canonical source."""
        assert AbstractEventSource.OPENCODE == "opencode"


class TestAbstractEventType:
    """Tests for the AbstractEventType enum."""

    def test_event_type__all_event_types_are_snake_case(self):
        """Abstract event-type values must be snake_case for stable DB column strings."""
        for et in AbstractEventType:
            assert "_" in et.value or et.value.islower(), f"{et.name} should be snake_case, got {et.value!r}"

    def test_event_type__tool_call_started(self):
        assert AbstractEventType.TOOL_CALL_STARTED == "tool_call_started"

    def test_event_type__tool_call_completed(self):
        assert AbstractEventType.TOOL_CALL_COMPLETED == "tool_call_completed"

    def test_event_type__notification(self):
        assert AbstractEventType.NOTIFICATION == "notification"

    def test_event_type__turn_completed(self):
        assert AbstractEventType.TURN_COMPLETED == "turn_completed"

    def test_event_type__subagent_completed(self):
        assert AbstractEventType.SUBAGENT_COMPLETED == "subagent_completed"

    def test_event_type__todo_updated(self):
        assert AbstractEventType.TODO_UPDATED == "todo_updated"

    def test_event_type__message_updated(self):
        assert AbstractEventType.MESSAGE_UPDATED == "message_updated"

    def test_event_type__subagent_started(self):
        assert AbstractEventType.SUBAGENT_STARTED == "subagent_started"


class TestToolCallPayloadValidation:
    """Tests for ToolCallPayload validation and field defaults."""

    def test_tool_call_payload__requires_tool_name_and_input(self):
        """tool_name and input are required — others are optional."""
        payload = ToolCallPayload(tool_name="Read", input={"file_path": "/x.py"})
        assert payload.tool_name == "Read"
        assert payload.input == {"file_path": "/x.py"}
        assert payload.tool_type is None
        assert payload.output is None
        assert payload.duration_ms is None
        assert payload.exit_code is None
        assert payload.error_message is None

    def test_tool_call_payload__all_optional_fields_default_to_none(self):
        """Optional fields must default to None (not absent) so DB writes are consistent."""
        payload = ToolCallPayload(tool_name="Bash", input={"command": "ls"})
        for field in ("tool_type", "output", "duration_ms", "exit_code", "error_message"):
            assert getattr(payload, field) is None

    def test_tool_call_payload__accepts_dict_input(self):
        """Input can be an arbitrary dict — tool-specific shape, no schema enforcement."""
        payload = ToolCallPayload(tool_name="Read", input={"file_path": "/x.py", "limit": 100})
        assert payload.input == {"file_path": "/x.py", "limit": 100}

    def test_tool_call_payload__output_accepts_arbitrary_shape(self):
        """Output can be dict, str, list — the wire shape varies by tool."""
        for output in (
            {"success": True, "content": "x"},
            "stdout line 1\nstdout line 2",
            [{"cellType": "code", "source": "print('hi')"}],
            None,
        ):
            payload = ToolCallPayload(tool_name="T", input={}, output=output)
            assert payload.output == output

    def test_tool_call_payload__rejects_unknown_field(self):
        """extra='forbid' — unknown fields surface as ValidationError, not silent drop."""
        with pytest.raises(ValidationError):
            ToolCallPayload(tool_name="Read", input={}, unknown_field="bogus")  # pyright: ignore[reportCallIssue]


class TestAbstractHookEventValidation:
    """Tests for AbstractHookEvent validation and field defaults."""

    def test_hook_event__required_fields(self):
        """session_id, event_type, source, working_directory are required."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.session_id == "s1"
        assert event.event_type == AbstractEventType.NOTIFICATION
        assert event.source == AbstractEventSource.CLAUDE_CODE
        assert event.working_directory == "/repo"

    def test_hook_event__timestamp_defaults_to_now(self):
        """timestamp defaults to datetime.now() at construction time."""
        before = datetime.now()
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        after = datetime.now()
        assert before <= event.timestamp <= after

    def test_hook_event__sequence_number_defaults_to_zero(self):
        """SessionManager assigns sequence numbers; default 0 lets pre-assignment construction work."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.sequence_number == 0

    def test_hook_event__metadata_defaults_to_empty_dict(self):
        """metadata defaults to {} — DB column accepts empty JSON safely."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.metadata == {}

    def test_hook_event__parent_session_id_defaults_to_none(self):
        """Top-level sessions have no parent; subagent sessions populate this."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.parent_session_id is None

    def test_hook_event__transcript_location_defaults_to_none(self):
        """transcript_location is harness-specific; absent for OpenCode."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.OPENCODE,
            working_directory="/repo",
        )
        assert event.transcript_location is None

    def test_hook_event__tool_call_defaults_to_none(self):
        """Tool calls are nullable — Notification and Stop events have no tool_call."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.tool_call is None

    def test_hook_event__project_defaults_to_none(self):
        """Project attribution is optional — fire-and-forget hooks may lack it."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.project is None

    def test_hook_event__git_state_defaults_to_none(self):
        """Git state is captured selectively (first event + turn complete), not always."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source=AbstractEventSource.CLAUDE_CODE,
            working_directory="/repo",
        )
        assert event.git_state is None

    def test_hook_event__accepts_full_event_with_tool_call_and_project(self):
        """A complete tool-call event with all enrichments validates cleanly."""
        event = AbstractHookEvent(
            session_id="s1",
            parent_session_id=None,
            event_type=AbstractEventType.TOOL_CALL_COMPLETED,
            source=AbstractEventSource.CLAUDE_CODE,
            timestamp=datetime(2025, 1, 1, 12, 0),
            tool_call=ToolCallPayload(
                tool_name="Read",
                tool_type="Read",
                input={"file_path": "/x.py"},
                output={"success": True},
                duration_ms=42,
            ),
            metadata={"transcript_path": "/tmp/t.jsonl", "tool_response": {"success": True}},
            working_directory="/repo",
            project=Project(name="my-project", source=ProjectSource.GIT),
            transcript_location="/tmp/t.jsonl",
            sequence_number=5,
        )
        assert event.parent_session_id is None
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "Read"
        assert event.tool_call.duration_ms == 42
        assert event.project is not None
        assert event.project.name == "my-project"
        assert event.transcript_location == "/tmp/t.jsonl"
        assert event.sequence_number == 5

    def test_hook_event__rejects_unknown_field(self):
        """extra='forbid' on AbstractHookEvent — wire-format drift surfaces immediately."""
        with pytest.raises(ValidationError):
            AbstractHookEvent(
                session_id="s1",
                event_type=AbstractEventType.NOTIFICATION,
                source=AbstractEventSource.CLAUDE_CODE,
                working_directory="/repo",
                legacy_field="ignored",  # pyright: ignore[reportCallIssue]
            )

    def test_hook_event__source_accepts_open_third_party_strings(self):
        """Source is an open string — third-party collectors don't need an enum extension."""
        event = AbstractHookEvent(
            session_id="s1",
            event_type=AbstractEventType.NOTIFICATION,
            source="some_other_agent",
            working_directory="/repo",
        )

        assert event.source == "some_other_agent"

    def test_hook_event__event_type_accepts_only_known_enum_values(self):
        """AbstractEventType is a closed enum — wire-format drift fails validation."""
        with pytest.raises(ValidationError):
            AbstractHookEvent(
                session_id="s1",
                event_type="custom_event",  # type: ignore[arg-type]
                source=AbstractEventSource.CLAUDE_CODE,
                working_directory="/repo",
            )

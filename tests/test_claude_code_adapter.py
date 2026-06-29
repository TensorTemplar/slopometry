"""Tests for ClaudeCodeAdapter — translates Claude Code hook payloads into AbstractHookEvent.

The adapter owns:
  - the field-name mapping (tool_name, tool_input, tool_response, etc.)
  - tool_type classification (PascalCase enum -> category string)
  - event-type inference from payload shape (when stop_hook_active is present)
  - transcript_location extraction (the harness-specific transcript path hint)

Migrated from `test_hook_handler.py` (TestEventTypeDetection, parse_hook_input)
and `test_posttooluse_validation.py` and `test_notebookread_integration.py` —
those tests exercised the now-removed PreToolUseInput/PostToolUseInput wire
models, which the adapter handles via `extra="allow"` semantics internally.
"""

import pytest

from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
)
from slopometry.core.protocol.adapters.claude_code import (
    ClaudeCodeAdapter,
    ToolType,
    resolve_tool_type,
)


class TestDetectEventType:
    """Tests for ClaudeCodeAdapter.detect_event_type — wire-shape inference."""

    def test_detect_event_type__pre_tool_use_returns_tool_call_started(self):
        """PreToolUse payload: tool_name + tool_input -> TOOL_CALL_STARTED."""
        adapter = ClaudeCodeAdapter()
        payload = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
        assert adapter.detect_event_type(payload) == AbstractEventType.TOOL_CALL_STARTED

    def test_detect_event_type__post_tool_use_returns_tool_call_completed(self):
        """PostToolUse payload: tool_name + tool_input + tool_response -> TOOL_CALL_COMPLETED."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1\nfile2",
        }
        assert adapter.detect_event_type(payload) == AbstractEventType.TOOL_CALL_COMPLETED

    def test_detect_event_type__notification_returns_notification(self):
        """Notification payload: 'message' field present -> NOTIFICATION."""
        adapter = ClaudeCodeAdapter()
        payload = {"message": "Test notification"}
        assert adapter.detect_event_type(payload) == AbstractEventType.NOTIFICATION

    def test_detect_event_type__stop_with_stop_hook_active_false_returns_turn_completed(self):
        """Stop payload: stop_hook_active=False -> TURN_COMPLETED (top-level turn ended)."""
        adapter = ClaudeCodeAdapter()
        payload = {"stop_hook_active": False}
        assert adapter.detect_event_type(payload) == AbstractEventType.TURN_COMPLETED

    def test_detect_event_type__stop_with_stop_hook_active_true_returns_subagent_completed(self):
        """Stop payload: stop_hook_active=True -> SUBAGENT_COMPLETED (subagent turn ended)."""
        adapter = ClaudeCodeAdapter()
        payload = {"stop_hook_active": True}
        assert adapter.detect_event_type(payload) == AbstractEventType.SUBAGENT_COMPLETED

    def test_detect_event_type__stop_without_stop_hook_active_returns_turn_completed(self):
        """Legacy Stop payload (session_id+transcript_path, no stop_hook_active) -> TURN_COMPLETED.

        Older Claude Code versions omit stop_hook_active; the adapter falls back to
        TURN_COMPLETED since session_id+transcript_path alone is the Stop signature.
        """
        adapter = ClaudeCodeAdapter()
        payload = {"session_id": "s1", "transcript_path": "/tmp/t.jsonl"}
        assert adapter.detect_event_type(payload) == AbstractEventType.TURN_COMPLETED

    def test_detect_event_type__raises_for_unknown_shape(self):
        """An unrecognized payload shape must fail loud, not silently default."""
        adapter = ClaudeCodeAdapter()
        with pytest.raises(ValueError, match="Unknown Claude Code hook payload shape"):
            adapter.detect_event_type({"bogus_field": "x"})


class TestParseToolCallStarted:
    """Tests for parse() with TOOL_CALL_STARTED payloads (PreToolUse)."""

    def test_parse_pre_tool_use__returns_event_with_tool_call_and_no_output(self):
        """PreToolUse has no output yet — tool_call.output must be None."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert isinstance(event, AbstractHookEvent)
        assert event.session_id == "s1"
        assert event.event_type == AbstractEventType.TOOL_CALL_STARTED
        assert event.source == AbstractEventSource.CLAUDE_CODE
        assert event.working_directory == "/repo"
        assert event.transcript_location == "/tmp/t.jsonl"
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "Bash"
        assert event.tool_call.tool_type == ToolType.BASH.value
        assert event.tool_call.input == {"command": "ls"}
        assert event.tool_call.output is None
        assert event.tool_call.duration_ms is None
        assert event.tool_call.exit_code is None
        assert event.tool_call.error_message is None

    def test_parse_pre_tool_use__metadata_contains_full_raw_payload(self):
        """metadata preserves the raw payload for forensic / re-processing."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Read",
            "tool_input": {"file_path": "/x.py"},
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.metadata == dict(payload)

    def test_parse_pre_tool_use__raises_when_session_id_missing(self):
        """session_id is required on every event — adapter enforces it."""
        adapter = ClaudeCodeAdapter()
        payload = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
        with pytest.raises(ValueError, match="missing required 'session_id'"):
            adapter.parse(payload, working_directory="/repo")

    def test_parse_pre_tool_use__raises_when_tool_name_missing(self):
        """A TOOL_CALL_STARTED payload must include a non-empty tool_name — adapter enforces it."""
        adapter = ClaudeCodeAdapter()
        payload = {"session_id": "s1", "tool_name": None, "tool_input": {"command": "ls"}}
        with pytest.raises(ValueError, match="missing 'tool_name'"):
            adapter.parse(payload, working_directory="/repo")


class TestParseToolCallCompleted:
    """Tests for parse() with TOOL_CALL_COMPLETED payloads (PostToolUse)."""

    def test_parse_post_tool_use__dict_response_preserves_shape(self):
        """PostToolUse output for Read/Edit is a dict — preserved verbatim."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Read",
            "tool_input": {"file_path": "/x.py"},
            "tool_response": {"success": True, "content": "file content"},
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TOOL_CALL_COMPLETED
        assert event.tool_call is not None
        assert event.tool_call.output == {"success": True, "content": "file content"}

    def test_parse_post_tool_use__str_response_preserved(self):
        """PostToolUse output for Bash is a stdout string — preserved verbatim."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt\n",
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == "file1.txt\nfile2.txt\n"

    def test_parse_post_tool_use__list_response_preserves_cells(self):
        """PostToolUse output for NotebookRead is a list of cells — preserved verbatim."""
        adapter = ClaudeCodeAdapter()
        cells = [
            {"cellType": "markdown", "id": "c1", "source": "# Test"},
            {"cellType": "code", "id": "c2", "source": "print('hello')", "language": "python", "outputs": []},
        ]
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/x.ipynb"},
            "tool_response": cells,
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == cells
        assert isinstance(event.tool_call.output, list)
        assert event.tool_call.output[0]["cellType"] == "markdown"
        assert event.tool_call.output[1]["cellType"] == "code"

    def test_parse_post_tool_use__empty_list_response_preserved(self):
        """Empty NotebookRead (no cells) must not crash."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/empty.ipynb"},
            "tool_response": [],
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.tool_call is not None
        assert event.tool_call.output == []

    def test_parse_post_tool_use__dict_response_extracts_duration_exit_code_error(self):
        """Dict tool_response carries Bash-style metadata in three sibling fields."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": {
                "interrupted": False,
                "duration_ms": 123,
                "exit_code": 0,
            },
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.tool_call is not None
        assert event.tool_call.duration_ms == 123
        assert event.tool_call.exit_code == 0
        assert event.tool_call.error_message is None

    def test_parse_post_tool_use__dict_response_extracts_error_message(self):
        """Error field on tool_response maps to error_message."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
            "tool_response": {
                "interrupted": False,
                "duration_ms": 5,
                "exit_code": 1,
                "error": "command failed",
            },
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.tool_call is not None
        assert event.tool_call.exit_code == 1
        assert event.tool_call.error_message == "command failed"

    def test_parse_post_tool_use__str_response_has_no_duration_or_exit_code(self):
        """Str tool_response (older Claude Code format) carries no duration/exit_code."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt",
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.tool_call is not None
        assert event.tool_call.duration_ms is None
        assert event.tool_call.exit_code is None
        assert event.tool_call.error_message is None


class TestParseNotification:
    """Tests for parse() with NOTIFICATION payloads."""

    def test_parse_notification__returns_event_with_no_tool_call(self):
        """Notifications have no tool_call — only metadata carries the message."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "message": "Test notification",
            "title": "Test Title",
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.NOTIFICATION
        assert event.tool_call is None
        assert event.metadata == dict(payload)


class TestParseStopEvents:
    """Tests for parse() with Stop and SubagentStop payloads."""

    def test_parse_stop_with_stop_hook_active_false__returns_turn_completed(self):
        """Top-level Stop (stop_hook_active=False) -> TURN_COMPLETED, no tool_call."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/t.jsonl",
            "stop_hook_active": False,
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TURN_COMPLETED
        assert event.tool_call is None

    def test_parse_stop_with_stop_hook_active_true__returns_subagent_completed(self):
        """Subagent Stop (stop_hook_active=True) -> SUBAGENT_COMPLETED, no tool_call."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "child-1",
            "transcript_path": "/tmp/t.jsonl",
            "stop_hook_active": True,
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.SUBAGENT_COMPLETED
        assert event.tool_call is None

    def test_parse_stop_without_stop_hook_active__returns_turn_completed(self):
        """Legacy Stop shape (session_id+transcript_path, no stop_hook_active) -> TURN_COMPLETED."""
        adapter = ClaudeCodeAdapter()
        payload = {"session_id": "s1", "transcript_path": "/tmp/t.jsonl"}
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TURN_COMPLETED
        assert event.tool_call is None


class TestParseTranscriptLocationExtraction:
    """Tests for transcript_location extraction from the wire payload."""

    def test_parse__transcript_location_is_transcript_path_field(self):
        """transcript_path maps to transcript_location — same value, canonical name."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "transcript_path": "/tmp/transcripts/s1.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.transcript_location == "/tmp/transcripts/s1.jsonl"

    def test_parse__transcript_location_is_none_when_transcript_path_omitted(self):
        """transcript_location is optional — None when not present in the payload."""
        adapter = ClaudeCodeAdapter()
        payload = {
            "session_id": "s1",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }
        event = adapter.parse(payload, working_directory="/repo")
        assert event.transcript_location is None


class TestParseEventTypeOverride:
    """Tests for event_type_override — bypasses detect_event_type."""

    def test_parse_event_type_override__skips_shape_detection(self):
        """override forces a specific event type even if shape suggests otherwise."""
        adapter = ClaudeCodeAdapter()
        payload = {"session_id": "s1"}  # bare session_id would be ambiguous
        event = adapter.parse(
            payload,
            working_directory="/repo",
            event_type_override=AbstractEventType.NOTIFICATION,
        )
        assert event.event_type == AbstractEventType.NOTIFICATION


class TestToolNameMapping:
    """Tests for the Claude-Code tool_name -> tool_type category mapping."""

    def test_resolve_tool_type__maps_known_pascal_case(self):
        """Known names map to their PascalCase category."""
        assert resolve_tool_type("Bash") == "Bash"
        assert resolve_tool_type("Read") == "Read"
        assert resolve_tool_type("Write") == "Write"
        assert resolve_tool_type("Edit") == "Edit"

    def test_resolve_tool_type__case_insensitive(self):
        """OpenCode sends lowercase names — resolution is case-insensitive."""
        assert resolve_tool_type("bash") == "Bash"
        assert resolve_tool_type("read") == "Read"
        assert resolve_tool_type("edit") == "Edit"

    def test_resolve_tool_type__unknown_lowercase_returns_other(self):
        """Unknown lowercase names fall back to 'Other'."""
        assert resolve_tool_type("SomeFutureTool") == ToolType.OTHER.value

    def test_resolve_tool_type__unknown_mcp_prefix_returns_mcp_other(self):
        """Unknown mcp__-prefixed names fall back to 'mcp__other' (MCP namespace)."""
        assert resolve_tool_type("mcp__unknown__thing") == ToolType.MCP_OTHER.value

    def test_resolve_tool_type__maps_mcp_ide_names(self):
        """Known MCP IDE tool names map to their specific categories."""
        assert resolve_tool_type("mcp__ide__getDiagnostics") == "mcp__ide__getDiagnostics"
        assert resolve_tool_type("mcp__ide__executeCode") == "mcp__ide__executeCode"

    def test_parse__tool_type_is_populated_from_tool_name(self):
        """End-to-end: tool_name='Bash' yields tool_type='Bash' on the event."""
        adapter = ClaudeCodeAdapter()
        event = adapter.parse(
            {
                "session_id": "s1",
                "tool_name": "Bash",
                "tool_input": {"command": "ls"},
            },
            working_directory="/repo",
        )
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "Bash"


class TestAdapterSource:
    """Tests for the adapter's source identity."""

    def test_adapter_source__is_claude_code(self):
        """Every ClaudeCodeAdapter instance advertises source=claude_code."""
        assert ClaudeCodeAdapter().source == AbstractEventSource.CLAUDE_CODE

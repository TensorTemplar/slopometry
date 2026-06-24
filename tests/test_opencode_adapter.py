"""Tests for OpenCodeAdapter — translates OpenCode plugin JSON into AbstractHookEvent.

The adapter owns:
  - the field-name mapping (tool -> tool_name, args -> input, output -> output)
  - event-type resolution from the CLI discriminator string (pre_tool_use, etc.)
  - parent_id -> parent_session_id mapping for subagent sessions
  - duration_ms extraction from a top-level field (Claude Code nests it in tool_response)

Migrated from `test_opencode_handler.py` (TestEventTypeMap, TestParseOpenCodeEvent,
TestGetToolType, TestGetSessionId, TestGetParentId, TestHandleOpenCodeStop).
"""

import pytest

from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
)
from slopometry.core.protocol.adapters.opencode import (
    OpenCodeAdapter,
    resolve_opencode_event_type,
)


class TestResolveOpenCodeEventType:
    """Tests for the CLI event-type discriminator -> canonical enum mapping."""

    def test_resolve_opencode_event_type__pre_tool_use(self):
        assert resolve_opencode_event_type("pre_tool_use") == AbstractEventType.TOOL_CALL_STARTED

    def test_resolve_opencode_event_type__post_tool_use(self):
        assert resolve_opencode_event_type("post_tool_use") == AbstractEventType.TOOL_CALL_COMPLETED

    def test_resolve_opencode_event_type__stop(self):
        assert resolve_opencode_event_type("stop") == AbstractEventType.TURN_COMPLETED

    def test_resolve_opencode_event_type__subagent_stop(self):
        assert resolve_opencode_event_type("subagent_stop") == AbstractEventType.SUBAGENT_COMPLETED

    def test_resolve_opencode_event_type__subagent_start(self):
        assert resolve_opencode_event_type("subagent_start") == AbstractEventType.SUBAGENT_STARTED

    def test_resolve_opencode_event_type__todo_updated(self):
        assert resolve_opencode_event_type("todo_updated") == AbstractEventType.TODO_UPDATED

    def test_resolve_opencode_event_type__message_updated(self):
        assert resolve_opencode_event_type("message_updated") == AbstractEventType.MESSAGE_UPDATED

    def test_resolve_opencode_event_type__covers_all_event_types(self):
        """All OpenCode event types must map to a canonical enum."""
        expected = {
            "pre_tool_use",
            "post_tool_use",
            "stop",
            "subagent_stop",
            "subagent_start",
            "todo_updated",
            "message_updated",
        }
        for et in expected:
            result = resolve_opencode_event_type(et)
            assert isinstance(result, AbstractEventType)

    def test_resolve_opencode_event_type__unknown_raises(self):
        """Unknown CLI event type fails loud — the dispatcher must not silently default."""
        with pytest.raises(ValueError, match="Unknown OpenCode event type"):
            resolve_opencode_event_type("totally_unknown")


class TestDetectEventType:
    """Tests for OpenCodeAdapter.detect_event_type — discriminator-based."""

    def test_detect_event_type__requires_event_type_field(self):
        """OpenCode requires an explicit event_type discriminator (unlike Claude Code)."""
        adapter = OpenCodeAdapter()
        with pytest.raises(ValueError, match="missing string 'event_type' field"):
            adapter.detect_event_type({"session_id": "s1"})

    def test_detect_event_type__rejects_non_string_event_type(self):
        """A non-string event_type (e.g., accidental bool/int) must fail validation."""
        adapter = OpenCodeAdapter()
        with pytest.raises(ValueError, match="missing string 'event_type' field"):
            adapter.detect_event_type({"event_type": 123})

    def test_detect_event_type__delegates_to_resolve(self):
        """Adapter delegates string-to-enum resolution to the module function."""
        adapter = OpenCodeAdapter()
        assert adapter.detect_event_type({"event_type": "pre_tool_use"}) == AbstractEventType.TOOL_CALL_STARTED


class TestParseToolCallEvents:
    """Tests for parse() with pre_tool_use and post_tool_use payloads."""

    def test_parse_pre_tool_use__returns_tool_call_started_with_args(self):
        """pre_tool_use: 'tool' -> tool_name, 'args' -> input, no output yet."""
        adapter = OpenCodeAdapter()
        payload = {"event_type": "pre_tool_use", "tool": "Bash", "session_id": "s1", "args": {"command": "ls"}}
        event = adapter.parse(payload, working_directory="/repo")

        assert isinstance(event, AbstractHookEvent)
        assert event.event_type == AbstractEventType.TOOL_CALL_STARTED
        assert event.source == AbstractEventSource.OPENCODE
        assert event.session_id == "s1"
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "Bash"
        assert event.tool_call.input == {"command": "ls"}
        assert event.tool_call.output is None

    def test_parse_post_tool_use__returns_tool_call_completed_with_output(self):
        """post_tool_use: 'tool' -> tool_name, 'args' -> input, 'output' -> output."""
        adapter = OpenCodeAdapter()
        payload = {
            "event_type": "post_tool_use",
            "tool": "Read",
            "session_id": "s1",
            "args": {"file_path": "/tmp/f.py"},
            "output": "file content",
            "duration_ms": 42,
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TOOL_CALL_COMPLETED
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "Read"
        assert event.tool_call.output == "file content"
        assert event.tool_call.duration_ms == 42
        assert event.tool_call.exit_code is None
        assert event.tool_call.error_message is None

    def test_parse_pre_tool_use__lowercase_tool_name_normalized(self):
        """OpenCode sends lowercase tool names — adapter normalizes via resolve_tool_type."""
        adapter = OpenCodeAdapter()
        payload = {"event_type": "pre_tool_use", "tool": "bash", "session_id": "s1", "args": {"command": "ls"}}
        event = adapter.parse(payload, working_directory="/repo")
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "bash"
        assert event.tool_call.tool_type == "Bash"


class TestParseSubagentEvents:
    """Tests for parse() with subagent_start and subagent_stop payloads."""

    def test_parse_subagent_start__parent_id_maps_to_parent_session_id(self):
        """OpenCode 'parent_id' -> canonical 'parent_session_id'."""
        adapter = OpenCodeAdapter()
        payload = {
            "event_type": "subagent_start",
            "session_id": "child-1",
            "parent_id": "parent-1",
            "agent": "explore",
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.SUBAGENT_STARTED
        assert event.session_id == "child-1"
        assert event.parent_session_id == "parent-1"
        assert event.tool_call is None

    def test_parse_subagent_start__no_parent_id_yields_none(self):
        """Top-level sessions have no parent — parent_session_id is None."""
        adapter = OpenCodeAdapter()
        payload = {"event_type": "subagent_start", "session_id": "main", "agent": "general"}
        event = adapter.parse(payload, working_directory="/repo")
        assert event.parent_session_id is None

    def test_parse_subagent_stop__parent_id_maps_to_parent_session_id(self):
        """subagent_stop: parent_id -> parent_session_id, no tool_call."""
        adapter = OpenCodeAdapter()
        payload = {
            "event_type": "subagent_stop",
            "session_id": "child-1",
            "parent_id": "parent-1",
            "agent": "explore",
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.SUBAGENT_COMPLETED
        assert event.parent_session_id == "parent-1"
        assert event.tool_call is None


class TestParseTodoAndMessageEvents:
    """Tests for parse() with todo_updated and message_updated payloads."""

    def test_parse_todo_updated__no_tool_call(self):
        """todo_updated events have no tool_call — todos live in metadata."""
        adapter = OpenCodeAdapter()
        payload = {
            "event_type": "todo_updated",
            "session_id": "s1",
            "todos": [{"content": "Fix bug", "status": "pending", "priority": "high"}],
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TODO_UPDATED
        assert event.tool_call is None
        assert event.metadata == dict(payload)

    def test_parse_message_updated__no_tool_call(self):
        """message_updated events have no tool_call — message metadata only."""
        adapter = OpenCodeAdapter()
        payload = {
            "event_type": "message_updated",
            "session_id": "s1",
            "message_id": "m1",
            "model_id": "claude-3-opus",
            "agent": "general",
        }
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.MESSAGE_UPDATED
        assert event.tool_call is None


class TestParseStopEvent:
    """Tests for parse() with stop payloads."""

    def test_parse_stop__no_tool_call(self):
        """stop events have no tool_call — they mark the end of a turn."""
        adapter = OpenCodeAdapter()
        payload = {"event_type": "stop", "session_id": "s1", "agent": "general"}
        event = adapter.parse(payload, working_directory="/repo")

        assert event.event_type == AbstractEventType.TURN_COMPLETED
        assert event.tool_call is None


class TestParseFieldMapping:
    """Tests for the OpenCode-specific field-name remapping."""

    def test_parse__tool_maps_to_tool_name(self):
        """OpenCode 'tool' field -> canonical 'tool_name'."""
        adapter = OpenCodeAdapter()
        event = adapter.parse(
            {"event_type": "pre_tool_use", "tool": "Edit", "session_id": "s1", "args": {"file_path": "/x.py"}},
            working_directory="/repo",
        )
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "Edit"

    def test_parse__args_maps_to_input(self):
        """OpenCode 'args' field -> canonical 'input'."""
        adapter = OpenCodeAdapter()
        event = adapter.parse(
            {"event_type": "pre_tool_use", "tool": "Bash", "session_id": "s1", "args": {"command": "ls"}},
            working_directory="/repo",
        )
        assert event.tool_call is not None
        assert event.tool_call.input == {"command": "ls"}

    def test_parse__output_maps_to_output(self):
        """OpenCode 'output' field -> canonical 'output'."""
        adapter = OpenCodeAdapter()
        event = adapter.parse(
            {
                "event_type": "post_tool_use",
                "tool": "Read",
                "session_id": "s1",
                "args": {"file_path": "/x"},
                "output": "file content",
            },
            working_directory="/repo",
        )
        assert event.tool_call is not None
        assert event.tool_call.output == "file content"

    def test_parse__duration_ms_is_top_level(self):
        """OpenCode carries duration_ms at the top level (Claude Code nests it in tool_response)."""
        adapter = OpenCodeAdapter()
        event = adapter.parse(
            {
                "event_type": "post_tool_use",
                "tool": "Read",
                "session_id": "s1",
                "args": {},
                "output": "x",
                "duration_ms": 99,
            },
            working_directory="/repo",
        )
        assert event.tool_call is not None
        assert event.tool_call.duration_ms == 99

    def test_parse__transcript_location_is_none(self):
        """OpenCode stores transcripts in metadata.transcript, not as a path — transcript_location stays None."""
        adapter = OpenCodeAdapter()
        event = adapter.parse(
            {"event_type": "stop", "session_id": "s1"},
            working_directory="/repo",
        )
        assert event.transcript_location is None

    def test_parse__raises_when_session_id_missing(self):
        """session_id is required on every event — adapter enforces it."""
        adapter = OpenCodeAdapter()
        with pytest.raises(ValueError, match="missing required 'session_id'"):
            adapter.parse({"event_type": "stop"}, working_directory="/repo")

    def test_parse__raises_when_tool_missing_for_tool_event(self):
        """tool_call events must include 'tool' field — adapter enforces it."""
        adapter = OpenCodeAdapter()
        with pytest.raises(ValueError, match="missing 'tool' field"):
            adapter.parse(
                {"event_type": "pre_tool_use", "session_id": "s1", "args": {}},
                working_directory="/repo",
            )

    def test_parse__metadata_contains_full_raw_payload(self):
        """metadata preserves the raw OpenCode payload for forensic / re-processing."""
        adapter = OpenCodeAdapter()
        payload = {"event_type": "stop", "session_id": "s1", "agent": "general", "transcript": [{"role": "user"}]}
        event = adapter.parse(payload, working_directory="/repo")
        assert event.metadata == dict(payload)


class TestAdapterSource:
    """Tests for the adapter's source identity."""

    def test_adapter_source__is_opencode(self):
        """Every OpenCodeAdapter instance advertises source=opencode."""
        assert OpenCodeAdapter().source == AbstractEventSource.OPENCODE

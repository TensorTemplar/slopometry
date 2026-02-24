"""Tests for OpenCode hook handler functionality."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from slopometry.core.database import SessionManager
from slopometry.core.models.hook import HookEventType, ToolType
from slopometry.core.models.opencode import (
    OpenCodeMessageEvent,
    OpenCodeSessionEvent,
    OpenCodeStopEvent,
    OpenCodeTodoEvent,
    OpenCodeToolEvent,
)
from slopometry.core.opencode_handler import (
    EVENT_TYPE_MAP,
    _get_parent_id,
    _get_session_id,
    _handle_opencode_stop,
    get_tool_type,
    handle_opencode_hook,
    parse_opencode_event,
)


class TestEventTypeMap:
    def test_event_type_map__pre_tool_use_maps_correctly(self):
        assert EVENT_TYPE_MAP["pre_tool_use"] == HookEventType.PRE_TOOL_USE

    def test_event_type_map__post_tool_use_maps_correctly(self):
        assert EVENT_TYPE_MAP["post_tool_use"] == HookEventType.POST_TOOL_USE

    def test_event_type_map__stop_maps_correctly(self):
        assert EVENT_TYPE_MAP["stop"] == HookEventType.STOP

    def test_event_type_map__subagent_stop_maps_correctly(self):
        assert EVENT_TYPE_MAP["subagent_stop"] == HookEventType.SUBAGENT_STOP

    def test_event_type_map__subagent_start_maps_correctly(self):
        assert EVENT_TYPE_MAP["subagent_start"] == HookEventType.SUBAGENT_START

    def test_event_type_map__todo_updated_maps_correctly(self):
        assert EVENT_TYPE_MAP["todo_updated"] == HookEventType.TODO_UPDATED

    def test_event_type_map__message_updated_maps_correctly(self):
        assert EVENT_TYPE_MAP["message_updated"] == HookEventType.MESSAGE_UPDATED

    def test_event_type_map__covers_all_opencode_event_types(self):
        expected_keys = {
            "pre_tool_use",
            "post_tool_use",
            "stop",
            "subagent_stop",
            "subagent_start",
            "todo_updated",
            "message_updated",
        }
        assert set(EVENT_TYPE_MAP.keys()) == expected_keys


class TestParseOpenCodeEvent:
    def test_parse_opencode_event__pre_tool_use_returns_tool_event(self):
        raw = {"tool": "Bash", "session_id": "s1", "call_id": "c1", "args": {"command": "ls"}}
        result = parse_opencode_event("pre_tool_use", raw)
        assert isinstance(result, OpenCodeToolEvent)
        assert result.tool == "Bash"
        assert result.session_id == "s1"

    def test_parse_opencode_event__post_tool_use_returns_tool_event_with_output(self):
        raw = {
            "tool": "Read",
            "session_id": "s1",
            "call_id": "c2",
            "args": {"file_path": "/tmp/f.py"},
            "output": "file content",
            "duration_ms": 42,
        }
        result = parse_opencode_event("post_tool_use", raw)
        assert isinstance(result, OpenCodeToolEvent)
        assert result.output == "file content"
        assert result.duration_ms == 42

    def test_parse_opencode_event__todo_updated_returns_todo_event(self):
        raw = {
            "session_id": "s1",
            "todos": [{"content": "Fix bug", "status": "pending", "priority": "high"}],
        }
        result = parse_opencode_event("todo_updated", raw)
        assert isinstance(result, OpenCodeTodoEvent)
        assert len(result.todos) == 1
        assert result.todos[0].content == "Fix bug"

    def test_parse_opencode_event__message_updated_returns_message_event(self):
        raw = {
            "session_id": "s1",
            "message_id": "m1",
            "model_id": "claude-3-opus",
            "agent": "general",
            "tokens": {"input": 100, "output": 50, "reasoning": 0, "cache_read": 0, "cache_write": 0},
            "cost": 0.01,
        }
        result = parse_opencode_event("message_updated", raw)
        assert isinstance(result, OpenCodeMessageEvent)
        assert result.agent == "general"
        assert result.tokens.input == 100

    def test_parse_opencode_event__subagent_start_returns_session_event(self):
        raw = {"session_id": "child-1", "parent_id": "parent-1", "agent": "explore"}
        result = parse_opencode_event("subagent_start", raw)
        assert isinstance(result, OpenCodeSessionEvent)
        assert result.parent_id == "parent-1"

    def test_parse_opencode_event__stop_returns_stop_event(self):
        raw = {"session_id": "s1", "agent": "general", "model_id": "claude-3-opus"}
        result = parse_opencode_event("stop", raw)
        assert isinstance(result, OpenCodeStopEvent)

    def test_parse_opencode_event__subagent_stop_returns_stop_event(self):
        raw = {"session_id": "child-1", "parent_id": "parent-1", "agent": "explore"}
        result = parse_opencode_event("subagent_stop", raw)
        assert isinstance(result, OpenCodeStopEvent)
        assert result.parent_id == "parent-1"

    def test_parse_opencode_event__unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown OpenCode event type"):
            parse_opencode_event("invalid_type", {"session_id": "s1"})


class TestGetToolType:
    def test_get_tool_type__maps_known_tool(self):
        assert get_tool_type("Bash") == ToolType.BASH

    def test_get_tool_type__maps_read_tool(self):
        assert get_tool_type("Read") == ToolType.READ

    def test_get_tool_type__maps_edit_tool(self):
        assert get_tool_type("Edit") == ToolType.EDIT

    def test_get_tool_type__unknown_tool_returns_other(self):
        assert get_tool_type("SomeFutureTool") == ToolType.OTHER


class TestGetSessionId:
    def test_get_session_id__from_tool_event(self):
        event = OpenCodeToolEvent(tool="Bash", session_id="s1", call_id="c1")
        assert _get_session_id(event) == "s1"

    def test_get_session_id__from_stop_event(self):
        event = OpenCodeStopEvent(session_id="s2")
        assert _get_session_id(event) == "s2"

    def test_get_session_id__from_todo_event(self):
        event = OpenCodeTodoEvent(session_id="s3")
        assert _get_session_id(event) == "s3"

    def test_get_session_id__from_message_event(self):
        event = OpenCodeMessageEvent(session_id="s4", message_id="m1")
        assert _get_session_id(event) == "s4"

    def test_get_session_id__from_session_event(self):
        event = OpenCodeSessionEvent(session_id="s5")
        assert _get_session_id(event) == "s5"


class TestGetParentId:
    def test_get_parent_id__from_session_event_with_parent(self):
        event = OpenCodeSessionEvent(session_id="child", parent_id="parent")
        assert _get_parent_id(event) == "parent"

    def test_get_parent_id__from_session_event_without_parent(self):
        event = OpenCodeSessionEvent(session_id="main")
        assert _get_parent_id(event) is None

    def test_get_parent_id__from_stop_event_with_parent(self):
        event = OpenCodeStopEvent(session_id="child", parent_id="parent")
        assert _get_parent_id(event) == "parent"

    def test_get_parent_id__from_tool_event_returns_none(self):
        event = OpenCodeToolEvent(tool="Bash", session_id="s1", call_id="c1")
        assert _get_parent_id(event) is None

    def test_get_parent_id__from_todo_event_returns_none(self):
        event = OpenCodeTodoEvent(session_id="s1")
        assert _get_parent_id(event) is None

    def test_get_parent_id__from_message_event_returns_none(self):
        event = OpenCodeMessageEvent(session_id="s1", message_id="m1")
        assert _get_parent_id(event) is None


class TestHandleOpenCodeStop:
    def test_handle_opencode_stop__non_stop_event_returns_zero(self):
        """Non-OpenCodeStopEvent types should return 0 immediately."""
        tool_event = OpenCodeToolEvent(tool="Bash", session_id="s1", call_id="c1")
        result = _handle_opencode_stop("s1", tool_event, "stop")
        assert result == 0

    def test_handle_opencode_stop__stop_event_calls_handle_stop_event(self):
        """Stop events should delegate to the shared handle_stop_event pipeline."""
        stop_event = OpenCodeStopEvent(session_id="s1", agent="general")

        with patch("slopometry.core.hook_handler.handle_stop_event", return_value=0) as mock_handle:
            result = _handle_opencode_stop("s1", stop_event, "stop")

        assert result == 0
        mock_handle.assert_called_once()
        call_args = mock_handle.call_args
        assert call_args[0][0] == "s1"
        stop_input = call_args[0][1]
        assert stop_input.session_id == "s1"
        assert stop_input.stop_hook_active is False

    def test_handle_opencode_stop__subagent_stop_also_calls_handle_stop_event(self):
        """Subagent stop events should also go through the feedback pipeline."""
        stop_event = OpenCodeStopEvent(session_id="child-1", parent_id="parent-1", agent="explore")

        with patch("slopometry.core.hook_handler.handle_stop_event", return_value=2) as mock_handle:
            result = _handle_opencode_stop("child-1", stop_event, "subagent_stop")

        assert result == 2
        mock_handle.assert_called_once()


class TestHandleOpenCodeHookSmoke:
    """Smoke tests for the full OpenCode hook pipeline.

    These tests patch _read_stdin_with_timeout instead of sys.stdin because
    the real implementation uses select.select() which requires a real file descriptor.
    """

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path):
        """Redirect database and session state to temp directories so smoke tests don't pollute the real DB."""
        db_path = tmp_path / "test.db"
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        original_init = SessionManager.__init__

        def _isolated_init(self_inner):
            original_init(self_inner)
            self_inner.state_dir = state_dir

        with (
            patch("slopometry.core.settings.settings.database_path", db_path),
            patch.object(SessionManager, "__init__", _isolated_init),
        ):
            yield

    def _init_git_repo(self, path: Path) -> None:
        """Initialize a git repo for testing."""
        subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "--local", "user.email", "test@example.com"],
            cwd=path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "--local", "user.name", "Test"],
            cwd=path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "--local", "commit.gpgsign", "false"],
            cwd=path,
            capture_output=True,
            check=True,
        )

    def test_handle_opencode_hook__pre_tool_use_does_not_crash(self):
        """Smoke test: pre_tool_use event should not crash."""

        input_data = {"tool": "Bash", "session_id": "oc-smoke-1", "call_id": "c1", "args": {"command": "ls"}}

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_opencode_hook("pre_tool_use")

        assert result == 0

    def test_handle_opencode_hook__post_tool_use_does_not_crash(self):
        """Smoke test: post_tool_use event should not crash."""

        input_data = {
            "tool": "Read",
            "session_id": "oc-smoke-2",
            "call_id": "c2",
            "args": {"file_path": "/tmp/f.py"},
            "output": "contents",
            "duration_ms": 10,
        }

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_opencode_hook("post_tool_use")

        assert result == 0

    def test_handle_opencode_hook__todo_updated_does_not_crash(self):
        """Smoke test: todo_updated event should not crash."""

        input_data = {
            "session_id": "oc-smoke-3",
            "todos": [{"content": "Fix tests", "status": "pending", "priority": "high"}],
        }

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_opencode_hook("todo_updated")

        assert result == 0

    def test_handle_opencode_hook__message_updated_does_not_crash(self):
        """Smoke test: message_updated event should not crash."""

        input_data = {
            "session_id": "oc-smoke-4",
            "message_id": "m1",
            "model_id": "claude-3-opus",
            "tokens": {"input": 100, "output": 50, "reasoning": 0, "cache_read": 0, "cache_write": 0},
            "cost": 0.01,
        }

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_opencode_hook("message_updated")

        assert result == 0

    def test_handle_opencode_hook__stop_does_not_crash(self):
        """Smoke test: stop event should not crash."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._init_git_repo(tmppath)
            (tmppath / "test.py").write_text("x = 1")
            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            input_data = {"session_id": "oc-smoke-stop", "agent": "general", "model_id": "claude-3-opus"}

            with (
                patch(
                    "slopometry.core.opencode_handler._read_stdin_with_timeout",
                    return_value=json.dumps(input_data),
                ),
                patch("os.getcwd", return_value=str(tmppath)),
            ):
                result = handle_opencode_hook("stop")

            assert result in (0, 2)

    def test_handle_opencode_hook__empty_stdin_returns_zero(self):
        """Empty stdin (timeout) should return 0 without errors."""

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value=""):
            result = handle_opencode_hook("pre_tool_use")

        assert result == 0

    def test_handle_opencode_hook__invalid_json_returns_zero(self):
        """Invalid JSON input should return 0 without crashing."""

        with patch("slopometry.core.opencode_handler._read_stdin_with_timeout", return_value="not valid json"):
            result = handle_opencode_hook("pre_tool_use")

        assert result == 0

    def test_handle_opencode_hook__unknown_event_type_returns_zero(self):
        """Unknown event type should return 0 without crashing."""

        with patch(
            "slopometry.core.opencode_handler._read_stdin_with_timeout",
            return_value=json.dumps({"session_id": "s1"}),
        ):
            result = handle_opencode_hook("totally_unknown")

        assert result == 0

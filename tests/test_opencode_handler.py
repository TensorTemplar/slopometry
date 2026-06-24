"""Smoke tests for the OpenCode hook handler entrypoint.

Pure plumbing tests: verify the entrypoint reads stdin, dispatches through the
OpenCode adapter, and never crashes on malformed or unknown input. Detailed
adapter semantics (parse, detect_event_type, tool-type mapping) live in
`test_opencode_adapter.py`.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from slopometry.core.database import SessionManager


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Redirect database and session state to temp directories so smoke tests don't pollute the real DB."""
    db_path = tmp_path / "test.db"
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    original_init = SessionManager.__init__

    def _isolated_init(self_inner, source: str = "opencode"):
        original_init(self_inner, source=source)
        self_inner.state_dir = state_dir

    with (
        patch("slopometry.core.settings.settings.database_path", db_path),
        patch.object(SessionManager, "__init__", _isolated_init),
    ):
        yield


def _init_git_repo(path: Path) -> None:
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


from slopometry.core.opencode_handler import handle_opencode_hook


class TestHandleOpenCodeHookSmoke:
    """Smoke tests for the full OpenCode hook pipeline.

    These tests patch _read_stdin_with_timeout instead of sys.stdin because
    the real implementation uses select.select() which requires a real file descriptor.
    """

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
            _init_git_repo(tmppath)
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

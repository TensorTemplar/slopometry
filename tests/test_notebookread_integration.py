"""Integration test for NotebookRead hook handling via ClaudeCodeAdapter.

Migrated from `parse_hook_input(raw)` + `PostToolUseInput` validation. The
adapter handles list-typed tool_response internally — these tests verify
end-to-end that a NotebookRead payload (list-shaped tool_response) round-trips
through parse() with the cells preserved.
"""

from slopometry.core.models.protocol.events import AbstractEventType
from slopometry.core.protocol.adapters.claude_code import ClaudeCodeAdapter


class TestNotebookReadIntegration:
    """Test NotebookRead integration with the Claude Code adapter."""

    def test_parse_notebookread_response__preserves_cells_correctly(self):
        """Adapter.parse() preserves the full NotebookRead list response."""
        adapter = ClaudeCodeAdapter()
        raw_hook_data = {
            "session_id": "test_session_123",
            "transcript_path": "/path/to/transcript.jsonl",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/path/to/notebook.ipynb"},
            "tool_response": [
                {"cellType": "markdown", "id": "cell_id_1", "source": "# Test Notebook\n\nThis is a markdown cell."},
                {
                    "cellType": "code",
                    "id": "cell_id_2",
                    "source": "print('Hello from notebook')\nx = 42",
                    "language": "python",
                    "outputs": [],
                },
            ],
        }

        event = adapter.parse(raw_hook_data, working_directory="/repo")

        assert event.session_id == "test_session_123"
        assert event.event_type == AbstractEventType.TOOL_CALL_COMPLETED
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "NotebookRead"
        assert isinstance(event.tool_call.output, list)
        assert len(event.tool_call.output) == 2
        assert event.tool_call.output[0]["cellType"] == "markdown"
        assert event.tool_call.output[1]["cellType"] == "code"
        assert "python" in event.tool_call.output[1]["language"]

    def test_parse_notebookread_empty_response__preserves_empty_list(self):
        """Adapter.parse() handles empty NotebookRead responses (no cells)."""
        adapter = ClaudeCodeAdapter()
        raw_hook_data = {
            "session_id": "test_session_456",
            "transcript_path": "/path/to/transcript.jsonl",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/path/to/empty_notebook.ipynb"},
            "tool_response": [],
        }

        event = adapter.parse(raw_hook_data, working_directory="/repo")

        assert event.session_id == "test_session_456"
        assert event.tool_call is not None
        assert event.tool_call.tool_name == "NotebookRead"
        assert isinstance(event.tool_call.output, list)
        assert len(event.tool_call.output) == 0

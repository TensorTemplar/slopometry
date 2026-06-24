"""Tests for ClaudeCodeAdapter.parse() across the three PostToolUse tool_response shapes.

Migrated from tests of the removed `PostToolUseInput` model. The adapter now
handles wire-format shape variability internally via `extra="allow"` semantics —
these tests verify the parser preserves dict / str / list shapes verbatim and
handles the empty-list edge case for NotebookRead.
"""

from slopometry.core.protocol.adapters.claude_code import ClaudeCodeAdapter


class TestPostToolUseAdapterValidation:
    """Test ClaudeCodeAdapter.parse() for PostToolUse payload shapes."""

    def test_posttooluse_with_dict_response__preserves_dict(self):
        """Read/Edit tool_response is a dict — preserved verbatim."""
        adapter = ClaudeCodeAdapter()
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test/file.py"},
            "tool_response": {"success": True, "content": "file content"},
        }

        event = adapter.parse(data, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == {"success": True, "content": "file content"}

    def test_posttooluse_with_str_response__preserves_str(self):
        """Bash tool_response is a stdout string — preserved verbatim."""
        adapter = ClaudeCodeAdapter()
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt\n",
        }

        event = adapter.parse(data, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == "file1.txt\nfile2.txt\n"

    def test_posttooluse_with_list_response__preserves_cells(self):
        """NotebookRead tool_response is a list of cells — preserved verbatim.

        This was the original bug: the old PostToolUseInput pydantic model
        rejected list-typed tool_response. The adapter now handles any shape.
        """
        adapter = ClaudeCodeAdapter()
        notebook_cells = [
            {
                "cellType": "markdown",
                "id": "cell1",
                "source": "# Test Notebook\nThis is a test.",
            },
            {
                "cellType": "code",
                "id": "cell2",
                "source": "print('hello world')",
                "language": "python",
                "outputs": [],
            },
        ]

        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/test/notebook.ipynb"},
            "tool_response": notebook_cells,
        }

        event = adapter.parse(data, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == notebook_cells
        assert isinstance(event.tool_call.output, list)
        assert len(event.tool_call.output) == 2
        cell0 = event.tool_call.output[0]
        cell1 = event.tool_call.output[1]
        assert isinstance(cell0, dict) and cell0["cellType"] == "markdown"
        assert isinstance(cell1, dict) and cell1["cellType"] == "code"

    def test_posttooluse_with_empty_list_response__preserves_empty_list(self):
        """Empty NotebookRead (no cells) — preserved as empty list, no crash."""
        adapter = ClaudeCodeAdapter()
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/empty/notebook.ipynb"},
            "tool_response": [],
        }

        event = adapter.parse(data, working_directory="/repo")

        assert event.tool_call is not None
        assert event.tool_call.output == []

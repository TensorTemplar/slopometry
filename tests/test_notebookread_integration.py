"""Integration test for NotebookRead hook handling."""

from slopometry.core.hook_handler import parse_hook_input


class TestNotebookReadIntegration:
    """Test NotebookRead integration with hook handler."""

    def test_parse_hook_input_with_notebookread_response__handles_list_correctly(self):
        """Test that parse_hook_input can handle NotebookRead responses with lists."""
        # This simulates the exact data structure that was causing the validation error
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

        # This should not raise a ValidationError anymore
        parsed_input = parse_hook_input(raw_hook_data)

        # Verify the parsing worked correctly
        assert parsed_input.session_id == "test_session_123"
        assert parsed_input.tool_name == "NotebookRead"
        assert isinstance(parsed_input.tool_response, list)
        assert len(parsed_input.tool_response) == 2
        assert parsed_input.tool_response[0]["cellType"] == "markdown"
        assert parsed_input.tool_response[1]["cellType"] == "code"
        assert "python" in parsed_input.tool_response[1]["language"]

    def test_parse_hook_input_with_notebookread_empty_response__handles_empty_list(self):
        """Test that parse_hook_input can handle empty NotebookRead responses."""
        raw_hook_data = {
            "session_id": "test_session_456",
            "transcript_path": "/path/to/transcript.jsonl",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/path/to/empty_notebook.ipynb"},
            "tool_response": [],
        }

        # This should not raise a ValidationError
        parsed_input = parse_hook_input(raw_hook_data)

        # Verify the parsing worked correctly
        assert parsed_input.session_id == "test_session_456"
        assert parsed_input.tool_name == "NotebookRead"
        assert isinstance(parsed_input.tool_response, list)
        assert len(parsed_input.tool_response) == 0

"""Tests for PostToolUseInput validation fix."""

from slopometry.core.models import PostToolUseInput


class TestPostToolUseInputValidation:
    """Test PostToolUseInput model validation."""

    def test_posttooluse_with_dict_response__validates_correctly(self):
        """Test that PostToolUseInput accepts dictionary responses."""
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "Read",
            "tool_input": {"file_path": "/test/file.py"},
            "tool_response": {"success": True, "content": "file content"},
        }

        input_model = PostToolUseInput(**data)
        assert input_model.tool_response == {"success": True, "content": "file content"}

    def test_posttooluse_with_str_response__validates_correctly(self):
        """Test that PostToolUseInput accepts string responses."""
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt\n",
        }

        input_model = PostToolUseInput(**data)
        assert input_model.tool_response == "file1.txt\nfile2.txt\n"

    def test_posttooluse_with_list_response__validates_correctly(self):
        """Test that PostToolUseInput accepts list responses (like NotebookRead)."""
        # This simulates the actual NotebookRead response that was causing the error
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

        input_model = PostToolUseInput(**data)
        assert input_model.tool_response == notebook_cells
        assert len(input_model.tool_response) == 2
        assert input_model.tool_response[0]["cellType"] == "markdown"
        assert input_model.tool_response[1]["cellType"] == "code"

    def test_posttooluse_with_empty_list_response__validates_correctly(self):
        """Test that PostToolUseInput accepts empty list responses."""
        data = {
            "session_id": "test_session",
            "transcript_path": "/path/to/transcript",
            "tool_name": "NotebookRead",
            "tool_input": {"notebook_path": "/empty/notebook.ipynb"},
            "tool_response": [],
        }

        input_model = PostToolUseInput(**data)
        assert input_model.tool_response == []

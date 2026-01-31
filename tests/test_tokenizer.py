"""Tests for tokenizer module."""

from pathlib import Path
from unittest.mock import patch

from slopometry.core.tokenizer import count_file_tokens, count_tokens, get_encoder


class TestGetEncoder:
    """Tests for get_encoder function."""

    def test_get_encoder__returns_encoder(self) -> None:
        """Should return a tiktoken encoder."""
        encoder = get_encoder()
        assert encoder is not None
        # Verify it can encode text (functional test rather than hasattr check)
        tokens = encoder.encode("test")
        assert isinstance(tokens, list)

    def test_get_encoder__caches_encoder(self) -> None:
        """Should return the same encoder instance on subsequent calls."""
        encoder1 = get_encoder()
        encoder2 = get_encoder()
        assert encoder1 is encoder2


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_tokens__counts_simple_text(self) -> None:
        """Should count tokens in simple text."""
        result = count_tokens("hello world")
        assert result > 0

    def test_count_tokens__empty_string_returns_zero(self) -> None:
        """Should return 0 for empty string."""
        result = count_tokens("")
        assert result == 0

    def test_count_tokens__handles_code(self) -> None:
        """Should count tokens in code."""
        code = """
def hello():
    print("world")
"""
        result = count_tokens(code)
        assert result > 0


class TestCountFileTokens:
    """Tests for count_file_tokens function."""

    def test_count_file_tokens__reads_and_counts(self, tmp_path: Path) -> None:
        """Should read file and count tokens."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        result = count_file_tokens(test_file)
        assert result > 0

    def test_count_file_tokens__missing_file_returns_zero(self, tmp_path: Path) -> None:
        """Should return 0 for missing file."""
        missing_file = tmp_path / "missing.py"

        result = count_file_tokens(missing_file)
        assert result == 0

    def test_count_file_tokens__unreadable_file_returns_zero(
        self, tmp_path: Path
    ) -> None:
        """Should return 0 when file cannot be read."""
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = count_file_tokens(test_file)
            assert result == 0

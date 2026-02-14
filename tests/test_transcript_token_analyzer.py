"""Tests for TranscriptTokenAnalyzer using real transcript data."""

from pathlib import Path

import pytest

from slopometry.core.models.session import TokenUsage
from slopometry.core.transcript_token_analyzer import TranscriptTokenAnalyzer, analyze_transcript_tokens


class TestTokenUsageModel:
    """Tests for the TokenUsage model."""

    def test_total_tokens__sums_input_and_output(self):
        """Test total_tokens property."""
        usage = TokenUsage(
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert usage.total_tokens == 1500

    def test_exploration_tokens__sums_exploration_input_and_output(self):
        """Test exploration_tokens property."""
        usage = TokenUsage(
            exploration_input_tokens=800,
            exploration_output_tokens=200,
        )
        assert usage.exploration_tokens == 1000

    def test_implementation_tokens__sums_implementation_input_and_output(self):
        """Test implementation_tokens property."""
        usage = TokenUsage(
            implementation_input_tokens=600,
            implementation_output_tokens=400,
        )
        assert usage.implementation_tokens == 1000

    def test_exploration_token_percentage__calculates_correctly(self):
        """Test exploration percentage calculation."""
        usage = TokenUsage(
            exploration_input_tokens=300,
            exploration_output_tokens=200,  # 500 total exploration
            implementation_input_tokens=300,
            implementation_output_tokens=200,  # 500 total implementation
        )
        # 500 / 1000 = 50%
        assert usage.exploration_token_percentage == 50.0

    def test_exploration_token_percentage__handles_zero_total(self):
        """Test exploration percentage with zero tokens."""
        usage = TokenUsage()
        assert usage.exploration_token_percentage == 0.0


class TestTranscriptTokenAnalyzer:
    """Tests for TranscriptTokenAnalyzer."""

    @pytest.fixture
    def fixture_transcript_path(self):
        """Path to the real transcript fixture."""
        path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert path.exists(), f"transcript.jsonl fixture missing at {path}"
        return path

    def test_analyze_transcript__parses_real_session(self, fixture_transcript_path):
        """Test analysis using a real transcript fixture."""
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(fixture_transcript_path)

        assert usage is not None
        assert isinstance(usage, TokenUsage)

        # Real transcript should have some tokens
        assert usage.total_input_tokens > 0
        assert usage.total_output_tokens > 0

    def test_analyze_transcript__categorizes_tokens(self, fixture_transcript_path):
        """Test that tokens are categorized into exploration/implementation."""
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(fixture_transcript_path)

        # Total categorized should roughly match total
        categorized = usage.exploration_tokens + usage.implementation_tokens
        total = usage.total_tokens

        # Allow some tolerance for rounding in mixed-tool messages
        assert categorized > 0
        # Categorized shouldn't exceed total
        assert categorized <= total + 100  # Small tolerance for edge cases

    def test_analyze_transcript__captures_exploration_percentage(self, fixture_transcript_path):
        """Test that exploration percentage is calculated."""
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(fixture_transcript_path)

        # The test transcript has Explore agents, so some exploration tokens expected
        percentage = usage.exploration_token_percentage
        assert 0 <= percentage <= 100

    def test_analyze_transcript__handles_missing_file(self):
        """Test graceful handling of missing file."""
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(Path("/nonexistent/path.jsonl"))

        assert usage is not None
        assert usage.total_tokens == 0

    def test_analyze_transcript__handles_malformed_json(self, tmp_path):
        """Test graceful handling of malformed JSON lines."""
        transcript_file = tmp_path / "malformed.jsonl"
        transcript_file.write_text('{"valid": "json"}\nnot valid json\n{"another": "valid"}\n')

        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript_file)

        # Should not crash, should return empty usage
        assert usage is not None

    def test_analyze_transcript__handles_empty_file(self, tmp_path):
        """Test handling of empty file."""
        transcript_file = tmp_path / "empty.jsonl"
        transcript_file.write_text("")

        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript_file)

        assert usage is not None
        assert usage.total_tokens == 0


class TestToolClassification:
    """Tests for tool classification logic."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TranscriptTokenAnalyzer()

    def test_classify_tool__read_is_exploration(self, analyzer):
        """Test that Read tool is classified as exploration."""
        result = analyzer._classify_tool("Read", {"file_path": "/test.py"})
        assert result == "exploration"

    def test_classify_tool__grep_is_exploration(self, analyzer):
        """Test that Grep tool is classified as exploration."""
        result = analyzer._classify_tool("Grep", {"pattern": "test"})
        assert result == "exploration"

    def test_classify_tool__glob_is_exploration(self, analyzer):
        """Test that Glob tool is classified as exploration."""
        result = analyzer._classify_tool("Glob", {"pattern": "*.py"})
        assert result == "exploration"

    def test_classify_tool__edit_is_implementation(self, analyzer):
        """Test that Edit tool is classified as implementation."""
        result = analyzer._classify_tool("Edit", {"file_path": "/test.py"})
        assert result == "implementation"

    def test_classify_tool__write_is_implementation(self, analyzer):
        """Test that Write tool is classified as implementation."""
        result = analyzer._classify_tool("Write", {"file_path": "/test.py"})
        assert result == "implementation"

    def test_classify_tool__bash_is_implementation(self, analyzer):
        """Test that Bash tool is classified as implementation."""
        result = analyzer._classify_tool("Bash", {"command": "ls"})
        assert result == "implementation"

    def test_classify_tool__task_explore_is_exploration(self, analyzer):
        """Test that Task with Explore subagent is exploration."""
        result = analyzer._classify_tool("Task", {"subagent_type": "Explore"})
        assert result == "exploration"

    def test_classify_tool__task_explore_case_insensitive(self, analyzer):
        """Test that Task Explore detection is case-insensitive."""
        result = analyzer._classify_tool("Task", {"subagent_type": "explore"})
        assert result == "exploration"

    def test_classify_tool__task_plan_is_implementation(self, analyzer):
        """Test that Task with Plan subagent is implementation."""
        result = analyzer._classify_tool("Task", {"subagent_type": "Plan"})
        assert result == "implementation"

    def test_classify_tool__unknown_is_implementation(self, analyzer):
        """Test that unknown tools default to implementation."""
        result = analyzer._classify_tool("UnknownTool", {})
        assert result == "implementation"


class TestConvenienceFunction:
    """Tests for the analyze_transcript_tokens convenience function."""

    @pytest.fixture
    def fixture_transcript_path(self):
        """Path to the real transcript fixture."""
        path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert path.exists(), f"transcript.jsonl fixture missing at {path}"
        return path

    def test_analyze_transcript_tokens__returns_token_usage(self, fixture_transcript_path):
        """Test convenience function returns TokenUsage."""
        usage = analyze_transcript_tokens(fixture_transcript_path)

        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens > 0


class TestRealTranscriptAnalysis:
    """Integration tests with the real transcript fixture."""

    @pytest.fixture
    def fixture_transcript_path(self):
        """Path to the real transcript fixture."""
        path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert path.exists(), f"transcript.jsonl fixture missing at {path}"
        return path

    def test_real_transcript__has_exploration_tokens(self, fixture_transcript_path):
        """Test that real transcript has exploration tokens from Explore agents."""
        usage = analyze_transcript_tokens(fixture_transcript_path)

        # The test transcript is from a session that used Explore agents
        # and Read/Grep tools, so exploration tokens should be present
        assert usage.exploration_tokens > 0

    def test_real_transcript__has_implementation_tokens(self, fixture_transcript_path):
        """Test that real transcript has implementation tokens."""
        usage = analyze_transcript_tokens(fixture_transcript_path)

        # The session made edits, so implementation tokens expected
        assert usage.implementation_tokens > 0

    def test_real_transcript__token_breakdown_reasonable(self, fixture_transcript_path):
        """Test that token breakdown is reasonable for the known session."""
        usage = analyze_transcript_tokens(fixture_transcript_path)

        # Session involved both exploration and implementation
        # Neither should dominate completely (sanity check)
        exploration_pct = usage.exploration_token_percentage
        assert exploration_pct > 0
        assert exploration_pct < 100

        # Verify we captured significant tokens
        assert usage.total_tokens > 10000  # Real session should have many tokens

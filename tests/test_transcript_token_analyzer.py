"""Tests for TranscriptTokenAnalyzer using real transcript data."""

from pathlib import Path

import pytest

from slopometry.core.models.session import TokenUsage
from slopometry.core.transcript_token_analyzer import (
    TranscriptTokenAnalyzer,
    analyze_opencode_transcript_tokens,
    analyze_transcript_tokens,
)


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

    def test_exploration_tokens__includes_explore_subagent(self):
        """Exploration tokens include main agent exploration + Explore subagent work."""
        usage = TokenUsage(
            exploration_input_tokens=800,
            exploration_output_tokens=200,
            explore_subagent_tokens=5000,
        )
        assert usage.exploration_tokens == 6000

    def test_implementation_tokens__is_final_context_plus_non_explore_subagent(self):
        """Implementation tokens = final context input + non-explore subagent tokens."""
        usage = TokenUsage(
            final_context_input_tokens=100_000,
            non_explore_subagent_tokens=20_000,
        )
        assert usage.implementation_tokens == 120_000

    def test_implementation_tokens__zero_when_no_final_context(self):
        """Implementation tokens are 0 with no final context or subagent tokens."""
        usage = TokenUsage(
            implementation_input_tokens=600,
            implementation_output_tokens=400,
        )
        assert usage.implementation_tokens == 0

    def test_exploration_token_percentage__calculates_correctly(self):
        """Test exploration percentage calculation with new definitions."""
        usage = TokenUsage(
            exploration_input_tokens=300,
            exploration_output_tokens=200,  # 500 exploration from main agent
            explore_subagent_tokens=500,  # 500 from Explore subagents = 1000 total exploration
            final_context_input_tokens=800,
            non_explore_subagent_tokens=200,  # 1000 total implementation
        )
        # 1000 / 2000 = 50%
        assert usage.exploration_token_percentage == 50.0

    def test_exploration_token_percentage__handles_zero_total(self):
        """Test exploration percentage with zero tokens."""
        usage = TokenUsage()
        assert usage.exploration_token_percentage == 0.0

    def test_changeset_tokens__default_zero(self):
        """Changeset tokens default to 0."""
        usage = TokenUsage()
        assert usage.changeset_tokens == 0


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

        # Exploration and implementation are now independent metrics (not a partition).
        # Exploration = main-agent exploration i/o + explore subagent tokens.
        # Implementation = final context input + non-explore subagent tokens.
        # Both should be populated for a real transcript.
        assert usage.exploration_tokens > 0 or usage.implementation_tokens > 0
        assert usage.total_tokens > 0

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


class TestIncrementalInputCounting:
    """Tests for incremental input token counting (avoids cumulative over-counting)."""

    def _make_transcript(self, tmp_path, turns: list[tuple[int, int, str]]) -> Path:
        """Create a synthetic transcript JSONL.

        Args:
            turns: List of (input_tokens, output_tokens, tool_name) per assistant turn.
                   tool_name is used to create a tool_use content block.
        """
        import json

        transcript_file = tmp_path / "test_transcript.jsonl"
        lines = []
        for input_tok, output_tok, tool_name in turns:
            content = []
            if tool_name:
                content.append({"type": "tool_use", "name": tool_name, "input": {}})
            else:
                content.append({"type": "text", "text": "hello"})

            event = {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": input_tok, "output_tokens": output_tok},
                    "content": content,
                },
            }
            lines.append(json.dumps(event))
        transcript_file.write_text("\n".join(lines))
        return transcript_file

    def test_incremental_input__avoids_cumulative_overcounting(self, tmp_path):
        """Input tokens grow cumulatively per turn; only deltas should be counted."""
        # Simulates 3 turns where input_tokens is cumulative (includes prior context)
        transcript = self._make_transcript(
            tmp_path,
            [
                (50_000, 5_000, "Edit"),  # Turn 1: 50K initial context
                (120_000, 3_000, "Read"),  # Turn 2: grew by 70K (prev output + new input)
                (200_000, 4_000, "Edit"),  # Turn 3: grew by 80K
            ],
        )
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        # Incremental input: 50K + 70K + 80K = 200K (not 50K + 120K + 200K = 370K)
        assert usage.total_input_tokens == 200_000
        assert usage.total_output_tokens == 12_000

    def test_incremental_input__single_turn_uses_full_value(self, tmp_path):
        """First turn's input_tokens should be counted in full."""
        transcript = self._make_transcript(tmp_path, [(100_000, 5_000, "Edit")])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.total_input_tokens == 100_000
        assert usage.total_output_tokens == 5_000

    def test_incremental_input__handles_context_compaction(self, tmp_path):
        """After context compaction, input_tokens may decrease — delta should clamp to 0."""
        transcript = self._make_transcript(
            tmp_path,
            [
                (100_000, 5_000, "Edit"),  # Turn 1
                (200_000, 3_000, "Read"),  # Turn 2: grew by 100K
                (80_000, 4_000, "Edit"),  # Turn 3: compacted down (input < previous)
                (120_000, 2_000, "Read"),  # Turn 4: grew by 40K from compacted state
            ],
        )
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        # Incremental: 100K + 100K + 0 (clamped) + 40K = 240K
        assert usage.total_input_tokens == 240_000
        assert usage.total_output_tokens == 14_000

    def test_incremental_input__categorization_uses_deltas(self, tmp_path):
        """Exploration/implementation split should use incremental input, not raw."""
        transcript = self._make_transcript(
            tmp_path,
            [
                (50_000, 5_000, "Read"),  # Turn 1: exploration, 50K input
                (120_000, 3_000, "Edit"),  # Turn 2: implementation, 70K incremental input
            ],
        )
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.exploration_input_tokens == 50_000
        assert usage.implementation_input_tokens == 70_000
        assert usage.total_input_tokens == 120_000


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


class TestOpenCodeTranscriptTokens:
    """Tests for analyze_opencode_transcript_tokens."""

    @staticmethod
    def _make_oc_msg(
        role: str,
        input_tok: int,
        output_tok: int,
        reasoning: int = 0,
        tools: list[str] | None = None,
        agent: str | None = None,
    ) -> dict:
        """Build an OpenCode transcript message."""
        parts = []
        if tools:
            for t in tools:
                parts.append({"type": "tool-invocation", "tool": t})
        else:
            parts.append({"type": "text", "text": "hello"})
        msg: dict = {"role": role, "parts": parts}
        if role == "assistant":
            msg["tokens"] = {"input": input_tok, "output": output_tok, "reasoning": reasoning}
        if agent is not None:
            msg["agent"] = agent
        return msg

    def test_basic_token_counting(self):
        """OpenCode transcript with cumulative input tokens should use incremental counting."""
        transcript = [
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["Edit"]),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 120_000, 3_000, tools=["Read"]),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 200_000, 4_000, tools=["Edit"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        # Incremental: 50K + 70K + 80K = 200K
        assert usage.total_input_tokens == 200_000
        assert usage.total_output_tokens == 12_000

    def test_reasoning_tokens_included_in_output(self):
        """Reasoning tokens should be counted as output."""
        transcript = [
            self._make_oc_msg("assistant", 10_000, 2_000, reasoning=1_000, tools=["Edit"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.total_input_tokens == 10_000
        assert usage.total_output_tokens == 3_000  # 2K output + 1K reasoning

    def test_exploration_vs_implementation_classification(self):
        """Tools should be classified as exploration or implementation."""
        transcript = [
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["Read"]),  # exploration
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 120_000, 3_000, tools=["Edit"]),  # implementation
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.exploration_input_tokens == 50_000
        assert usage.exploration_output_tokens == 5_000
        assert usage.implementation_input_tokens == 70_000  # 120K - 50K
        assert usage.implementation_output_tokens == 3_000

    def test_mixed_tools_in_single_message(self):
        """Message with both exploration and implementation tools should split tokens."""
        transcript = [
            self._make_oc_msg("assistant", 100_000, 10_000, tools=["Read", "Edit"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        # 50/50 split
        assert usage.exploration_input_tokens == 50_000
        assert usage.implementation_input_tokens == 50_000

    def test_user_messages_ignored(self):
        """User messages should not contribute to token counts."""
        transcript = [
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["Edit"]),
            self._make_oc_msg("user", 0, 0),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.total_input_tokens == 50_000
        assert usage.total_output_tokens == 5_000

    def test_empty_transcript(self):
        """Empty transcript should return zero usage."""
        usage = analyze_opencode_transcript_tokens([])

        assert usage.total_tokens == 0

    def test_messages_without_tokens(self):
        """Messages missing tokens field should be skipped gracefully."""
        transcript = [
            {"role": "assistant", "parts": [{"type": "text", "text": "hi"}]},
            self._make_oc_msg("assistant", 10_000, 1_000, tools=["Edit"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.total_input_tokens == 10_000
        assert usage.total_output_tokens == 1_000

    def test_context_compaction_clamps_to_zero(self):
        """Input tokens decreasing (compaction) should clamp delta to 0."""
        transcript = [
            self._make_oc_msg("assistant", 100_000, 5_000, tools=["Edit"]),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 50_000, 3_000, tools=["Read"]),  # compacted
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 80_000, 2_000, tools=["Edit"]),  # grew from compacted
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        # Incremental: 100K + 0 (clamped) + 30K = 130K
        assert usage.total_input_tokens == 130_000
        assert usage.total_output_tokens == 10_000

    def test_agent_explore__classifies_as_exploration(self):
        """Messages with agent=explore should classify entirely as exploration."""
        transcript = [
            self._make_oc_msg("assistant", 5_000, 1_000, tools=["grep", "glob"], agent="explore"),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 10_000, 2_000, tools=["read"], agent="explore"),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 25_000, 3_000, agent="explore"),  # text-only response
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        # All tokens should be exploration
        assert usage.exploration_input_tokens == usage.total_input_tokens
        assert usage.exploration_output_tokens == usage.total_output_tokens
        assert usage.implementation_input_tokens == 0
        assert usage.implementation_output_tokens == 0

    def test_agent_explore__ignores_tool_names(self):
        """agent=explore should override tool classification, even for 'Edit' tools."""
        transcript = [
            self._make_oc_msg("assistant", 10_000, 2_000, tools=["Edit"], agent="explore"),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.exploration_input_tokens == 10_000
        assert usage.implementation_input_tokens == 0

    def test_lowercase_tools__classified_correctly_without_agent(self):
        """Lowercase tool names (OpenCode format) should match ToolType enum case-insensitively."""
        transcript = [
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["grep", "read", "glob"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.exploration_input_tokens == 50_000
        assert usage.exploration_output_tokens == 5_000
        assert usage.implementation_input_tokens == 0

    def test_lowercase_implementation_tools__classified_correctly(self):
        """Lowercase implementation tools should also match correctly."""
        transcript = [
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["edit", "write", "bash"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.implementation_input_tokens == 50_000
        assert usage.exploration_input_tokens == 0

    def test_mixed_session__agent_and_no_agent(self):
        """Mixed session: some messages with agent=explore, some without."""
        transcript = [
            # Explore agent messages
            self._make_oc_msg("assistant", 5_000, 1_000, tools=["grep"], agent="explore"),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 10_000, 2_000, tools=["read"], agent="explore"),
            self._make_oc_msg("user", 0, 0),
            # General agent message with implementation tool
            self._make_oc_msg("assistant", 30_000, 4_000, tools=["Edit"], agent="general"),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        # Explore messages: 5K + 5K = 10K input, 1K + 2K = 3K output
        assert usage.exploration_input_tokens == 10_000
        assert usage.exploration_output_tokens == 3_000
        # General message: 30K - 10K = 20K incremental input
        assert usage.implementation_input_tokens == 20_000
        assert usage.implementation_output_tokens == 4_000

    def test_agent_search__classifies_as_exploration(self):
        """Messages with agent=search should classify as exploration."""
        transcript = [
            self._make_oc_msg("assistant", 10_000, 2_000, tools=["grep"], agent="search"),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)

        assert usage.exploration_input_tokens == 10_000
        assert usage.implementation_input_tokens == 0

    def test_final_context_input_tokens__set_from_last_message(self):
        """final_context_input_tokens should equal raw input_tokens of last assistant message."""
        transcript = [
            self._make_oc_msg("assistant", 50_000, 5_000, tools=["Edit"]),
            self._make_oc_msg("user", 0, 0),
            self._make_oc_msg("assistant", 120_000, 3_000, tools=["Read"]),
        ]
        usage = analyze_opencode_transcript_tokens(transcript)
        assert usage.final_context_input_tokens == 120_000


class TestSubagentRouting:
    """Tests for routing subagent tokens to explore vs non-explore buckets."""

    def _make_transcript(self, tmp_path, events: list[dict]) -> Path:
        """Create a synthetic transcript JSONL from raw event dicts."""
        import json

        transcript_file = tmp_path / "test_transcript.jsonl"
        lines = [json.dumps(e) for e in events]
        transcript_file.write_text("\n".join(lines))
        return transcript_file

    def _assistant_event(self, input_tokens: int, output_tokens: int, tool_uses: list[dict]) -> dict:
        """Build an assistant event with tool_use content blocks."""
        content = []
        for tu in tool_uses:
            content.append({
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu.get("input", {}),
            })
        return {
            "type": "assistant",
            "message": {
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "content": content,
            },
        }

    def _user_event_with_subagent(self, tool_use_id: str, total_tokens: int) -> dict:
        """Build a user event with toolUseResult carrying subagent tokens."""
        return {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id},
                ],
            },
            "toolUseResult": {"totalTokens": total_tokens},
        }

    def test_explore_subagent_tokens__routed_to_explore_bucket(self, tmp_path):
        """Explore Task subagent tokens should go to explore_subagent_tokens."""
        transcript = self._make_transcript(tmp_path, [
            self._assistant_event(50_000, 5_000, [
                {"id": "tu_1", "name": "Task", "input": {"subagent_type": "Explore"}},
            ]),
            self._user_event_with_subagent("tu_1", 100_000),
        ])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.explore_subagent_tokens == 100_000
        assert usage.non_explore_subagent_tokens == 0
        assert usage.subagent_tokens == 100_000
        # exploration_tokens includes explore_subagent_tokens
        assert usage.exploration_tokens == 5_000 + 50_000 + 100_000

    def test_non_explore_subagent_tokens__routed_to_non_explore_bucket(self, tmp_path):
        """Non-Explore Task subagent tokens should go to non_explore_subagent_tokens."""
        transcript = self._make_transcript(tmp_path, [
            self._assistant_event(50_000, 5_000, [
                {"id": "tu_2", "name": "Task", "input": {"subagent_type": "Bash"}},
            ]),
            self._user_event_with_subagent("tu_2", 80_000),
        ])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.explore_subagent_tokens == 0
        assert usage.non_explore_subagent_tokens == 80_000
        assert usage.subagent_tokens == 80_000

    def test_mixed_subagent_tokens__routed_correctly(self, tmp_path):
        """Mixed Explore and non-Explore Task invocations route to correct buckets."""
        transcript = self._make_transcript(tmp_path, [
            self._assistant_event(50_000, 5_000, [
                {"id": "tu_explore", "name": "Task", "input": {"subagent_type": "Explore"}},
            ]),
            self._user_event_with_subagent("tu_explore", 100_000),
            self._assistant_event(100_000, 3_000, [
                {"id": "tu_bash", "name": "Task", "input": {"subagent_type": "Bash"}},
            ]),
            self._user_event_with_subagent("tu_bash", 50_000),
        ])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.explore_subagent_tokens == 100_000
        assert usage.non_explore_subagent_tokens == 50_000
        assert usage.subagent_tokens == 150_000

    def test_unknown_tool_use_id__defaults_to_non_explore(self, tmp_path):
        """Subagent tokens with unknown tool_use_id default to non-explore."""
        transcript = self._make_transcript(tmp_path, [
            self._assistant_event(50_000, 5_000, [
                {"id": "tu_known", "name": "Read", "input": {}},
            ]),
            self._user_event_with_subagent("tu_unknown", 30_000),
        ])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.non_explore_subagent_tokens == 30_000
        assert usage.explore_subagent_tokens == 0


class TestFinalContextInputTokens:
    """Tests for final_context_input_tokens tracking."""

    def _make_transcript(self, tmp_path, turns: list[tuple[int, int, str]]) -> Path:
        """Create a synthetic transcript JSONL."""
        import json

        transcript_file = tmp_path / "test_transcript.jsonl"
        lines = []
        for input_tok, output_tok, tool_name in turns:
            content = []
            if tool_name:
                content.append({"type": "tool_use", "id": f"tu_{input_tok}", "name": tool_name, "input": {}})
            else:
                content.append({"type": "text", "text": "hello"})
            event = {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": input_tok, "output_tokens": output_tok},
                    "content": content,
                },
            }
            lines.append(json.dumps(event))
        transcript_file.write_text("\n".join(lines))
        return transcript_file

    def test_final_context__equals_last_raw_input(self, tmp_path):
        """final_context_input_tokens should be raw input_tokens of the last assistant message."""
        transcript = self._make_transcript(
            tmp_path,
            [
                (50_000, 5_000, "Read"),
                (120_000, 3_000, "Edit"),
                (200_000, 4_000, "Edit"),
            ],
        )
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.final_context_input_tokens == 200_000

    def test_final_context__single_turn(self, tmp_path):
        """Single turn should set final_context_input_tokens to that turn's input."""
        transcript = self._make_transcript(tmp_path, [(100_000, 5_000, "Edit")])
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.final_context_input_tokens == 100_000

    def test_final_context__after_compaction(self, tmp_path):
        """After compaction, final_context_input_tokens should reflect the smaller value."""
        transcript = self._make_transcript(
            tmp_path,
            [
                (200_000, 5_000, "Edit"),
                (80_000, 4_000, "Edit"),  # compacted
            ],
        )
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.final_context_input_tokens == 80_000

    def test_final_context__empty_transcript(self, tmp_path):
        """Empty transcript should have final_context_input_tokens = 0."""
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")
        analyzer = TranscriptTokenAnalyzer()
        usage = analyzer.analyze_transcript(transcript)

        assert usage.final_context_input_tokens == 0


class TestCountGitDiffTokens:
    """Tests for count_git_diff_tokens function."""

    def test_count_git_diff_tokens__returns_zero_for_non_git_dir(self, tmp_path):
        """Non-git directory should return 0."""
        from slopometry.core.tokenizer import count_git_diff_tokens

        result = count_git_diff_tokens(tmp_path)
        assert result == 0

    def test_count_git_diff_tokens__returns_zero_for_clean_repo(self, tmp_path):
        """Clean git repo with no changes should return 0."""
        import subprocess

        from slopometry.core.tokenizer import count_git_diff_tokens

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        result = count_git_diff_tokens(tmp_path)
        assert result == 0

    def test_count_git_diff_tokens__counts_unstaged_changes(self, tmp_path):
        """Unstaged changes should produce non-zero token count."""
        import subprocess

        from slopometry.core.tokenizer import count_git_diff_tokens

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        # Make unstaged changes
        (tmp_path / "file.txt").write_text("hello world\nmore content\n")

        result = count_git_diff_tokens(tmp_path)
        assert result > 0

    def test_count_git_diff_tokens__counts_staged_changes(self, tmp_path):
        """Staged changes should produce non-zero token count."""
        import subprocess

        from slopometry.core.tokenizer import count_git_diff_tokens

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        # Make staged changes
        (tmp_path / "file.txt").write_text("hello world\nmore content\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        result = count_git_diff_tokens(tmp_path)
        assert result > 0

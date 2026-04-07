"""Tests for behavioral pattern analyzer."""

import json
from pathlib import Path

import pytest

from slopometry.core.behavioral_pattern_analyzer import (
    OWNERSHIP_DODGING_RE,
    SIMPLE_WORKAROUND_RE,
    TranscriptAssistantEvent,
    _extract_assistant_text,
    _extract_snippet,
    analyze_behavioral_patterns,
    analyze_opencode_behavioral_patterns,
)


def _make_assistant_event(text: str, timestamp: str = "2026-04-07T10:00:00Z") -> dict:
    """Create a minimal assistant transcript event with a text block."""
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": text}],
        },
        "timestamp": timestamp,
    }


def _make_thinking_event(thinking: str, text: str = "") -> dict:
    """Create an assistant event with a thinking block and optional text."""
    content: list[dict] = [{"type": "thinking", "thinking": thinking}]
    if text:
        content.append({"type": "text", "text": text})
    return {
        "type": "assistant",
        "message": {"content": content},
        "timestamp": "2026-04-07T10:00:00Z",
    }


def _make_user_event(text: str) -> dict:
    """Create a user transcript event."""
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": "2026-04-07T10:00:00Z",
    }


def _make_tool_use_event(tool_name: str, tool_input: dict) -> dict:
    """Create an assistant event with a tool_use block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": tool_name, "input": tool_input},
            ],
        },
        "timestamp": "2026-04-07T10:00:00Z",
    }


def _write_transcript(tmp_path: Path, events: list[dict]) -> Path:
    """Write events to a JSONL file and return its path."""
    path = tmp_path / "transcript.jsonl"
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


class TestAnalyzeTranscript:
    """Tests for analyze_behavioral_patterns."""

    def test_analyze_transcript__detects_ownership_dodging_in_text_blocks(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("This is a pre-existing bug in the codebase."),
            _make_assistant_event("The error was not introduced by my changes."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=10.0)

        assert result.ownership_dodging.count == 2
        assert result.ownership_dodging.matches[0].pattern == "pre-existing"
        assert result.ownership_dodging.matches[1].pattern == "not introduced by"

    def test_analyze_transcript__detects_simple_workaround_patterns(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("The simplest fix would be to skip the check for now."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.simple_workaround.count == 2
        patterns = {m.pattern for m in result.simple_workaround.matches}
        assert "simplest" in patterns
        assert "for now" in patterns

    def test_analyze_transcript__ignores_thinking_blocks(self, tmp_path: Path) -> None:
        events = [
            _make_thinking_event(
                thinking="This is a pre-existing issue, the simplest approach is...",
                text="I'll fix the authentication module.",
            ),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 0
        assert result.simple_workaround.count == 0

    def test_analyze_transcript__ignores_user_messages(self, tmp_path: Path) -> None:
        events = [
            _make_user_event("Is this a pre-existing bug? Give me the simplest fix."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 0
        assert result.simple_workaround.count == 0

    def test_analyze_transcript__ignores_tool_use_blocks(self, tmp_path: Path) -> None:
        events = [
            _make_tool_use_event("Bash", {"command": "echo 'pre-existing simplest fix'"}),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 0
        assert result.simple_workaround.count == 0

    def test_analyze_transcript__calculates_per_minute_rate(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("This is a pre-existing issue."),
            _make_assistant_event("Another pre-existing problem here."),
            _make_assistant_event("Also a known limitation."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=10.0)

        assert result.ownership_dodging.count == 3
        assert result.ownership_dodging_rate == pytest.approx(0.3, abs=0.01)

    def test_analyze_transcript__zero_duration_returns_zero_rate(self, tmp_path: Path) -> None:
        events = [_make_assistant_event("This is a pre-existing bug.")]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=0.0)

        assert result.ownership_dodging.count == 1
        assert result.ownership_dodging_rate == 0.0

    def test_analyze_transcript__handles_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 0
        assert result.simple_workaround.count == 0
        assert not result.has_any

    def test_analyze_transcript__handles_empty_file(self, tmp_path: Path) -> None:
        path = _write_transcript(tmp_path, [])
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert not result.has_any

    def test_analyze_transcript__word_boundary_prevents_false_positives(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("I preexistingly checked the code."),
            _make_assistant_event("This is the easiestest approach."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 0
        assert result.simple_workaround.count == 0

    def test_analyze_transcript__context_snippet_truncation(self, tmp_path: Path) -> None:
        long_text = "A" * 200 + " pre-existing " + "B" * 200
        events = [_make_assistant_event(long_text)]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 1
        snippet = result.ownership_dodging.matches[0].context_snippet
        assert len(snippet) <= 130  # MAX_SNIPPET_LEN + "..." prefix

    def test_analyze_transcript__has_any_property(self, tmp_path: Path) -> None:
        events = [_make_assistant_event("Everything looks good, no issues.")]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert not result.has_any

    def test_analyze_transcript__preserves_timestamp(self, tmp_path: Path) -> None:
        events = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "This is a known issue."}]},
                "timestamp": "2026-04-07T14:30:00Z",
            }
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 1
        ts = result.ownership_dodging.matches[0].timestamp
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 4
        assert ts.hour == 14

    def test_analyze_transcript__multiple_matches_in_single_message(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("This existing bug is a separate issue and a known limitation."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 3
        patterns = {m.pattern for m in result.ownership_dodging.matches}
        assert "existing bug" in patterns
        assert "separate issue" in patterns
        assert "known limitation" in patterns

    def test_analyze_transcript__case_insensitive(self, tmp_path: Path) -> None:
        events = [
            _make_assistant_event("This is a Pre-Existing issue."),
            _make_assistant_event("The SIMPLEST approach works."),
        ]
        path = _write_transcript(tmp_path, events)
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 1
        assert result.simple_workaround.count == 1

    def test_analyze_transcript__skips_malformed_json_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        with open(path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_make_assistant_event("This is a pre-existing bug.")) + "\n")
            f.write("{broken\n")
        result = analyze_behavioral_patterns(path, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 1


class TestAnalyzeOpencodeTranscript:
    """Tests for analyze_opencode_behavioral_patterns."""

    def test_analyze_opencode_transcript__detects_patterns(self) -> None:
        transcript = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "text": "This is a pre-existing bug. The simplest fix is..."}],
            },
            {
                "role": "user",
                "parts": [{"type": "text", "text": "Fix it properly."}],
            },
        ]
        result = analyze_opencode_behavioral_patterns(transcript, session_duration_minutes=5.0)

        assert result.ownership_dodging.count == 1
        assert result.simple_workaround.count == 1

    def test_analyze_opencode_transcript__ignores_user_messages(self) -> None:
        transcript = [
            {"role": "user", "parts": [{"type": "text", "text": "Is this pre-existing?"}]},
        ]
        result = analyze_opencode_behavioral_patterns(transcript, session_duration_minutes=5.0)

        assert not result.has_any


class TestRealTranscripts:
    """Tests against real Claude Code transcript fixtures across versions."""

    FIXTURES_DIR = Path(__file__).parent / "fixtures"

    def test_analyze_transcript__real_v1_fixture_parses_without_error(self) -> None:
        """The v1 fixture (CC ~2.0.65, summary-prefixed format) parses successfully."""
        path = self.FIXTURES_DIR / "transcript.jsonl"
        if not path.exists():
            pytest.skip("transcript.jsonl fixture not present")
        result = analyze_behavioral_patterns(path, session_duration_minutes=10.0)

        assert result.session_duration_minutes == 10.0
        # Pattern counts are non-negative (don't assert specific counts — fixture content may vary)
        assert result.ownership_dodging.count >= 0
        assert result.simple_workaround.count >= 0
        # Every match has required fields populated
        for match in result.ownership_dodging.matches + result.simple_workaround.matches:
            assert match.pattern
            assert match.line_number > 0
            assert match.context_snippet

    def test_analyze_transcript__real_v2_fixture_parses_without_error(self) -> None:
        """The v2 fixture (CC ~2.1.34, file-history-snapshot format) parses successfully."""
        path = self.FIXTURES_DIR / "transcript_v2.jsonl"
        if not path.exists():
            pytest.skip("transcript_v2.jsonl fixture not present")
        result = analyze_behavioral_patterns(path, session_duration_minutes=10.0)

        assert result.session_duration_minutes == 10.0
        assert result.ownership_dodging.count >= 0
        assert result.simple_workaround.count >= 0
        for match in result.ownership_dodging.matches + result.simple_workaround.matches:
            assert match.pattern
            assert match.line_number > 0
            assert match.context_snippet

    def test_analyze_transcript__real_fixtures_only_match_assistant_text(self) -> None:
        """Verify matches come from assistant messages, not user or system events."""
        for fixture_name in ("transcript.jsonl", "transcript_v2.jsonl"):
            path = self.FIXTURES_DIR / fixture_name
            if not path.exists():
                continue
            result = analyze_behavioral_patterns(path, session_duration_minutes=10.0)

            # Verify each match's line_number corresponds to an assistant event
            if not result.ownership_dodging.matches and not result.simple_workaround.matches:
                continue  # No matches to verify

            import json

            with open(path) as f:
                lines = f.readlines()
            for match in result.ownership_dodging.matches + result.simple_workaround.matches:
                line_idx = match.line_number - 1  # 1-indexed
                assert line_idx < len(lines), f"Line {match.line_number} out of range in {fixture_name}"
                event = json.loads(lines[line_idx])
                assert event.get("type") == "assistant", (
                    f"Match '{match.pattern}' at line {match.line_number} in {fixture_name} "
                    f"came from type={event.get('type')}, expected assistant"
                )


class TestHelpers:
    """Tests for helper functions."""

    def test_extract_assistant_text__string_content(self) -> None:
        event = TranscriptAssistantEvent.model_validate({"message": {"content": "Hello, this is a string response."}})
        assert _extract_assistant_text(event) == "Hello, this is a string response."

    def test_extract_assistant_text__list_content_filters_text_only(self) -> None:
        event = TranscriptAssistantEvent.model_validate(
            {
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Here is the answer."},
                        {"type": "tool_use", "name": "Bash", "input": {}},
                        {"type": "text", "text": "And more details."},
                    ]
                }
            }
        )
        result = _extract_assistant_text(event)
        assert "Here is the answer." in result
        assert "And more details." in result
        assert "Let me think" not in result

    def test_extract_snippet__short_text(self) -> None:
        text = "This is a pre-existing bug."
        snippet = _extract_snippet(text, 10, 22)
        assert "pre-existing" in snippet

    def test_extract_snippet__adds_ellipsis_prefix(self) -> None:
        text = "A" * 100 + " pre-existing " + "B" * 100
        snippet = _extract_snippet(text, 101, 113)
        assert snippet.startswith("...")

    def test_regex_patterns__ownership_dodging_matches(self) -> None:
        text = "This is a pre-existing issue in the codebase"
        matches = list(OWNERSHIP_DODGING_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group().lower() == "pre-existing"

    def test_regex_patterns__simple_workaround_matches(self) -> None:
        text = "The simplest approach for now would be a quick fix"
        matches = list(SIMPLE_WORKAROUND_RE.finditer(text))
        patterns = {m.group().lower() for m in matches}
        assert "simplest" in patterns
        assert "for now" in patterns
        assert "quick fix" in patterns

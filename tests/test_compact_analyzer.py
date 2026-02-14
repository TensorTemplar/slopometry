"""Tests for compact event analysis from Claude Code transcripts."""

import json
from pathlib import Path

from slopometry.core.compact_analyzer import (
    CompactEventAnalyzer,
    analyze_transcript_compacts,
    find_compact_instructions,
)
from slopometry.core.models.session import CompactEvent


class TestCompactEventAnalyzer:
    """Tests for CompactEventAnalyzer class."""

    def test_analyze_transcript__finds_compact_events_in_fixture(self) -> None:
        """Test that analyzer finds compact events in the real transcript fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert fixture_path.exists(), "Transcript fixture required at tests/fixtures/transcript.jsonl"

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(fixture_path)

        assert len(compacts) >= 1
        assert all(isinstance(c, CompactEvent) for c in compacts)

    def test_analyze_transcript__extracts_correct_metadata(self) -> None:
        """Test that analyzer extracts correct metadata from compact events."""
        fixture_path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert fixture_path.exists(), "Transcript fixture required at tests/fixtures/transcript.jsonl"

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(fixture_path)

        assert len(compacts) >= 1
        first_compact = compacts[0]

        assert first_compact.trigger == "auto"
        assert first_compact.pre_tokens == 155317
        assert first_compact.line_number == 398
        assert first_compact.uuid == "947c352a-de46-478b-aadd-16ba1db38bbb"
        assert "This session is being continued" in first_compact.summary_content
        assert first_compact.version == "2.0.65"
        assert first_compact.git_branch == "opinionated-metrics"

    def test_analyze_transcript__handles_missing_file(self) -> None:
        """Test that analyzer handles missing file gracefully."""
        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(Path("/nonexistent/path/transcript.jsonl"))

        assert compacts == []

    def test_analyze_transcript__handles_empty_file(self, tmp_path: Path) -> None:
        """Test that analyzer handles empty file gracefully."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(empty_file)

        assert compacts == []

    def test_analyze_transcript__handles_malformed_json(self, tmp_path: Path) -> None:
        """Test that analyzer handles malformed JSON lines gracefully."""
        malformed_file = tmp_path / "malformed.jsonl"
        malformed_file.write_text("not valid json\n{also: invalid}")

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(malformed_file)

        assert compacts == []

    def test_analyze_transcript__parses_compact_boundary_and_summary_pair(self, tmp_path: Path) -> None:
        """Test that analyzer correctly pairs compact_boundary with isCompactSummary."""
        transcript_file = tmp_path / "transcript.jsonl"

        boundary_event = {
            "type": "system",
            "subtype": "compact_boundary",
            "content": "Conversation compacted",
            "timestamp": "2025-12-12T14:31:13.441Z",
            "uuid": "test-uuid-123",
            "compactMetadata": {"trigger": "manual", "preTokens": 50000},
        }
        summary_event = {
            "type": "user",
            "parentUuid": "test-uuid-123",
            "isCompactSummary": True,
            "message": {"content": "Summary of previous conversation..."},
            "timestamp": "2025-12-12T14:31:13.442Z",
        }

        with open(transcript_file, "w") as f:
            f.write(json.dumps(boundary_event) + "\n")
            f.write(json.dumps(summary_event) + "\n")

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(transcript_file)

        assert len(compacts) == 1
        compact = compacts[0]
        assert compact.trigger == "manual"
        assert compact.pre_tokens == 50000
        assert compact.line_number == 1
        assert compact.uuid == "test-uuid-123"
        assert compact.summary_content == "Summary of previous conversation..."

    def test_analyze_transcript__ignores_orphan_boundary(self, tmp_path: Path) -> None:
        """Test that analyzer ignores compact_boundary without matching summary."""
        transcript_file = tmp_path / "transcript.jsonl"

        boundary_event = {
            "type": "system",
            "subtype": "compact_boundary",
            "content": "Conversation compacted",
            "timestamp": "2025-12-12T14:31:13.441Z",
            "uuid": "orphan-uuid",
            "compactMetadata": {"trigger": "auto", "preTokens": 10000},
        }

        with open(transcript_file, "w") as f:
            f.write(json.dumps(boundary_event) + "\n")

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(transcript_file)

        assert compacts == []

    def test_analyze_transcript__handles_multiple_compacts(self, tmp_path: Path) -> None:
        """Test that analyzer finds multiple compact events."""
        transcript_file = tmp_path / "transcript.jsonl"

        events = []
        for i in range(3):
            boundary = {
                "type": "system",
                "subtype": "compact_boundary",
                "timestamp": f"2025-12-12T14:3{i}:00.000Z",
                "uuid": f"uuid-{i}",
                "compactMetadata": {"trigger": "auto", "preTokens": 50000 * (i + 1)},
            }
            summary = {
                "type": "user",
                "parentUuid": f"uuid-{i}",
                "isCompactSummary": True,
                "message": {"content": f"Summary {i}"},
                "timestamp": f"2025-12-12T14:3{i}:01.000Z",
            }
            events.extend([boundary, summary])

        with open(transcript_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        analyzer = CompactEventAnalyzer()
        compacts = analyzer.analyze_transcript(transcript_file)

        assert len(compacts) == 3
        assert compacts[0].pre_tokens == 50000
        assert compacts[1].pre_tokens == 100000
        assert compacts[2].pre_tokens == 150000


class TestFindCompactInstructions:
    """Tests for find_compact_instructions function."""

    def test_find_compact_instructions__finds_compact_command(self, tmp_path: Path) -> None:
        """Test that function finds /compact command before compact event."""
        transcript_file = tmp_path / "transcript.jsonl"

        events = [
            {"type": "user", "message": {"content": "Let's fix this bug"}},
            {"type": "assistant", "message": {"content": "Working on it..."}},
            {
                "type": "user",
                "message": {"content": "/compact please summarize what we've done"},
            },
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "uuid-1",
                "compactMetadata": {"trigger": "manual"},
            },
        ]

        with open(transcript_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        instructions = find_compact_instructions(transcript_file, 4)

        assert instructions is not None
        assert "/compact" in instructions.lower()
        assert "summarize" in instructions

    def test_find_compact_instructions__returns_none_for_auto_compact(self, tmp_path: Path) -> None:
        """Test that function returns None when no /compact command found."""
        transcript_file = tmp_path / "transcript.jsonl"

        events = [
            {"type": "user", "message": {"content": "Regular message"}},
            {"type": "assistant", "message": {"content": "Response"}},
            {
                "type": "system",
                "subtype": "compact_boundary",
                "uuid": "uuid-1",
                "compactMetadata": {"trigger": "auto"},
            },
        ]

        with open(transcript_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        instructions = find_compact_instructions(transcript_file, 3)

        assert instructions is None

    def test_find_compact_instructions__handles_missing_file(self) -> None:
        """Test that function handles missing file gracefully."""
        instructions = find_compact_instructions(Path("/nonexistent/transcript.jsonl"), 10)
        assert instructions is None


class TestAnalyzeTranscriptCompactsConvenience:
    """Tests for analyze_transcript_compacts convenience function."""

    def test_analyze_transcript_compacts__works_with_fixture(self) -> None:
        """Test convenience function works with real fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert fixture_path.exists(), "Transcript fixture required at tests/fixtures/transcript.jsonl"

        compacts = analyze_transcript_compacts(fixture_path)

        assert len(compacts) >= 1
        assert all(isinstance(c, CompactEvent) for c in compacts)

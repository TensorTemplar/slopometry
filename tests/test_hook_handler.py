"""Tests for hook handler functionality."""

from slopometry.core.hook_handler import (
    detect_event_type_from_parsed,
    extract_dev_guidelines_from_claude_md,
    format_code_smell_feedback,
    format_context_coverage_feedback,
)
from slopometry.core.models import (
    ContextCoverage,
    ExtendedComplexityMetrics,
    FileCoverageStatus,
    HookEventType,
    ImpactAssessment,
    ImpactCategory,
    NotificationInput,
    PostToolUseInput,
    PreToolUseInput,
    SmellField,
    StopInput,
    SubagentStopInput,
    ZScoreInterpretation,
)


class TestEventTypeDetection:
    """Test the pattern match logic for detecting event types."""

    def test_pre_tool_use_input_detection(self):
        """Test that PreToolUseInput maps to PRE_TOOL_USE event type."""
        input_data = PreToolUseInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.PRE_TOOL_USE

    def test_post_tool_use_input_detection(self):
        """Test that PostToolUseInput maps to POST_TOOL_USE event type."""
        input_data = PostToolUseInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"success": True},
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.POST_TOOL_USE

    def test_notification_input_detection(self):
        """Test that NotificationInput maps to NOTIFICATION event type."""
        input_data = NotificationInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            message="Test notification",
            title="Test Title",
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.NOTIFICATION

    def test_stop_input_detection(self):
        """Test that StopInput maps to STOP event type."""
        input_data = StopInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            stop_hook_active=True,
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.STOP

    def test_subagent_stop_input_detection(self):
        """Test that SubagentStopInput maps to SUBAGENT_STOP event type."""
        input_data = SubagentStopInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            stop_hook_active=True,
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.SUBAGENT_STOP

    def test_all_input_types_are_handled(self):
        """Test that all defined input types have corresponding pattern matches.

        This test ensures we don't forget to update the pattern match when adding new input types.
        """
        input_types = [
            PreToolUseInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                tool_name="Test",
            ),
            PostToolUseInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                tool_name="Test",
                tool_response="success",
            ),
            NotificationInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                message="test",
            ),
            StopInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
            ),
            SubagentStopInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
            ),
        ]

        expected_types = [
            HookEventType.PRE_TOOL_USE,
            HookEventType.POST_TOOL_USE,
            HookEventType.NOTIFICATION,
            HookEventType.STOP,
            HookEventType.SUBAGENT_STOP,
        ]

        for input_data, expected_type in zip(input_types, expected_types):
            result = detect_event_type_from_parsed(input_data)
            assert result == expected_type, f"Input {type(input_data).__name__} should map to {expected_type}"


class TestExtractDevGuidelines:
    """Tests for extracting dev guidelines from CLAUDE.md."""

    def test_extract_dev_guidelines__returns_content_when_section_exists(self, tmp_path):
        """Test extraction when ## Development guidelines section exists."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("""# Project

## Setup
Some setup info

## Development guidelines

- Guideline 1
- Guideline 2

## Other section
Something else
""")

        result = extract_dev_guidelines_from_claude_md(str(tmp_path))

        assert "Guideline 1" in result
        assert "Guideline 2" in result
        assert "Other section" not in result

    def test_extract_dev_guidelines__returns_empty_when_no_claude_md(self, tmp_path):
        """Test returns empty string when CLAUDE.md doesn't exist."""
        result = extract_dev_guidelines_from_claude_md(str(tmp_path))
        assert result == ""

    def test_extract_dev_guidelines__returns_empty_when_section_missing(self, tmp_path):
        """Test returns empty string when section doesn't exist."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Project\n\nSome content")

        result = extract_dev_guidelines_from_claude_md(str(tmp_path))
        assert result == ""


class TestFormatCodeSmellFeedback:
    """Tests for code smell feedback formatting."""

    def test_format_code_smell_feedback__returns_empty_when_no_smells(self):
        """Test returns empty when no smells detected."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=0,
            average_complexity=0,
            total_volume=0,
            total_effort=0,
            total_difficulty=0,
            average_volume=0,
            average_effort=0,
            average_difficulty=0,
            total_mi=0,
            average_mi=0,
        )

        feedback, has_smells, has_blocking = format_code_smell_feedback(metrics, None)

        assert has_smells is False
        assert has_blocking is False
        assert feedback == ""

    def test_format_code_smell_feedback__includes_smell_when_count_nonzero(self):
        """Test that smells are included when count is non-zero."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=0,
            average_complexity=0,
            total_volume=0,
            total_effort=0,
            total_difficulty=0,
            average_volume=0,
            average_effort=0,
            average_difficulty=0,
            total_mi=0,
            average_mi=0,
            orphan_comment_count=5,
            orphan_comment_files=["src/foo.py"],
        )

        feedback, has_smells, has_blocking = format_code_smell_feedback(metrics, None)

        assert has_smells is True
        assert has_blocking is False  # Orphan comments are not blocking
        assert "Orphan Comments" in feedback
        assert "5" in feedback
        assert "not in edited files" in feedback  # Non-blocking smells show as summary

    def test_format_code_smell_feedback__includes_actionable_guidance(self):
        """Test that actionable guidance from SmellField is included."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=0,
            average_complexity=0,
            total_volume=0,
            total_effort=0,
            total_difficulty=0,
            average_volume=0,
            average_effort=0,
            average_difficulty=0,
            total_mi=0,
            average_mi=0,
            swallowed_exception_count=2,
            swallowed_exception_files=["src/bar.py"],
        )

        # Pass edited files to trigger blocking - smell file was edited
        feedback, has_smells, has_blocking = format_code_smell_feedback(metrics, None, edited_files={"src/bar.py"})

        assert has_smells is True
        assert has_blocking is True  # Swallowed exceptions ARE blocking when file was edited
        assert "Swallowed Exceptions" in feedback
        # Check that the guidance from SmellField is included
        assert "BLOCKING" in feedback
        assert "table" in feedback

    def test_format_code_smell_feedback__test_skips_are_blocking(self):
        """Test that test skips are marked as blocking when related file edited."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=0,
            average_complexity=0,
            total_volume=0,
            total_effort=0,
            total_difficulty=0,
            average_volume=0,
            average_effort=0,
            average_difficulty=0,
            total_mi=0,
            average_mi=0,
            test_skip_count=3,
            test_skip_files=["tests/test_foo.py"],
        )

        # Pass edited files - editing src/foo.py should trigger blocking for test_foo.py
        feedback, has_smells, has_blocking = format_code_smell_feedback(metrics, None, edited_files={"src/foo.py"})

        assert has_smells is True
        assert has_blocking is True  # Test skips ARE blocking when source file edited
        assert "Test Skips" in feedback

    def test_format_code_smell_feedback__not_blocking_when_unrelated_files(self):
        """Test that blocking smells don't block when files are unrelated to edits."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=0,
            average_complexity=0,
            total_volume=0,
            total_effort=0,
            total_difficulty=0,
            average_volume=0,
            average_effort=0,
            average_difficulty=0,
            total_mi=0,
            average_mi=0,
            test_skip_count=3,
            test_skip_files=["tests/test_foo.py"],
        )

        # Pass unrelated edited files - should NOT trigger blocking
        feedback, has_smells, has_blocking = format_code_smell_feedback(
            metrics, None, edited_files={"src/unrelated.py"}
        )

        assert has_smells is True
        assert has_blocking is False  # NOT blocking because files are unrelated
        assert "Test Skips" in feedback


class TestSmellField:
    """Tests for SmellField helper function."""

    def test_smell_field__creates_field_with_is_smell_marker(self):
        """Test that SmellField creates a field with is_smell=True in json_schema_extra."""
        field = SmellField(
            label="Test Smell",
            files_field="test_files",
            guidance="Fix this smell",
        )

        # Access the json_schema_extra
        extra = field.json_schema_extra
        assert extra is not None
        assert extra.get("is_smell") is True
        assert extra.get("label") == "Test Smell"
        assert extra.get("files_field") == "test_files"

    def test_smell_field__stores_guidance_in_description(self):
        """Test that SmellField stores guidance in the description field."""
        field = SmellField(
            label="Test Smell",
            files_field="test_files",
            guidance="This is the actionable guidance",
        )

        assert field.description == "This is the actionable guidance"


class TestZScoreInterpretation:
    """Tests for ZScoreInterpretation enum."""

    def test_from_z_score__much_better(self):
        """Test z-score > 1.0 is MUCH_BETTER (compact mode)."""
        result = ZScoreInterpretation.from_z_score(1.5)
        assert result == ZScoreInterpretation.MUCH_BETTER
        assert result.value == "much better than avg"

    def test_from_z_score__better(self):
        """Test z-score 0.3-1.0 is BETTER (compact mode)."""
        result = ZScoreInterpretation.from_z_score(0.5)
        assert result == ZScoreInterpretation.BETTER

    def test_from_z_score__about_avg(self):
        """Test z-score -0.3 to 0.3 is ABOUT_AVERAGE (compact mode)."""
        assert ZScoreInterpretation.from_z_score(0.0) == ZScoreInterpretation.ABOUT_AVERAGE
        assert ZScoreInterpretation.from_z_score(0.2) == ZScoreInterpretation.ABOUT_AVERAGE
        assert ZScoreInterpretation.from_z_score(-0.2) == ZScoreInterpretation.ABOUT_AVERAGE

    def test_from_z_score__worse(self):
        """Test z-score -1.0 to -0.3 is WORSE (compact mode)."""
        assert ZScoreInterpretation.from_z_score(-0.5) == ZScoreInterpretation.WORSE

    def test_from_z_score__much_worse(self):
        """Test z-score < -1.0 is MUCH_WORSE (compact mode)."""
        assert ZScoreInterpretation.from_z_score(-1.5) == ZScoreInterpretation.MUCH_WORSE

    def test_from_z_score__verbose_uses_wider_thresholds(self):
        """Test verbose mode uses 1.5/0.5 thresholds instead of 1.0/0.3."""
        # At z=1.2: compact=MUCH_BETTER, verbose=BETTER
        assert ZScoreInterpretation.from_z_score(1.2) == ZScoreInterpretation.MUCH_BETTER
        assert ZScoreInterpretation.from_z_score(1.2, verbose=True) == ZScoreInterpretation.BETTER

        # At z=0.4: compact=BETTER, verbose=ABOUT_AVERAGE
        assert ZScoreInterpretation.from_z_score(0.4) == ZScoreInterpretation.BETTER
        assert ZScoreInterpretation.from_z_score(0.4, verbose=True) == ZScoreInterpretation.ABOUT_AVERAGE


class TestImpactAssessmentInterpretation:
    """Tests for ImpactAssessment interpretation methods."""

    def _create_assessment(self, cc_z: float, effort_z: float, mi_z: float) -> ImpactAssessment:
        """Create a test assessment."""
        return ImpactAssessment(
            cc_z_score=cc_z,
            effort_z_score=effort_z,
            mi_z_score=mi_z,
            impact_score=0.0,
            impact_category=ImpactCategory.NEUTRAL,
            cc_delta=0.0,
            effort_delta=0.0,
            mi_delta=0.0,
        )

    def test_interpret_cc__inverts_zscore(self):
        """Test CC interpretation inverts z-score (lower CC is better)."""
        # Positive CC z-score = worse than avg (more complexity added)
        assessment = self._create_assessment(cc_z=1.5, effort_z=0.0, mi_z=0.0)
        assert assessment.interpret_cc() == ZScoreInterpretation.MUCH_WORSE

        # Negative CC z-score = better than avg (less complexity added)
        assessment = self._create_assessment(cc_z=-1.5, effort_z=0.0, mi_z=0.0)
        assert assessment.interpret_cc() == ZScoreInterpretation.MUCH_BETTER

    def test_interpret_effort__inverts_zscore(self):
        """Test Effort interpretation inverts z-score (lower effort is better)."""
        assessment = self._create_assessment(cc_z=0.0, effort_z=1.5, mi_z=0.0)
        assert assessment.interpret_effort() == ZScoreInterpretation.MUCH_WORSE

    def test_interpret_mi__does_not_invert(self):
        """Test MI interpretation does not invert (higher MI is better)."""
        assessment = self._create_assessment(cc_z=0.0, effort_z=0.0, mi_z=1.5)
        assert assessment.interpret_mi() == ZScoreInterpretation.MUCH_BETTER


class TestFormatContextCoverageFeedback:
    """Tests for context coverage feedback formatting."""

    def test_format_context_coverage_feedback__shows_unread_tests(self):
        """Test that unread test files are listed with guidance."""
        coverage = ContextCoverage(
            files_edited=["src/foo.py"],
            files_read=["src/foo.py"],
            file_coverage=[
                FileCoverageStatus(
                    file_path="src/foo.py",
                    was_read_before_edit=True,
                    test_files=["tests/test_foo.py", "tests/test_bar.py"],
                    test_files_read=[],  # None read
                )
            ],
        )

        result = format_context_coverage_feedback(coverage)

        assert "RELATED Tests - MUST REVIEW" in result
        assert "tests/test_foo.py" in result
        assert "correspond to files you edited" in result

    def test_format_context_coverage_feedback__excludes_read_tests(self):
        """Test that read test files are not listed as unreviewed."""
        coverage = ContextCoverage(
            files_edited=["src/foo.py"],
            files_read=["src/foo.py", "tests/test_foo.py"],
            file_coverage=[
                FileCoverageStatus(
                    file_path="src/foo.py",
                    was_read_before_edit=True,
                    test_files=["tests/test_foo.py"],
                    test_files_read=["tests/test_foo.py"],  # Was read
                )
            ],
        )

        result = format_context_coverage_feedback(coverage)

        assert "RELATED Tests - MUST REVIEW" not in result

    def test_format_context_coverage_feedback__no_tests_section_when_empty(self):
        """Test that no tests section appears when all tests were read."""
        coverage = ContextCoverage(
            files_edited=["src/foo.py"],
            files_read=["src/foo.py"],
            file_coverage=[
                FileCoverageStatus(
                    file_path="src/foo.py",
                    was_read_before_edit=True,
                    test_files=[],  # No related tests
                    test_files_read=[],
                )
            ],
        )

        result = format_context_coverage_feedback(coverage)

        assert "RELATED Tests - MUST REVIEW" not in result


class TestFormattersInterpretZScore:
    """Tests for _interpret_z_score in formatters.py."""

    def test_interpret_z_score__uses_verbose_mode(self):
        """Test that _interpret_z_score uses verbose mode thresholds."""
        from slopometry.display.formatters import _interpret_z_score

        # At z=1.2: compact would be "much better", verbose is "better"
        result = _interpret_z_score(1.2)
        assert result == "better than avg"

    def test_interpret_z_score__returns_string_value(self):
        """Test that _interpret_z_score returns string, not enum."""
        from slopometry.display.formatters import _interpret_z_score

        result = _interpret_z_score(2.0)
        assert isinstance(result, str)
        assert result == "much better than avg"

    def test_interpret_z_score__negative_is_worse(self):
        """Test negative z-scores are interpreted as worse."""
        from slopometry.display.formatters import _interpret_z_score

        assert _interpret_z_score(-2.0) == "much worse than avg"
        assert _interpret_z_score(-0.8) == "worse than avg"
        assert _interpret_z_score(0.0) == "about avg"

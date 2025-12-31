"""Tests for hook handler functionality."""

import subprocess
import tempfile
from pathlib import Path

from slopometry.core.hook_handler import (
    _get_related_files_via_imports,
    detect_event_type_from_parsed,
    extract_dev_guidelines_from_claude_md,
    format_code_smell_feedback,
    format_context_coverage_feedback,
    parse_hook_input,
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


def test_parse_hook_input__stop_hook_active_true_returns_subagent_stop():
    """Test that stop_hook_active=true is parsed as SubagentStopInput."""
    raw_data = {
        "session_id": "test-session",
        "transcript_path": "/tmp/test.jsonl",
        "stop_hook_active": True,
    }

    result = parse_hook_input(raw_data)

    assert isinstance(result, SubagentStopInput)


def test_parse_hook_input__stop_hook_active_false_returns_stop():
    """Test that stop_hook_active=false is parsed as StopInput."""
    raw_data = {
        "session_id": "test-session",
        "transcript_path": "/tmp/test.jsonl",
        "stop_hook_active": False,
    }

    result = parse_hook_input(raw_data)

    assert isinstance(result, StopInput)


def test_parse_hook_input__stop_hook_active_omitted_returns_stop():
    """Test that missing stop_hook_active is parsed as StopInput."""
    raw_data = {
        "session_id": "test-session",
        "transcript_path": "/tmp/test.jsonl",
    }

    result = parse_hook_input(raw_data)

    assert isinstance(result, StopInput)


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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "bar.py").write_text("def bar(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

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

            feedback, has_smells, has_blocking = format_code_smell_feedback(
                metrics, None, edited_files={"src/bar.py"}, working_directory=str(tmppath)
            )

            assert has_smells is True
            assert has_blocking is True
            assert "Swallowed Exceptions" in feedback
            assert "BLOCKING" in feedback
            assert "table" in feedback

    def test_format_code_smell_feedback__test_skips_are_blocking(self):
        """Test that test skips are marked as blocking when related file edited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            tests_dir = tmppath / "tests"
            tests_dir.mkdir()
            (src_dir / "foo.py").write_text("def foo(): pass")
            (tests_dir / "test_foo.py").write_text("def test_foo(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

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

            feedback, has_smells, has_blocking = format_code_smell_feedback(
                metrics, None, edited_files={"src/foo.py"}, working_directory=str(tmppath)
            )

            assert has_smells is True
            assert has_blocking is True
            assert "Test Skips" in feedback

    def test_format_code_smell_feedback__not_blocking_when_unrelated_files(self):
        """Test that blocking smells don't block when files are unrelated to edits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            tests_dir = tmppath / "tests"
            tests_dir.mkdir()
            (src_dir / "unrelated.py").write_text("def unrelated(): pass")
            (tests_dir / "test_foo.py").write_text("def test_foo(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

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

            feedback, has_smells, has_blocking = format_code_smell_feedback(
                metrics, None, edited_files={"src/unrelated.py"}, working_directory=str(tmppath)
            )

            assert has_smells is True
            assert has_blocking is False
            assert "Test Skips" in feedback

    def test_format_code_smell_feedback__splits_related_and_unrelated_files(self):
        """Test that smells are split between related (blocking) and unrelated files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "foo.py").write_text("def foo(): pass")
            (src_dir / "bar.py").write_text("def bar(): pass")
            (src_dir / "baz.py").write_text("def baz(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

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
                swallowed_exception_count=3,
                swallowed_exception_files=["src/foo.py", "src/bar.py", "src/baz.py"],
            )

            feedback, has_smells, has_blocking = format_code_smell_feedback(
                metrics, None, edited_files={"src/bar.py"}, working_directory=str(tmppath)
            )

            assert has_smells is True
            assert has_blocking is True
            assert "ACTION REQUIRED" in feedback
            assert "bar.py" in feedback
            assert "not in edited files" in feedback
            assert "Swallowed Exceptions: 2" in feedback

    def test_format_code_smell_feedback__related_test_file_triggers_blocking(self):
        """Test that editing a source file makes its test file's smells blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            tests_dir = tmppath / "tests"
            tests_dir.mkdir()
            (src_dir / "foo.py").write_text("def foo(): pass")
            (tests_dir / "test_foo.py").write_text("def test_foo(): pass")
            (tests_dir / "test_bar.py").write_text("def test_bar(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

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
                test_skip_count=2,
                test_skip_files=["tests/test_foo.py", "tests/test_bar.py"],
            )

            feedback, has_smells, has_blocking = format_code_smell_feedback(
                metrics, None, edited_files={"src/foo.py"}, working_directory=str(tmppath)
            )

            assert has_smells is True
            assert has_blocking is True
            assert "test_foo.py" in feedback
            assert "not in edited files" in feedback


class TestGetRelatedFilesViaImports:
    """Tests for import graph-based file relationship detection."""

    def test_get_related_files_via_imports__finds_dependents_not_imports(self):
        """Test that files importing edited files are found, but not files that edited files import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Initialize git repo (required for git_tracker)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            # Create src directory with modules
            src_dir = tmppath / "src"
            src_dir.mkdir()

            # core.py - a dependency
            (src_dir / "core.py").write_text("def core_func(): pass")

            # service.py - the file we'll edit (imports core.py)
            (src_dir / "service.py").write_text("from src.core import core_func\ndef service_func(): pass")

            # handler.py - imports service.py (is a dependent)
            (src_dir / "handler.py").write_text("from src.service import service_func")

            # unrelated.py - doesn't import service.py
            (src_dir / "unrelated.py").write_text("def other(): pass")

            # Track files in git
            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            # Get related files for editing service.py
            edited = {"src/service.py"}
            related = _get_related_files_via_imports(edited, str(tmppath))

            # Should include the edited file
            assert "src/service.py" in related
            # Should include handler.py (imports service.py - is a dependent)
            assert "src/handler.py" in related
            # Should NOT include core.py (service.py imports it, but changes don't flow upstream)
            assert "src/core.py" not in related
            # Should NOT include unrelated.py
            assert "src/unrelated.py" not in related

    def test_get_related_files_via_imports__finds_test_files(self):
        """Test that test files for edited modules are found as related."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            # Create src and tests directories
            src_dir = tmppath / "src"
            src_dir.mkdir()
            tests_dir = tmppath / "tests"
            tests_dir.mkdir()

            (src_dir / "core.py").write_text("def core_func(): pass")
            (tests_dir / "test_core.py").write_text("def test_core(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            edited = {"src/core.py"}
            related = _get_related_files_via_imports(edited, str(tmppath))

            assert "src/core.py" in related
            assert "tests/test_core.py" in related


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

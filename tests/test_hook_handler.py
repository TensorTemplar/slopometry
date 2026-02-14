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
    scope_smells_for_session,
)
from slopometry.core.models.baseline import ImpactAssessment, ImpactCategory, ZScoreInterpretation
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import (
    HookEventType,
    NotificationInput,
    PostToolUseInput,
    PreToolUseInput,
    StopInput,
    SubagentStopInput,
)
from slopometry.core.models.session import ContextCoverage, FileCoverageStatus
from slopometry.core.models.smell import SmellField


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

    def _make_metrics(self, **kwargs) -> ExtendedComplexityMetrics:
        """Create metrics with sensible defaults."""
        defaults = dict(
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
        defaults.update(kwargs)
        return ExtendedComplexityMetrics(**defaults)

    def test_format_code_smell_feedback__returns_empty_when_no_smells(self):
        """Test returns empty when no smells detected."""
        metrics = self._make_metrics()
        scoped = scope_smells_for_session(metrics, None, set(), "/tmp")

        feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

        assert has_smells is False
        assert has_blocking is False
        assert feedback == ""

    def test_format_code_smell_feedback__includes_smell_when_count_nonzero(self):
        """Test that non-blocking smells only show when there are changes (deltas)."""
        metrics = self._make_metrics(
            orphan_comment_count=5,
            orphan_comment_files=["src/foo.py"],
        )

        # Without delta, non-blocking smells don't show (no changes to report)
        scoped = scope_smells_for_session(metrics, None, set(), "/tmp")
        feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)
        assert has_smells is False
        assert has_blocking is False
        assert feedback == ""

        # With a delta showing changes, non-blocking smells are shown
        delta = ComplexityDelta(
            orphan_comment_change=2,  # New orphan comments added
        )
        scoped = scope_smells_for_session(metrics, delta, set(), "/tmp")
        feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)
        assert has_smells is True
        assert has_blocking is False
        assert "Orphan Comments" in feedback
        assert "(+2)" in feedback
        assert "Code Smells" in feedback
        assert "src/foo.py" in feedback

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

            metrics = self._make_metrics(
                swallowed_exception_count=2,
                swallowed_exception_files=["src/bar.py"],
            )

            scoped = scope_smells_for_session(metrics, None, {"src/bar.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

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

            metrics = self._make_metrics(
                test_skip_count=3,
                test_skip_files=["tests/test_foo.py"],
            )

            scoped = scope_smells_for_session(metrics, None, {"src/foo.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is True
            assert has_blocking is True
            assert "Test Skips" in feedback

    def test_format_code_smell_feedback__not_blocking_when_unrelated_files(self):
        """Test that blocking smells don't block when files are unrelated to edits.

        Note: When a blocking smell (test_skip, swallowed_exception) is found in files
        unrelated to edits, it's added to other_smells with change=0 because we can't
        attribute the global change to specific files. This means unrelated blocking
        smells won't show in the "non-edited files" summary (which only shows changes).
        """
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

            metrics = self._make_metrics(
                test_skip_count=3,
                test_skip_files=["tests/test_foo.py"],
            )

            # When blocking smells are in unrelated files, they're not blocking
            # and don't show in the summary (no changes to report for unrelated split)
            scoped = scope_smells_for_session(metrics, None, {"src/unrelated.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is False
            assert has_blocking is False
            assert feedback == ""

            # Even with a delta, unrelated blocking smells don't show because
            # changes can't be attributed to the unrelated portion
            delta = ComplexityDelta(test_skip_change=1)
            scoped = scope_smells_for_session(metrics, delta, {"src/unrelated.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is False
            assert has_blocking is False
            assert feedback == ""

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

            metrics = self._make_metrics(
                swallowed_exception_count=3,
                swallowed_exception_files=["src/foo.py", "src/bar.py", "src/baz.py"],
            )

            scoped = scope_smells_for_session(metrics, None, {"src/bar.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is True
            assert has_blocking is True
            assert "ACTION REQUIRED" in feedback
            assert "bar.py" in feedback
            # Non-blocking smells (foo.py, baz.py) only show when there are changes
            # Without a delta, only blocking smells in edited files are shown

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

            metrics = self._make_metrics(
                test_skip_count=2,
                test_skip_files=["tests/test_foo.py", "tests/test_bar.py"],
            )

            scoped = scope_smells_for_session(metrics, None, {"src/foo.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is True
            assert has_blocking is True
            assert "test_foo.py" in feedback
            # test_bar.py (unrelated) only shows in non-edited files section when there are changes

    def test_format_code_smell_feedback__unread_tests_are_blocking(self):
        """Test that unread related tests trigger blocking when context_coverage provided."""
        metrics = self._make_metrics()

        context_coverage = ContextCoverage(
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

        scoped = scope_smells_for_session(metrics, None, set(), "/tmp", context_coverage=context_coverage)
        feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

        assert has_smells is True
        assert has_blocking is True
        assert "Unread Related Tests" in feedback
        assert "BLOCKING" in feedback
        assert "tests/test_foo.py" in feedback

    def test_format_code_smell_feedback__read_tests_not_blocking(self):
        """Test that read tests are not included in unread tests blocking."""
        metrics = self._make_metrics()

        context_coverage = ContextCoverage(
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

        scoped = scope_smells_for_session(metrics, None, set(), "/tmp", context_coverage=context_coverage)
        feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

        assert has_smells is False
        assert has_blocking is False
        assert "Unread Related Tests" not in feedback

    def test_format_code_smell_feedback__non_blocking_smells_only_list_edited_files(self):
        """Test that non-blocking smell file lists are filtered to edited + test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "edited.py").write_text("def edited(): pass")
            (src_dir / "unrelated.py").write_text("def unrelated(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            metrics = self._make_metrics(
                inline_import_count=10,
                inline_import_files=["src/edited.py", "src/unrelated.py", "src/other.py"],
            )

            delta = ComplexityDelta(inline_import_change=3)
            scoped = scope_smells_for_session(metrics, delta, {"src/edited.py"}, str(tmppath))
            feedback, has_smells, has_blocking = format_code_smell_feedback(scoped)

            assert has_smells is True
            assert has_blocking is False
            # Total count is repo-level
            assert "10" in feedback
            assert "(+3)" in feedback
            # Only edited file is listed, not unrelated ones
            assert "edited.py" in feedback
            assert "unrelated.py" not in feedback
            assert "other.py" not in feedback


class TestScopeSmellsForSession:
    """Tests for scope_smells_for_session classification logic."""

    def _make_metrics(self, **kwargs) -> ExtendedComplexityMetrics:
        """Create metrics with sensible defaults."""
        defaults = dict(
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
        defaults.update(kwargs)
        return ExtendedComplexityMetrics(**defaults)

    def test_scope_smells_for_session__returns_empty_when_no_smells(self):
        """Test returns empty list when metrics have no smells."""
        metrics = self._make_metrics()
        result = scope_smells_for_session(metrics, None, set(), "/tmp")
        assert result == []

    def test_scope_smells_for_session__classifies_swallowed_exception_as_blocking_when_in_edited_files(self):
        """Test that swallowed_exception in edited files is classified as blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "foo.py").write_text("def foo(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            metrics = self._make_metrics(
                swallowed_exception_count=1,
                swallowed_exception_files=["src/foo.py"],
            )

            result = scope_smells_for_session(metrics, None, {"src/foo.py"}, str(tmppath))

            blocking = [s for s in result if s.is_blocking]
            assert len(blocking) == 1
            assert blocking[0].name == "swallowed_exception"
            assert blocking[0].actionable_files == ["src/foo.py"]

    def test_scope_smells_for_session__classifies_swallowed_exception_as_non_blocking_when_unrelated(self):
        """Test that swallowed_exception in unrelated files is non-blocking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "foo.py").write_text("def foo(): pass")
            (src_dir / "bar.py").write_text("def bar(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            metrics = self._make_metrics(
                swallowed_exception_count=1,
                swallowed_exception_files=["src/foo.py"],
            )

            result = scope_smells_for_session(metrics, None, {"src/bar.py"}, str(tmppath))

            blocking = [s for s in result if s.is_blocking]
            assert len(blocking) == 0
            non_blocking = [s for s in result if s.name == "swallowed_exception"]
            assert len(non_blocking) == 1
            assert non_blocking[0].is_blocking is False

    def test_scope_smells_for_session__splits_blocking_smell_files_between_related_and_unrelated(self):
        """Test that a blocking smell with mixed files produces two ScopedSmells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "edited.py").write_text("def edited(): pass")
            (src_dir / "other.py").write_text("def other(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            metrics = self._make_metrics(
                swallowed_exception_count=2,
                swallowed_exception_files=["src/edited.py", "src/other.py"],
            )

            result = scope_smells_for_session(metrics, None, {"src/edited.py"}, str(tmppath))

            swallowed = [s for s in result if s.name == "swallowed_exception"]
            assert len(swallowed) == 2
            blocking = [s for s in swallowed if s.is_blocking]
            non_blocking = [s for s in swallowed if not s.is_blocking]
            assert len(blocking) == 1
            assert blocking[0].actionable_files == ["src/edited.py"]
            assert len(non_blocking) == 1
            assert non_blocking[0].actionable_files == ["src/other.py"]

    def test_scope_smells_for_session__non_blocking_smell_preserves_repo_count_and_change(self):
        """Test that non-blocking smells keep repo-level count and delta."""
        metrics = self._make_metrics(
            orphan_comment_count=5,
            orphan_comment_files=["src/a.py", "src/b.py"],
        )
        delta = ComplexityDelta(orphan_comment_change=2)

        result = scope_smells_for_session(metrics, delta, set(), "/tmp")

        orphan = [s for s in result if s.name == "orphan_comment"]
        assert len(orphan) == 1
        assert orphan[0].count == 5
        assert orphan[0].change == 2
        assert orphan[0].is_blocking is False

    def test_scope_smells_for_session__unread_tests_produce_synthetic_blocking_smell(self):
        """Test that unread related tests from context_coverage produce a blocking ScopedSmell."""
        metrics = self._make_metrics()
        context_coverage = ContextCoverage(
            files_edited=["src/foo.py"],
            files_read=["src/foo.py"],
            file_coverage=[
                FileCoverageStatus(
                    file_path="src/foo.py",
                    was_read_before_edit=True,
                    test_files=["tests/test_foo.py"],
                    test_files_read=[],
                )
            ],
        )

        result = scope_smells_for_session(metrics, None, set(), "/tmp", context_coverage=context_coverage)

        blocking = [s for s in result if s.is_blocking]
        assert len(blocking) == 1
        assert blocking[0].name == "unread_related_tests"
        assert blocking[0].actionable_files == ["tests/test_foo.py"]

    def test_scope_smells_for_session__filters_actionable_files_for_non_blocking_with_edits(self):
        """Test that non-blocking smells only list actionable files when edited_files is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()
            (src_dir / "edited.py").write_text("def edited(): pass")
            (src_dir / "other.py").write_text("def other(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            metrics = self._make_metrics(
                orphan_comment_count=3,
                orphan_comment_files=["src/edited.py", "src/other.py"],
            )

            result = scope_smells_for_session(metrics, None, {"src/edited.py"}, str(tmppath))

            orphan = [s for s in result if s.name == "orphan_comment"]
            assert len(orphan) == 1
            assert orphan[0].actionable_files == ["src/edited.py"]


class TestGetRelatedFilesViaImports:
    """Tests for import graph-based file relationship detection."""

    def test_get_related_files_via_imports__only_includes_edited_and_test_files(self):
        """Test that only edited files and their test files are related, not import dependents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            subprocess.run(["git", "init"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmppath, capture_output=True)

            src_dir = tmppath / "src"
            src_dir.mkdir()

            # core.py - a dependency of service.py
            (src_dir / "core.py").write_text("def core_func(): pass")

            # service.py - the file we'll edit (imports core.py)
            (src_dir / "service.py").write_text("from src.core import core_func\ndef service_func(): pass")

            # handler.py - imports service.py (is a dependent, but NOT edited)
            (src_dir / "handler.py").write_text("from src.service import service_func")

            # unrelated.py - doesn't import service.py
            (src_dir / "unrelated.py").write_text("def other(): pass")

            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            edited = {"src/service.py"}
            related = _get_related_files_via_imports(edited, str(tmppath))

            # Should include the edited file
            assert "src/service.py" in related
            # Should NOT include handler.py (imports service.py but wasn't edited)
            assert "src/handler.py" not in related
            # Should NOT include core.py (service.py imports it)
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

    def test_format_context_coverage_feedback__no_tests_section(self):
        """Test that context coverage no longer includes tests section (moved to smell feedback)."""
        coverage = ContextCoverage(
            files_edited=["src/foo.py"],
            files_read=["src/foo.py"],
            file_coverage=[
                FileCoverageStatus(
                    file_path="src/foo.py",
                    was_read_before_edit=True,
                    test_files=["tests/test_foo.py"],
                    test_files_read=[],
                )
            ],
        )

        result = format_context_coverage_feedback(coverage)

        # Unread tests are now handled by format_code_smell_feedback, not here
        assert "RELATED Tests" not in result
        assert "Unread Related Tests" not in result


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


class TestHookHandlerSmokeTests:
    """Smoke tests to ensure hook handlers don't crash with valid input."""

    def _init_git_repo(self, path: Path) -> None:
        """Initialize a git repo for testing."""
        subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "--local", "user.email", "test@example.com"],
            cwd=path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "--local", "user.name", "Test"],
            cwd=path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "--local", "commit.gpgsign", "false"],
            cwd=path,
            capture_output=True,
            check=True,
        )

    def test_handle_hook__pre_tool_use_does_not_crash(self):
        """Smoke test: PreToolUse hook should not crash."""
        import json
        from io import StringIO
        from unittest.mock import patch

        from slopometry.core.hook_handler import handle_hook
        from slopometry.core.models.hook import HookEventType

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }

        with patch("sys.stdin", StringIO(json.dumps(input_data))):
            result = handle_hook(event_type_override=HookEventType.PRE_TOOL_USE)

        assert result == 0

    def test_handle_hook__post_tool_use_does_not_crash(self):
        """Smoke test: PostToolUse hook should not crash."""
        import json
        from io import StringIO
        from unittest.mock import patch

        from slopometry.core.hook_handler import handle_hook
        from slopometry.core.models.hook import HookEventType

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt",
        }

        with patch("sys.stdin", StringIO(json.dumps(input_data))):
            result = handle_hook(event_type_override=HookEventType.POST_TOOL_USE)

        assert result == 0

    def test_handle_hook__notification_does_not_crash(self):
        """Smoke test: Notification hook should not crash."""
        import json
        from io import StringIO
        from unittest.mock import patch

        from slopometry.core.hook_handler import handle_hook
        from slopometry.core.models.hook import HookEventType

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "message": "Test notification",
        }

        with patch("sys.stdin", StringIO(json.dumps(input_data))):
            result = handle_hook(event_type_override=HookEventType.NOTIFICATION)

        assert result == 0

    def test_handle_hook__stop_does_not_crash(self):
        """Smoke test: Stop hook should not crash."""
        import json
        from io import StringIO
        from unittest.mock import patch

        from slopometry.core.hook_handler import handle_hook
        from slopometry.core.models.hook import HookEventType

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            self._init_git_repo(tmppath)
            (tmppath / "test.py").write_text("x = 1")
            subprocess.run(["git", "add", "."], cwd=tmppath, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, capture_output=True)

            input_data = {
                "session_id": "smoke-test-stop",
                "transcript_path": "/tmp/test.jsonl",
                "stop_hook_active": False,
            }

            with (
                patch("sys.stdin", StringIO(json.dumps(input_data))),
                patch("os.getcwd", return_value=str(tmppath)),
            ):
                result = handle_hook(event_type_override=HookEventType.STOP)

            # Stop hook returns 0 (no feedback) or 2 (with feedback) - both are valid
            assert result in (0, 2)

    def test_handle_hook__subagent_stop_does_not_crash(self):
        """Smoke test: SubagentStop hook should not crash and return 0."""
        import json
        from io import StringIO
        from unittest.mock import patch

        from slopometry.core.hook_handler import handle_hook
        from slopometry.core.models.hook import HookEventType

        input_data = {
            "session_id": "smoke-test-subagent",
            "transcript_path": "/tmp/test.jsonl",
            "stop_hook_active": True,
        }

        with patch("sys.stdin", StringIO(json.dumps(input_data))):
            result = handle_hook(event_type_override=HookEventType.STOP)

        # Subagent stops should return 0 (no feedback for subagents)
        assert result == 0

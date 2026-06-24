"""Tests for hook handler functionality — feedback pipeline, smoke tests, working-tree probes.

Pure handler-level tests. Detailed wire-format parsing/detection lives in
`test_claude_code_adapter.py` since the wire-validation models (PreToolUseInput,
PostToolUseInput, NotificationInput, StopInput, SubagentStopInput) have been
replaced by `ClaudeCodeAdapter.parse()` / `ClaudeCodeAdapter.detect_event_type()`.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from slopometry.core.database import SessionManager
from slopometry.core.hook_handler import (
    _get_related_files_via_imports,
    _has_analyzable_source_files,
    _has_source_changes,
    _resolve_working_directory,
    extract_dev_guidelines_from_claude_md,
    format_code_smell_feedback,
    format_context_coverage_feedback,
    handle_hook,
    handle_stop_event,
    scope_smells_for_session,
)
from slopometry.core.models.baseline import ImpactAssessment, ImpactCategory, ZScoreInterpretation
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import FeedbackCacheState
from slopometry.core.models.protocol.events import AbstractEventType
from slopometry.core.models.session import ContextCoverage, FileCoverageStatus
from slopometry.core.models.smell import SmellField
from slopometry.display.formatters import _interpret_z_score


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
        defaults: dict[str, object] = {
            "total_complexity": 0,
            "average_complexity": 0.0,
            "total_volume": 0.0,
            "total_effort": 0.0,
            "total_difficulty": 0.0,
            "average_volume": 0.0,
            "average_effort": 0.0,
            "average_difficulty": 0.0,
            "total_mi": 0.0,
            "average_mi": 0.0,
        }
        defaults.update(kwargs)
        return ExtendedComplexityMetrics.model_validate(defaults)

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

    def test_scope_smells__acknowledged_silent_blocks_on_increase(self):
        """A rising acknowledged_silent_except count is blocking in edited files."""
        metrics = self._make_metrics(
            acknowledged_silent_except_count=3,
            acknowledged_silent_except_files=["src/foo.py"],
        )
        delta = ComplexityDelta(acknowledged_silent_except_change=2)
        scoped = scope_smells_for_session(metrics, delta, {"src/foo.py"}, "/tmp")

        ack = [s for s in scoped if s.name == "acknowledged_silent_except"]
        assert ack, "acknowledged_silent_except should be scoped"
        assert any(s.is_blocking for s in ack), "an increase in edited files must block"

    def test_scope_smells__acknowledged_silent_unchanged_does_not_block(self):
        """A steady acknowledged_silent_except count is informational, not blocking."""
        metrics = self._make_metrics(
            acknowledged_silent_except_count=3,
            acknowledged_silent_except_files=["src/foo.py"],
        )
        delta = ComplexityDelta(acknowledged_silent_except_change=0)
        scoped = scope_smells_for_session(metrics, delta, {"src/foo.py"}, "/tmp")

        ack = [s for s in scoped if s.name == "acknowledged_silent_except"]
        assert not any(s.is_blocking for s in ack), "no increase => not blocking"

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

    def test_format_code_smell_feedback__swallow_hint_shown_when_swallowed_blocking(self):
        """Concrete marker comment format appears after ACTION REQUIRED when swallow-related smell blocks."""
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
                swallowed_exception_count=1,
                swallowed_exception_files=["src/bar.py"],
            )
            scoped = scope_smells_for_session(metrics, None, {"src/bar.py"}, str(tmppath))
            feedback, _, _ = format_code_smell_feedback(scoped)

            assert "# slopometry: allow-silent" in feedback
            assert "lock already released on context exit" in feedback
            assert "**To acknowledge after review**" in feedback
            assert "Place `# slopometry: allow-silent - <short reason>`" in feedback

    def test_format_code_smell_feedback__swallow_hint_shown_when_acknowledged_increased(self):
        """Hint also appears when acknowledged_silent_except increases (potential mass-suppression)."""
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
                acknowledged_silent_except_count=3,
                acknowledged_silent_except_files=["src/bar.py"],
            )
            delta = ComplexityDelta(acknowledged_silent_except_change=2)
            scoped = scope_smells_for_session(metrics, delta, {"src/bar.py"}, str(tmppath))
            feedback, _, _ = format_code_smell_feedback(scoped)

            assert "# slopometry: allow-silent" in feedback
            assert "lock already released on context exit" in feedback

    def test_format_code_smell_feedback__swallow_hint_absent_when_no_swallow_smell(self):
        """Hint does NOT appear when a non-swallow smell is blocking."""
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
                test_skip_count=1,
                test_skip_files=["src/bar.py"],
            )
            scoped = scope_smells_for_session(metrics, None, {"src/bar.py"}, str(tmppath))
            feedback, _, _ = format_code_smell_feedback(scoped)

            assert "**To acknowledge after review**" not in feedback
            assert "lock already released on context exit" not in feedback

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
        defaults: dict[str, object] = {
            "total_complexity": 0,
            "average_complexity": 0.0,
            "total_volume": 0.0,
            "total_effort": 0.0,
            "total_difficulty": 0.0,
            "average_volume": 0.0,
            "average_effort": 0.0,
            "average_difficulty": 0.0,
            "total_mi": 0.0,
            "average_mi": 0.0,
        }
        defaults.update(kwargs)
        return ExtendedComplexityMetrics.model_validate(defaults)

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

        # At z=1.2: compact would be "much better", verbose is "better"
        result = _interpret_z_score(1.2)
        assert result == "better than avg"

    def test_interpret_z_score__returns_string_value(self):
        """Test that _interpret_z_score returns string, not enum."""

        result = _interpret_z_score(2.0)
        assert isinstance(result, str)
        assert result == "much better than avg"

    def test_interpret_z_score__negative_is_worse(self):
        """Test negative z-scores are interpreted as worse."""

        assert _interpret_z_score(-2.0) == "much worse than avg"
        assert _interpret_z_score(-0.8) == "worse than avg"
        assert _interpret_z_score(0.0) == "about avg"


class TestHookHandlerSmokeTests:
    """Smoke tests to ensure hook handlers don't crash with valid input.

    These tests patch _read_stdin_with_timeout instead of sys.stdin because
    the real implementation uses select.select() which requires a real file descriptor.
    """

    @pytest.fixture(autouse=True)
    def _isolate_db(self, tmp_path):
        """Redirect database and session state to temp directories so smoke tests don't pollute the real DB."""
        db_path = tmp_path / "test.db"
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        original_init = SessionManager.__init__

        def _isolated_init(self_inner, source: str, state_root=None):
            original_init(self_inner, source, state_root=state_dir)
            self_inner.state_dir = state_dir

        with (
            patch("slopometry.core.settings.settings.database_path", db_path),
            patch.object(SessionManager, "__init__", _isolated_init),
        ):
            yield

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

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }

        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_hook(event_type_override=AbstractEventType.TOOL_CALL_STARTED)

        assert result == 0

    def test_handle_hook__post_tool_use_does_not_crash(self):
        """Smoke test: PostToolUse hook should not crash."""

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_response": "file1.txt\nfile2.txt",
        }

        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_hook(event_type_override=AbstractEventType.TOOL_CALL_COMPLETED)

        assert result == 0

    def test_handle_hook__notification_does_not_crash(self):
        """Smoke test: Notification hook should not crash."""

        input_data = {
            "session_id": "smoke-test-session",
            "transcript_path": "/tmp/test.jsonl",
            "message": "Test notification",
        }

        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_hook(event_type_override=AbstractEventType.NOTIFICATION)

        assert result == 0

    def test_handle_hook__stop_does_not_crash(self):
        """Smoke test: Stop hook should not crash."""

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
                patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)),
                patch("os.getcwd", return_value=str(tmppath)),
            ):
                result = handle_hook(event_type_override=AbstractEventType.TURN_COMPLETED)

            # Stop hook returns 0 (no feedback) or 2 (with feedback) - both are valid
            assert result in (0, 2)

    def test_handle_hook__subagent_stop_does_not_crash(self):
        """Smoke test: SubagentStop hook should not crash and return 0."""

        input_data = {
            "session_id": "smoke-test-subagent",
            "transcript_path": "/tmp/test.jsonl",
            "stop_hook_active": True,
        }

        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=json.dumps(input_data)):
            result = handle_hook(event_type_override=AbstractEventType.TURN_COMPLETED)

        # Subagent stops should return 0 (no feedback for subagents)
        assert result == 0

    def test_handle_hook__empty_stdin_returns_zero(self):
        """Test that empty stdin (timeout) returns 0 without errors."""
        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value=""):
            result = handle_hook()

        assert result == 0

    def test_handle_hook__invalid_json_returns_zero(self):
        """Test that invalid JSON input returns 0 without crashing."""
        with patch("slopometry.core.hook_handler._read_stdin_with_timeout", return_value="not valid json"):
            result = handle_hook()

        assert result == 0


class TestHasAnalyzableSourceFiles:
    """Tests for the _has_analyzable_source_files early-exit gate."""

    def test_has_analyzable_source_files__returns_true_for_python_repo(self, tmp_path):
        """Returns True when git repo contains .py files."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.py").write_text("x = 1")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        assert _has_analyzable_source_files(str(tmp_path)) is True

    def test_has_analyzable_source_files__returns_true_for_rust_repo(self, tmp_path):
        """Returns True when git repo contains .rs files."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.rs").write_text("fn main() {}")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        assert _has_analyzable_source_files(str(tmp_path)) is True

    def test_has_analyzable_source_files__returns_false_for_non_code_repo(self, tmp_path):
        """Returns False when git repo has no .py or .rs files (e.g. hardware project)."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
        (tmp_path / "schematic.kicad_sch").write_text("(kicad_sch ...)")
        (tmp_path / "README.md").write_text("# Hardware project")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        assert _has_analyzable_source_files(str(tmp_path)) is False

    def test_has_analyzable_source_files__returns_false_for_non_git_dir(self, tmp_path):
        """Returns False for non-git directories (no rglob fallback)."""
        (tmp_path / "main.py").write_text("x = 1")  # Has Python, but not a git repo

        assert _has_analyzable_source_files(str(tmp_path)) is False

    def test_has_analyzable_source_files__ignores_submodule_only_sources(self, tmp_path):
        """Returns False when the only .py/.rs files live inside a submodule.

        A parent repo with no source files of its own but a submodule containing
        source files must not trigger the stop-hook analysis path — that code
        belongs to the submodule's repository, not the parent project.
        """
        sub = tmp_path / "subrepo"
        sub.mkdir()
        subprocess.run(["git", "init"], cwd=sub, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=sub, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=sub, capture_output=True)
        (sub / "sub.py").write_text("def sub(): pass")
        subprocess.run(["git", "add", "."], cwd=sub, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=sub, capture_output=True)

        main = tmp_path / "main"
        main.mkdir()
        subprocess.run(["git", "init"], cwd=main, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=main, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=main, capture_output=True)
        (main / "README.md").write_text("# docs only")
        subprocess.run(["git", "add", "."], cwd=main, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=main, capture_output=True)

        subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "vendor/sub"],
            cwd=main,
            capture_output=True,
            check=True,
        )
        subprocess.run(["git", "commit", "-m", "add submodule"], cwd=main, capture_output=True)

        assert _has_analyzable_source_files(str(main)) is False


class TestHandleStopEventEarlyExits:
    """Tests for handle_stop_event early exit paths."""

    def test_handle_stop_event__returns_zero_when_no_session_data(self):
        """Subagent stops (stop_hook_active=True) and any other stop event with no DB data exit 0.

        Original test passed SubagentStopInput; legacy early-exit is now folded into
        handle_stop_event's "no working_directory" path. The semantic guarantee
        preserved: when there's nothing to analyze, return 0 without expensive work.
        """
        with patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls:
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = None

            assert handle_stop_event("test") == 0
            mock_db.get_session_working_directory.assert_called_once_with("test")
            mock_db.get_session_statistics.assert_not_called()

    def test_handle_stop_event__returns_zero_when_no_working_directory(self):
        """Returns 0 when session has no events (no working directory found)."""
        with patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls:
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = None

            assert handle_stop_event("nonexistent-session-xyz") == 0
            mock_db.get_session_working_directory.assert_called_once_with("nonexistent-session-xyz")
            # get_session_statistics should NOT have been called
            mock_db.get_session_statistics.assert_not_called()

    def test_handle_stop_event__fast_path_cache_hit_skips_expensive_computation(self, tmp_path):
        """Fast-path: same commit SHA + no source delta = instant return.

        This is the critical optimization for large repos like k8s-hq where the full
        content key hashes every source file. The fast-path uses only a few git
        commands and bails before reading any file content.
        """
        with (
            patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls,
            patch("slopometry.core.hook_handler._load_feedback_cache") as mock_cache,
            patch("slopometry.core.hook_handler._get_current_commit_sha") as mock_sha,
            patch("slopometry.core.hook_handler._has_source_changes") as mock_changes,
            patch("slopometry.core.hook_handler._compute_working_tree_cache_key") as mock_full_key,
        ):
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = str(tmp_path)

            # Cache has commit_sha from previous run
            mock_cache.return_value = FeedbackCacheState(last_key="old_key", file_hashes={}, commit_sha="abc123def")
            mock_sha.return_value = "abc123def"  # Same commit
            mock_changes.return_value = False  # No source delta (no mods, no new files)

            assert handle_stop_event("test-fast-cache") == 0
            # The expensive full key computation should NOT have been called
            mock_full_key.assert_not_called()
            mock_db.get_session_statistics.assert_not_called()

    def test_handle_stop_event__falls_through_when_commit_sha_differs(self, tmp_path):
        """When commit SHA changed, fast-path doesn't match, falls to full check."""
        with (
            patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls,
            patch("slopometry.core.hook_handler._load_feedback_cache") as mock_cache,
            patch("slopometry.core.hook_handler._get_current_commit_sha") as mock_sha,
            patch("slopometry.core.hook_handler._has_analyzable_source_files") as mock_has_src,
            patch("slopometry.core.hook_handler._compute_working_tree_cache_key"),
        ):
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = str(tmp_path)

            # Cache has old commit SHA
            mock_cache.return_value = FeedbackCacheState(last_key="old_key", file_hashes={}, commit_sha="old_sha")
            mock_sha.return_value = "new_sha"  # Different commit

            # Make it bail at the source files check for simplicity
            mock_has_src.return_value = False

            assert handle_stop_event("test-new-commit") == 0
            # _has_source_changes should NOT be called (SHA mismatch short-circuits)
            mock_has_src.assert_called_once()

    def test_handle_stop_event__legacy_cache_without_commit_sha_falls_through(self, tmp_path):
        """Caches from before the commit_sha field skip the fast-path gracefully."""
        with (
            patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls,
            patch("slopometry.core.hook_handler._load_feedback_cache") as mock_cache,
            patch("slopometry.core.hook_handler._has_analyzable_source_files") as mock_has_src,
            patch("slopometry.core.hook_handler._compute_working_tree_cache_key"),
        ):
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = str(tmp_path)

            # Legacy cache: no commit_sha field (defaults to None)
            mock_cache.return_value = FeedbackCacheState(last_key="old_key", file_hashes={})

            # Make it bail at source files check
            mock_has_src.return_value = False

            assert handle_stop_event("test-legacy-cache") == 0
            # Should fall through to _has_analyzable_source_files, not crash
            mock_has_src.assert_called_once()

    def test_handle_stop_event__full_cache_key_hit_after_fast_path_miss(self, tmp_path):
        """When fast-path misses (source modifications) but full key matches, still returns 0."""
        with (
            patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls,
            patch("slopometry.core.hook_handler._load_feedback_cache") as mock_cache,
            patch("slopometry.core.hook_handler._get_current_commit_sha") as mock_sha,
            patch("slopometry.core.hook_handler._has_source_changes") as mock_changes,
            patch("slopometry.core.hook_handler._has_analyzable_source_files") as mock_has_src,
            patch("slopometry.core.hook_handler._compute_working_tree_cache_key") as mock_full_key,
        ):
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = str(tmp_path)

            mock_cache.return_value = FeedbackCacheState(last_key="full_key_abc", file_hashes={}, commit_sha="abc123")
            mock_sha.return_value = "abc123"  # Same commit
            mock_changes.return_value = True  # Has source delta — fast-path can't confirm

            mock_has_src.return_value = True
            mock_full_key.return_value = "full_key_abc"  # But full key matches

            assert handle_stop_event("test-full-key-hit") == 0
            mock_full_key.assert_called_once()
            mock_db.get_session_statistics.assert_not_called()

    def test_handle_stop_event__returns_zero_when_no_source_files(self, tmp_path):
        """Returns 0 without computing stats when repo has no analyzable source files."""
        with (
            patch("slopometry.core.hook_handler.EventDatabase") as mock_db_cls,
            patch("slopometry.core.hook_handler._load_feedback_cache") as mock_cache,
            patch("slopometry.core.hook_handler._has_analyzable_source_files") as mock_has_src,
        ):
            mock_db = mock_db_cls.return_value
            mock_db.get_session_working_directory.return_value = str(tmp_path)

            mock_cache.return_value = None  # No cache (first run)
            mock_has_src.return_value = False  # No Python/Rust files

            assert handle_stop_event("test-no-source") == 0
            mock_db.get_session_statistics.assert_not_called()
            mock_has_src.assert_called_once_with(str(tmp_path))


class TestResolveWorkingDirectory:
    """Tests for _resolve_working_directory — repo-move tolerance."""

    def test_resolve_working_directory__returns_stored_when_directory_exists(self, tmp_path):
        """Existing stored path is used as-is."""
        assert _resolve_working_directory(str(tmp_path)) == str(tmp_path)

    def test_resolve_working_directory__falls_back_to_cwd_when_stored_missing(self, tmp_path, monkeypatch):
        """When the recorded path no longer exists (repo moved/renamed), use cwd instead.

        Simulates: session started at /old/path/repo, user `mv`s it to /new/path/repo,
        next Stop event must find the cache at /new/path/repo, not crash trying to
        mkdir under /old/path/repo/.slopometry.
        """
        monkeypatch.chdir(tmp_path)
        missing = "/this/path/does/not/exist/anywhere"
        assert _resolve_working_directory(missing) == str(tmp_path)

    def test_resolve_working_directory__returns_none_when_stored_is_none(self):
        """None stored_wd means unknown session — propagate None so caller bails."""
        assert _resolve_working_directory(None) is None


class TestHasSourceChanges:
    """Tests for _has_source_changes — the fast-path source-delta probe.

    Returns True when the working tree has any in-scope source delta (tracked
    modification OR new untracked source file), and False only when the source
    tree is provably identical to the committed state. Ignored dirs and submodule
    contents never count.
    """

    @staticmethod
    def _init_git_repo(path: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.email=t@t.com",
                "-c",
                "user.name=T",
                "commit",
                "--allow-empty",
                "-qm",
                "init",
            ],
            cwd=path,
            check=True,
        )

    @staticmethod
    def _commit(path: Path, message: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=path, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.email=t@t.com",
                "-c",
                "user.name=T",
                "commit",
                "-qm",
                message,
            ],
            cwd=path,
            check=True,
        )

    def test_has_source_changes__returns_false_when_only_ignored_dir_files_changed(self, tmp_path):
        """A modified .py inside __pycache__/ or .venv/ must not invalidate the cache.

        This is the perf+correctness fix for repos with editable installs: a tool
        regenerating .venv/site-packages/*.py would otherwise force every Stop event
        to compute the full cache key, even though those files are never in scope
        for slopometry's smell analysis.
        """
        self._init_git_repo(tmp_path)

        ignored_dir = tmp_path / ".venv" / "site-packages"
        ignored_dir.mkdir(parents=True)
        ignored_file = ignored_dir / "x.py"
        ignored_file.write_text("a = 1\n")
        self._commit(tmp_path, "add ignored")

        # Now MODIFY the ignored file — git diff reports it, but should_ignore_path filters it
        ignored_file.write_text("a = 2\n")

        assert _has_source_changes(str(tmp_path)) is False

    def test_has_source_changes__returns_true_when_non_ignored_file_changed(self, tmp_path):
        """A modified .py outside ignored dirs still trips the check."""
        self._init_git_repo(tmp_path)

        real_file = tmp_path / "real.py"
        real_file.write_text("a = 1\n")
        self._commit(tmp_path, "add real")

        real_file.write_text("a = 2\n")
        assert _has_source_changes(str(tmp_path)) is True

    def test_has_source_changes__returns_true_for_new_untracked_source_file(self, tmp_path):
        """A brand-new untracked .py is invisible to git diff but must still count.

        Without this, the fast-path would silently swallow the fire for newly
        created source files.
        """
        self._init_git_repo(tmp_path)
        (tmp_path / "real.py").write_text("a = 1\n")
        self._commit(tmp_path, "add real")

        # Create a new untracked source file (no diff against HEAD)
        (tmp_path / "brand_new.py").write_text("def feature(): pass\n")

        assert _has_source_changes(str(tmp_path)) is True

    def test_has_source_changes__returns_false_for_untracked_ignored_file(self, tmp_path):
        """A new untracked .py inside an ignored dir (build/) must not count."""
        self._init_git_repo(tmp_path)
        (tmp_path / "real.py").write_text("a = 1\n")
        self._commit(tmp_path, "add real")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "generated.py").write_text("# generated\n")

        assert _has_source_changes(str(tmp_path)) is False

    def test_has_source_changes__returns_false_for_clean_repo(self, tmp_path):
        """Clean repo with no source delta takes the cheap quick-exit path."""
        self._init_git_repo(tmp_path)
        assert _has_source_changes(str(tmp_path)) is False

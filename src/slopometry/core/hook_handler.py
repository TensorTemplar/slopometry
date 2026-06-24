"""Claude Code hook handler — receives Claude Code's stdin JSON, delegates to the
Claude-Code adapter for parsing, persists via the abstract protocol, and runs
the Claude-Code-specific stop-hook feedback pipeline (code smells, context
coverage, CLAUDE.md dev guidelines).

This module is the Claude-Code-specific glue. Harness-agnostic types live in
`core.protocol.events`; the wire-format parser is in
`core.protocol.adapters.claude_code`. New harnesses should not add code here —
write a new adapter.
"""

import json
import logging
import os
import select
import subprocess
import sys
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.git_tracker import GitTracker
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import FeedbackCacheState
from slopometry.core.models.protocol.events import AbstractEventSource, AbstractEventType
from slopometry.core.models.session import ContextCoverage
from slopometry.core.models.smell import ScopedSmell
from slopometry.core.protocol.dispatch import dispatch_event
from slopometry.core.settings import settings
from slopometry.core.working_tree_state import WorkingTreeStateCalculator
from slopometry.display.formatters import truncate_path

logger = logging.getLogger(__name__)


def _read_stdin_with_timeout(timeout_seconds: float = 5.0) -> str:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        return ""
    return sys.stdin.read().strip()


def handle_hook(event_type_override: AbstractEventType | None = None) -> int:
    """Main entry point for Claude Code hook invocations.

    Reads stdin, dispatches through the Claude-Code adapter, and runs the
    Claude-Code-specific stop-hook feedback pipeline when applicable.

    Args:
        event_type_override: Force a specific event type (used by per-event CLI
            subcommands: hook-pre-tool-use, hook-post-tool-use, etc.).
    """
    try:
        stdin_input = _read_stdin_with_timeout()
    except Exception:
        return 0
    if not stdin_input:
        return 0

    try:
        raw_payload = json.loads(stdin_input)
    except json.JSONDecodeError as e:
        if settings.debug_mode:
            print(f"Slopometry: Failed to parse hook input: {e}", file=sys.stderr)
        return 0

    try:
        event = dispatch_event(
            AbstractEventSource.CLAUDE_CODE,
            raw_payload,
            event_type_override=event_type_override,
        )
    except Exception as e:
        if settings.debug_mode:
            print(f"Slopometry hook error: {e}", file=sys.stderr)
        return 0

    stop_hook_active = bool(raw_payload.get("stop_hook_active"))
    if (
        settings.enable_complexity_analysis
        and not stop_hook_active
        and event.event_type in (AbstractEventType.TURN_COMPLETED, AbstractEventType.SUBAGENT_COMPLETED)
    ):
        return handle_stop_event(event.session_id, event.working_directory)

    if settings.debug_mode:
        debug_info = {
            "slopometry_event": {
                "session_id": event.session_id,
                "event_type": event.event_type.value,
                "sequence_number": event.sequence_number,
                "tool_name": event.tool_call.tool_name if event.tool_call else None,
                "tool_type": event.tool_call.tool_type if event.tool_call else None,
                "timestamp": event.timestamp.isoformat(),
            }
        }
        print(f"Slopometry captured: {json.dumps(debug_info, indent=2)}", file=sys.stderr)

    return 0


def _get_feedback_cache_path(working_directory: str) -> Path:
    cache_dir = Path(working_directory) / ".slopometry"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "feedback_cache.json"


def _get_current_commit_sha(working_directory: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass  # slopometry: allow-silent - return None below signals "git unavailable" to callers
    return None


def _has_source_changes(working_directory: str) -> bool:
    """Cheap working-tree delta probe used by the feedback-cache fast path.

    Two checks so the fast path never suppresses a genuine fire:
      1. Tracked modifications — `git diff --quiet -- *.py *.rs` (cheap first).
      2. Untracked source files — `git ls-files --others` filtered.
    """
    wt = WorkingTreeStateCalculator(working_directory, languages=None)

    any_diff = False
    for diff_args in [
        ["git", "diff", "--quiet", "--ignore-submodules=all", "--", "*.py", "*.rs"],
        ["git", "diff", "--cached", "--quiet", "--ignore-submodules=all", "--", "*.py", "*.rs"],
    ]:
        try:
            result = subprocess.run(diff_args, cwd=working_directory, capture_output=True, timeout=10)
            if result.returncode != 0:
                any_diff = True
                break
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return True

    if any_diff and wt._get_modified_source_files_from_git():
        return True
    return bool(wt.get_untracked_source_files())


def _compute_working_tree_cache_key(working_directory: str) -> str:
    wt_calculator = WorkingTreeStateCalculator(working_directory, languages=None)
    return wt_calculator.calculate_source_content_key()


def _load_feedback_cache(working_directory: str) -> FeedbackCacheState | None:
    cache_path = _get_feedback_cache_path(working_directory)
    if not cache_path.exists():
        return None
    try:
        return FeedbackCacheState.model_validate_json(cache_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _save_feedback_cache(
    working_directory: str, cache_key: str, file_hashes: dict[str, str], commit_sha: str | None = None
) -> None:
    cache_path = _get_feedback_cache_path(working_directory)
    try:
        state = FeedbackCacheState(last_key=cache_key, file_hashes=file_hashes, commit_sha=commit_sha)
        cache_path.write_text(state.model_dump_json())
    except OSError as e:
        logger.debug(f"Failed to save feedback cache: {e}")


def _has_analyzable_source_files(working_directory: str) -> bool:
    return GitTracker(Path(working_directory)).has_analyzable_source_files()


def _resolve_working_directory(stored_wd: str | None) -> str | None:
    """Resolve the effective working directory for a stop event.

    Falls back to `os.getcwd()` (which Claude Code passes through to the hook
    subprocess) when the stored working directory from the first event no
    longer resolves to a live directory (e.g., repo was renamed or moved).
    """
    if stored_wd is None:
        return None
    if Path(stored_wd).is_dir():
        return stored_wd
    cwd = os.getcwd()
    if Path(cwd).is_dir():
        return cwd
    return stored_wd


def handle_stop_event(session_id: str, working_directory: str | None = None) -> int:
    """Run the Claude-Code stop-hook feedback pipeline.

    Args:
        session_id: The session ID.
        working_directory: The working directory to scope analysis to. If None,
            resolved from the DB-recorded value with a cwd fallback.

    Returns:
        Exit code (0 for silent success, 2 for blocking with feedback).
    """
    if working_directory is None:
        db = EventDatabase()
        working_directory = _resolve_working_directory(db.get_session_working_directory(session_id))
    if not working_directory:
        return 0

    cached_state = _load_feedback_cache(working_directory)
    if cached_state is not None and cached_state.commit_sha is not None:
        current_sha = _get_current_commit_sha(working_directory)
        if current_sha == cached_state.commit_sha and not _has_source_changes(working_directory):
            return 0

    if not _has_analyzable_source_files(working_directory):
        return 0

    cache_key = _compute_working_tree_cache_key(working_directory)
    if cached_state is not None and cached_state.last_key == cache_key:
        return 0

    db = EventDatabase()
    stats = db.get_session_statistics(session_id)
    if not stats:
        return 0

    current_metrics = stats.complexity_metrics
    delta = stats.complexity_delta

    wt_calculator = WorkingTreeStateCalculator(working_directory, languages=None)
    current_file_hashes = wt_calculator.get_source_file_content_hashes()

    if cached_state is not None:
        edited_files = wt_calculator.get_files_changed_since(cached_state.file_hashes)
    else:
        edited_files = wt_calculator.get_modified_source_file_paths()

    feedback_parts: list[str] = []

    if current_metrics:
        scoped_smells = scope_smells_for_session(
            current_metrics, delta, edited_files, working_directory, stats.context_coverage
        )
        code_smells = [s for s in scoped_smells if s.name != "unread_related_tests"]
        context_smells = [s for s in scoped_smells if s.name == "unread_related_tests"]
        code_feedback, has_code_smells, _ = format_code_smell_feedback(code_smells, session_id)
        if has_code_smells:
            feedback_parts.append(code_feedback)
        context_smell_feedback, has_context_smells, _ = format_code_smell_feedback(context_smells, session_id)
        if has_context_smells:
            feedback_parts.append(context_smell_feedback)

    if settings.enable_complexity_feedback and stats.context_coverage and stats.context_coverage.has_gaps:
        context_feedback = format_context_coverage_feedback(stats.context_coverage)
        if context_feedback:
            feedback_parts.append(context_feedback)

    if settings.feedback_dev_guidelines:
        dev_guidelines = extract_dev_guidelines_from_claude_md(working_directory)
        if dev_guidelines:
            feedback_parts.append(f"\n**Project Development Guidelines:**\n{dev_guidelines}")

    current_commit_sha = _get_current_commit_sha(working_directory)
    _save_feedback_cache(working_directory, cache_key, current_file_hashes, commit_sha=current_commit_sha)

    if feedback_parts:
        feedback = "\n\n".join(feedback_parts)
        feedback += (
            f"\n\n---\n**Session**: `{session_id}` | Details: `slopometry solo show {session_id} --smell-details`"
        )
        hook_output = {"decision": "block", "reason": feedback}
        print(json.dumps(hook_output))
        return 2
    return 0


def format_context_coverage_feedback(coverage: ContextCoverage) -> str:
    lines = []
    lines.append("")
    lines.append("**Context Coverage**")
    read_ratio = coverage.files_read_before_edit_ratio
    if read_ratio < 1.0:
        lines.append(
            f"   • Read before edit: {read_ratio:.0%} ({int(read_ratio * len(coverage.files_edited))}/{len(coverage.files_edited)} files)"
        )
    else:
        lines.append(f"   • Read before edit: {read_ratio:.0%} ✓")
    imports_cov = coverage.overall_imports_coverage
    if imports_cov < 100:
        lines.append(f"   • Imports coverage: {imports_cov:.0f}%")
    dependents_cov = coverage.overall_dependents_coverage
    if dependents_cov < 100:
        lines.append(f"   • Dependents coverage: {dependents_cov:.0f}%")
    if coverage.blind_spots:
        lines.append("")
        lines.append("**Blind spots** (related files not read):")
        for blind_spot in coverage.blind_spots[:5]:
            lines.append(f"   • {truncate_path(blind_spot, max_width=65)}")
        if len(coverage.blind_spots) > 5:
            lines.append(f"   ... and {len(coverage.blind_spots) - 5} more")
    return "\n".join(lines)


def extract_dev_guidelines_from_claude_md(working_directory: str) -> str:
    """Extract the `## Development guidelines` section from CLAUDE.md.

    CLAUDE.md is Claude Code's project-level instructions file convention.
    """
    claude_md_path = Path(working_directory) / "CLAUDE.md"
    if not claude_md_path.exists():
        return ""
    content = claude_md_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("## Development guidelines"):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("## ") or line.strip().startswith("# "):
                break
            section_lines.append(line)
    if not section_lines:
        return ""
    return "\n".join(section_lines).strip()


def _get_related_files_via_imports(edited_files: set[str], working_directory: str) -> set[str]:
    """Build the set of files related to edited files for blocking smell scoping."""
    from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

    related = set(edited_files)
    analyzer = ContextCoverageAnalyzer(Path(working_directory))
    analyzer._build_import_graph()
    for edited_file in edited_files:
        test_files = analyzer._find_test_files(edited_file)
        related.update(test_files)
    return related


def _is_file_related_to_edits(smell_file: str, edited_files: set[str], related_files: set[str]) -> bool:
    return smell_file in edited_files or smell_file in related_files


def scope_smells_for_session(
    current_metrics: ExtendedComplexityMetrics,
    delta: ComplexityDelta | None,
    edited_files: set[str],
    working_directory: str,
    context_coverage: ContextCoverage | None = None,
) -> list[ScopedSmell]:
    """Classify smells for a specific session context.

    Extracts the scoping/classification logic that determines which smells are
    blocking vs informational and which files are actionable for this session.
    """
    blocking_smell_names = {"test_skip", "swallowed_exception"}
    # REASON: acknowledged_silent_except is acceptable individually but blocks on INCREASE — an `# slopometry: allow-silent` marker moves a handler out of swallowed_exception, so a rising marker count is the anti-reward-hack signal that new suppressions need justifying.
    block_on_increase_names = {"acknowledged_silent_except"}

    related_via_imports: set[str] = set()
    if edited_files:
        related_via_imports = _get_related_files_via_imports(edited_files, working_directory)

    result: list[ScopedSmell] = []

    if context_coverage:
        unread_tests: list[str] = []
        for file_cov in context_coverage.file_coverage:
            for test_file in file_cov.test_files:
                if test_file not in file_cov.test_files_read and test_file not in unread_tests:
                    unread_tests.append(test_file)
        if unread_tests:
            result.append(
                ScopedSmell(
                    label="Unread Related Tests",
                    name="unread_related_tests",
                    count=len(unread_tests),
                    change=0,
                    actionable_files=unread_tests,
                    guidance="BLOCKING: You MUST review these tests to ensure changes are accounted for and necessary coverage is added for new functionality",
                    is_blocking=True,
                )
            )

    smell_changes = delta.get_smell_changes() if delta else {}

    for smell in current_metrics.get_smells():
        if smell.count == 0:
            continue

        change = smell_changes.get(smell.name, 0)
        guidance = smell.definition.guidance

        is_blocking_smell = smell.name in blocking_smell_names or (
            smell.name in block_on_increase_names and change > 0
        )

        if is_blocking_smell and edited_files:
            related_files = [
                f for f in smell.files if _is_file_related_to_edits(f, edited_files, related_via_imports)
            ]
            unrelated_files = [f for f in smell.files if f not in related_files]

            if related_files:
                result.append(
                    ScopedSmell(
                        label=smell.label,
                        name=smell.name,
                        count=len(related_files),
                        change=change,
                        actionable_files=related_files,
                        guidance=guidance,
                        is_blocking=True,
                    )
                )

            if unrelated_files:
                result.append(
                    ScopedSmell(
                        label=smell.label,
                        name=smell.name,
                        count=len(unrelated_files),
                        change=0,
                        actionable_files=unrelated_files,
                        guidance=guidance,
                        is_blocking=False,
                    )
                )
        else:
            if edited_files:
                actionable_files = [
                    f for f in smell.files if _is_file_related_to_edits(f, edited_files, related_via_imports)
                ]
            else:
                actionable_files = list(smell.files)
            result.append(
                ScopedSmell(
                    label=smell.label,
                    name=smell.name,
                    count=smell.count,
                    change=change,
                    actionable_files=actionable_files,
                    guidance=guidance,
                    is_blocking=False,
                )
            )

    return result


def format_code_smell_feedback(
    scoped_smells: list[ScopedSmell],
    session_id: str | None = None,
) -> tuple[str, bool, bool]:
    """Format pre-classified smell data into feedback output.

    Returns:
        Tuple of (formatted feedback string, has_smells, has_blocking_smells).
    """
    blocking_smells = [s for s in scoped_smells if s.is_blocking]
    other_smells = [s for s in scoped_smells if not s.is_blocking]

    lines: list[str] = []
    has_blocking = len(blocking_smells) > 0

    blocking_increased = [s for s in blocking_smells if s.change > 0]
    blocking_decreased = [s for s in blocking_smells if s.change < 0]
    blocking_unchanged = [s for s in blocking_smells if s.change == 0]

    if blocking_decreased:
        lines.append("")
        lines.append("**Code Smell Improvements** (decreases - great work!):")
        lines.append("")
        for smell in blocking_decreased:
            change_str = f" ({smell.change})"
            lines.append(f"   • **{smell.label}**: {smell.count} file(s){change_str}")
        lines.append("")

    blocking_requiring_action = blocking_unchanged + blocking_increased
    if blocking_requiring_action:
        if not blocking_decreased:
            lines.append("")
        lines.append("**ACTION REQUIRED** - The following issues are in files that are in scope for this PR:")
        lines.append("")
        for smell in blocking_requiring_action:
            change_str = f" (+{smell.change})" if smell.change > 0 else ""
            lines.append(f"   • **{smell.label}**: {smell.count} file(s){change_str}")
            for f in smell.actionable_files[:5]:
                lines.append(f"     - {truncate_path(f, max_width=60)}")
            if len(smell.actionable_files) > 5:
                lines.append(f"     ... and {len(smell.actionable_files) - 5} more")
            if smell.guidance:
                lines.append(f"     → {smell.guidance}")
        lines.append("")

    smells_increased = [s for s in other_smells if s.change > 0]
    smells_decreased = [s for s in other_smells if s.change < 0]
    other_smells_with_changes = smells_increased + smells_decreased

    if other_smells_with_changes:
        if not blocking_increased:
            lines.append("")

        if smells_decreased:
            lines.append("**Code Smell Improvements** (decreases - great work!):")
            lines.append("")
            for smell in smells_decreased:
                change_str = f" ({smell.change})"
                lines.append(f"   • **{smell.label}**: {smell.count}{change_str}")
            lines.append("")

        if smells_increased:
            lines.append(
                "**Code Smells** (increases require review, irrespective of which session edited related files):"
            )
            lines.append("")
            for smell in smells_increased:
                change_str = f" (+{smell.change})"
                lines.append(f"   • **{smell.label}**: {smell.count}{change_str}")
                for f in smell.actionable_files[:3]:
                    lines.append(f"     - {truncate_path(f, max_width=60)}")
                if len(smell.actionable_files) > 3:
                    lines.append(f"     ... and {len(smell.actionable_files) - 3} more")
                if smell.guidance:
                    lines.append(f"     → {smell.guidance}")

    has_smells = len(blocking_smells) > 0 or len(other_smells_with_changes) > 0
    if has_smells:
        return "\n".join(lines), True, has_blocking
    return "", False, False


__all__ = [
    "handle_hook",
    "handle_stop_event",
    "format_context_coverage_feedback",
    "extract_dev_guidelines_from_claude_md",
    "scope_smells_for_session",
    "format_code_smell_feedback",
]

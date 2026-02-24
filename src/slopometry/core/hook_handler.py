"""Hook handler script invoked by Claude Code for each event."""

import hashlib
import json
import logging
import os
import select
import sys
from pathlib import Path

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.git_tracker import GitTracker
from slopometry.core.lock import SlopometryLock
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import (
    HookEvent,
    HookEventType,
    HookInputUnion,
    NotificationInput,
    PostToolUseInput,
    PreToolUseInput,
    StopInput,
    SubagentStopInput,
    ToolType,
)
from slopometry.core.models.session import ContextCoverage
from slopometry.core.models.smell import ScopedSmell
from slopometry.core.project_tracker import ProjectTracker
from slopometry.core.settings import settings
from slopometry.core.working_tree_state import WorkingTreeStateCalculator
from slopometry.display.formatters import truncate_path

logger = logging.getLogger(__name__)


def get_tool_type(tool_name: str) -> ToolType:
    """Map tool name to ToolType enum."""
    tool_map = {
        "bash": ToolType.BASH,
        "read": ToolType.READ,
        "write": ToolType.WRITE,
        "edit": ToolType.EDIT,
        "multiedit": ToolType.MULTI_EDIT,
        "grep": ToolType.GREP,
        "glob": ToolType.GLOB,
        "ls": ToolType.LS,
        "task": ToolType.TASK,
        "todoread": ToolType.TODO_READ,
        "todowrite": ToolType.TODO_WRITE,
        "taskcreate": ToolType.TASK_CREATE,
        "taskupdate": ToolType.TASK_UPDATE,
        "tasklist": ToolType.TASK_LIST,
        "taskget": ToolType.TASK_GET,
        "webfetch": ToolType.WEB_FETCH,
        "websearch": ToolType.WEB_SEARCH,
        "notebookread": ToolType.NOTEBOOK_READ,
        "notebookedit": ToolType.NOTEBOOK_EDIT,
        "exit_plan_mode": ToolType.EXIT_PLAN_MODE,
        "mcp__ide__getdiagnostics": ToolType.MCP_IDE_GET_DIAGNOSTICS,
        "mcp__ide__executecode": ToolType.MCP_IDE_EXECUTE_CODE,
        "mcp__ide__getworkspaceinfo": ToolType.MCP_IDE_GET_WORKSPACE_INFO,
        "mcp__ide__getfilecontents": ToolType.MCP_IDE_GET_FILE_CONTENTS,
        "mcp__ide__createfile": ToolType.MCP_IDE_CREATE_FILE,
        "mcp__ide__deletefile": ToolType.MCP_IDE_DELETE_FILE,
        "mcp__ide__renamefile": ToolType.MCP_IDE_RENAME_FILE,
        "mcp__ide__searchfiles": ToolType.MCP_IDE_SEARCH_FILES,
        "mcp__filesystem__read": ToolType.MCP_FILESYSTEM_READ,
        "mcp__filesystem__write": ToolType.MCP_FILESYSTEM_WRITE,
        "mcp__filesystem__list": ToolType.MCP_FILESYSTEM_LIST,
        "mcp__database__query": ToolType.MCP_DATABASE_QUERY,
        "mcp__database__schema": ToolType.MCP_DATABASE_SCHEMA,
        "mcp__web__scrape": ToolType.MCP_WEB_SCRAPE,
        "mcp__web__search": ToolType.MCP_WEB_SEARCH,
        "mcp__github__getrepo": ToolType.MCP_GITHUB_GET_REPO,
        "mcp__github__createissue": ToolType.MCP_GITHUB_CREATE_ISSUE,
        "mcp__github__listissues": ToolType.MCP_GITHUB_LIST_ISSUES,
        "mcp__slack__sendmessage": ToolType.MCP_SLACK_SEND_MESSAGE,
        "mcp__slack__listchannels": ToolType.MCP_SLACK_LIST_CHANNELS,
    }

    if tool_name.lower().startswith("mcp__") and tool_name.lower() not in tool_map:
        return ToolType.MCP_OTHER

    return tool_map.get(tool_name.lower(), ToolType.OTHER)


def parse_hook_input(raw_data: dict) -> HookInputUnion:
    """Parse and validate hook input using appropriate Pydantic model.

    Since Claude Code doesn't send explicit hook type info, we infer the type
    from the data structure based on the documented schemas.
    """

    fields = set(raw_data.keys())

    if "tool_name" in fields and "tool_input" in fields and "tool_response" not in fields:
        return PreToolUseInput(**raw_data)

    elif "tool_name" in fields and "tool_input" in fields and "tool_response" in fields:
        return PostToolUseInput(**raw_data)

    elif "message" in fields:
        return NotificationInput(**raw_data)

    elif "stop_hook_active" in fields:
        if raw_data.get("stop_hook_active"):
            return SubagentStopInput(**raw_data)
        return StopInput(**raw_data)

    elif "session_id" in fields and "transcript_path" in fields:
        return StopInput(**raw_data)

    else:
        raise ValueError(f"Unknown hook input schema with fields: {fields}")


def _read_stdin_with_timeout(timeout_seconds: float = 5.0) -> str:
    """Read stdin with a timeout to prevent hanging on unclosed pipes.

    Uses select() to check if stdin has data available before reading.
    Returns empty string if stdin is not ready within the timeout.

    Args:
        timeout_seconds: Maximum seconds to wait for stdin data.

    Returns:
        Stripped stdin content, or empty string on timeout/error.
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        return ""
    return sys.stdin.read().strip()


def handle_hook(event_type_override: HookEventType | None = None) -> int:
    """Main hook handler function.

    Reads and parses stdin BEFORE acquiring the lock to prevent hung pipes
    from holding the lock and starving all other hook invocations.

    Args:
        event_type_override: Optional override for the event type, used when called via specific hook entrypoints
    """
    try:
        stdin_input = _read_stdin_with_timeout()
    except Exception:
        return 0
    if not stdin_input:
        return 0

    try:
        raw_data = json.loads(stdin_input)
        parsed_input = parse_hook_input(raw_data)
    except Exception as e:
        if settings.debug_mode:
            print(f"Slopometry: Failed to parse hook input: {e}", file=sys.stderr)
        return 0

    lock = SlopometryLock(project_dir=os.getcwd())
    with lock.acquire() as acquired:
        if not acquired:
            print("Slopometry: Could not acquire lock, skipping hook execution.", file=sys.stderr)
            return 0

        return _handle_hook_internal(event_type_override, parsed_input, raw_data)


def _handle_hook_internal(
    event_type_override: HookEventType | None,
    parsed_input: HookInputUnion,
    raw_data: dict,
) -> int:
    """Internal hook handler logic (runs under lock with pre-parsed data).

    Args:
        event_type_override: Optional override for the event type.
        parsed_input: Pre-parsed and validated hook input.
        raw_data: Raw JSON data from stdin (stored as event metadata).
    """
    try:
        event_type = event_type_override if event_type_override else detect_event_type_from_parsed(parsed_input)

        session_id = parsed_input.session_id

        session_manager = SessionManager()
        sequence_number = session_manager.get_next_sequence_number(session_id)

        git_tracker = GitTracker()
        git_state = None
        match (event_type, sequence_number):
            case (HookEventType.PRE_TOOL_USE, 1) | (HookEventType.STOP, 1):
                git_state = git_tracker.get_git_state()
            case (HookEventType.STOP, _):
                git_state = git_tracker.get_git_state()

        working_directory = os.getcwd()
        project_tracker = ProjectTracker(working_dir=Path(working_directory))
        project = project_tracker.get_project()

        event = HookEvent(
            session_id=session_id,
            event_type=event_type,
            sequence_number=sequence_number,
            metadata=raw_data,
            git_state=git_state,
            working_directory=working_directory,
            project=project,
            transcript_path=parsed_input.transcript_path,
        )

        if isinstance(parsed_input, PreToolUseInput | PostToolUseInput):
            event.tool_name = parsed_input.tool_name
            event.tool_type = get_tool_type(parsed_input.tool_name)

            if isinstance(parsed_input, PostToolUseInput):
                if isinstance(parsed_input.tool_response, dict):
                    event.duration_ms = parsed_input.tool_response.get("duration_ms")
                    event.exit_code = parsed_input.tool_response.get("exit_code")
                    event.error_message = parsed_input.tool_response.get("error")
                else:
                    event.duration_ms = None
                    event.exit_code = None
                    event.error_message = None

        db = EventDatabase()
        db.save_event(event)

        if settings.enable_complexity_analysis and isinstance(parsed_input, StopInput | SubagentStopInput):
            return handle_stop_event(session_id, parsed_input)

        if settings.debug_mode:
            debug_info = {
                "slopometry_event": {
                    "session_id": session_id,
                    "event_type": event_type.value,
                    "sequence_number": sequence_number,
                    "tool_name": event.tool_name,
                    "tool_type": event.tool_type.value if event.tool_type else None,
                    "timestamp": event.timestamp.isoformat(),
                    "parsed_input_type": type(parsed_input).__name__,
                }
            }
            print(f"Slopometry captured: {json.dumps(debug_info, indent=2)}", file=sys.stderr)

        return 0

    except Exception as e:
        import traceback

        error_msg = f"Slopometry hook error: {e}\n{traceback.format_exc()}"

        if settings.debug_mode:
            print(error_msg, file=sys.stderr)

        return 0


def get_modified_python_files(working_directory: str | None) -> set[str]:
    """Get modified Python files from git working tree.

    Uses `git diff --name-only` to get uncommitted changes (both staged and unstaged).
    This is more reliable than transcript-based context coverage for detecting
    which files the user has actually modified.

    Args:
        working_directory: Path to git repository

    Returns:
        Set of relative paths to modified Python files

    Raises:
        ValueError: If working_directory is None or doesn't exist
        RuntimeError: If git commands fail
    """
    if not working_directory:
        raise ValueError("working_directory is required for git diff")

    import subprocess

    working_dir = Path(working_directory)
    if not working_dir.exists():
        raise ValueError(f"working_directory does not exist: {working_directory}")

    modified_files: set[str] = set()

    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "*.py"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr}")
    modified_files.update(line.strip() for line in result.stdout.strip().split("\n") if line.strip())

    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--", "*.py"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {result.stderr}")
    modified_files.update(line.strip() for line in result.stdout.strip().split("\n") if line.strip())

    return modified_files


def _get_feedback_cache_path(working_directory: str) -> Path:
    """Get path to the feedback cache file for a working directory."""
    cache_dir = Path(working_directory) / ".slopometry"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "feedback_cache.json"


def _compute_feedback_cache_key(working_directory: str, edited_files: set[str], feedback_hash: str) -> str:
    """Compute a cache key for the current state.

    Uses language-aware change detection to avoid cache invalidation from
    non-source file changes (like uv.lock, submodules, build artifacts, etc.).

    The languages parameter defaults to None (all supported languages).
    Currently only Python is supported; future languages will be auto-detected
    via LanguageDetector when added to the registry.

    Args:
        working_directory: Path to the working directory
        edited_files: Set of edited file paths
        feedback_hash: Hash of the feedback content

    Returns:
        Cache key string
    """
    tracker = GitTracker(Path(working_directory))
    git_state = tracker.get_git_state()
    commit_sha = git_state.commit_sha or "unknown"

    wt_calculator = WorkingTreeStateCalculator(working_directory, languages=None)
    has_source_changes = bool(wt_calculator._get_modified_source_files_from_git())
    working_tree_hash = wt_calculator.calculate_working_tree_hash(commit_sha) if has_source_changes else "clean"

    files_key = ",".join(sorted(edited_files))
    key_parts = f"{commit_sha}:{working_tree_hash}:{files_key}:{feedback_hash}"
    return hashlib.blake2b(key_parts.encode(), digest_size=8).hexdigest()


def _is_feedback_cached(working_directory: str, cache_key: str) -> bool:
    """Check if the feedback for this state was already shown.

    Args:
        working_directory: Path to the working directory
        cache_key: Cache key to check

    Returns:
        True if feedback was already shown for this state
    """
    cache_path = _get_feedback_cache_path(working_directory)
    if not cache_path.exists():
        return False

    try:
        cache_data = json.loads(cache_path.read_text())
        return cache_data.get("last_key") == cache_key
    except (json.JSONDecodeError, OSError):
        return False


def _save_feedback_cache(working_directory: str, cache_key: str) -> None:
    """Save the feedback cache key.

    Args:
        working_directory: Path to the working directory
        cache_key: Cache key to save
    """
    cache_path = _get_feedback_cache_path(working_directory)
    try:
        cache_path.write_text(json.dumps({"last_key": cache_key}))
    except OSError as e:
        logger.debug(f"Failed to save feedback cache: {e}")


def handle_stop_event(session_id: str, parsed_input: "StopInput | SubagentStopInput") -> int:
    """Handle Stop events with code smell feedback and optional complexity analysis.

    Code smells are always checked (independent of enable_complexity_feedback).
    Complexity metrics are only shown when enable_complexity_feedback is True.
    Dev guidelines are shown when feedback_dev_guidelines is True.

    Feedback is cached - if the same feedback would be shown twice without code changes,
    the second invocation returns silently.

    Args:
        session_id: The session ID
        parsed_input: The stop event input

    Returns:
        Exit code (0 for success, 2 for blocking with feedback)
    """
    if parsed_input.stop_hook_active:
        return 0

    db = EventDatabase()
    stats = db.get_session_statistics(session_id)

    if not stats:
        return 0

    current_metrics, delta = db.calculate_extended_complexity_metrics(stats.working_directory)

    feedback_parts: list[str] = []
    cache_stable_parts: list[str] = []

    # Get edited files from git (more reliable than transcript-based context coverage)
    try:
        edited_files = get_modified_python_files(stats.working_directory)
    except (ValueError, RuntimeError) as e:
        logger.debug(f"Failed to get modified Python files: {e}")
        edited_files = set()

    # Smell feedback is stable (based on code state, not session activity)
    if current_metrics:
        scoped_smells = scope_smells_for_session(
            current_metrics, delta, edited_files, stats.working_directory, stats.context_coverage
        )
        smell_feedback, has_smells, _ = format_code_smell_feedback(scoped_smells, session_id, stats.working_directory)
        if has_smells:
            feedback_parts.append(smell_feedback)
            cache_stable_parts.append(smell_feedback)

    # Context coverage - informational but NOT stable (changes with every Read/Glob/Grep)
    # Excluded from cache hash to avoid invalidation on tool calls
    if settings.enable_complexity_feedback and stats.context_coverage and stats.context_coverage.has_gaps:
        context_feedback = format_context_coverage_feedback(stats.context_coverage)
        if context_feedback:
            feedback_parts.append(context_feedback)

    if settings.feedback_dev_guidelines:
        dev_guidelines = extract_dev_guidelines_from_claude_md(stats.working_directory)
        if dev_guidelines:
            feedback_parts.append(f"\n**Project Development Guidelines:**\n{dev_guidelines}")

    if feedback_parts:
        feedback = "\n\n".join(feedback_parts)

        # Cache key uses only code-based smell feedback — context coverage
        # changes with every tool call and would invalidate cache
        cache_content = "\n\n".join(cache_stable_parts) if cache_stable_parts else ""
        feedback_hash = hashlib.blake2b(cache_content.encode(), digest_size=8).hexdigest()

        feedback += (
            f"\n\n---\n**Session**: `{session_id}` | Details: `slopometry solo show {session_id} --smell-details`"
        )

        cache_key = _compute_feedback_cache_key(stats.working_directory, edited_files, feedback_hash)

        if _is_feedback_cached(stats.working_directory, cache_key):
            return 0

        _save_feedback_cache(stats.working_directory, cache_key)

        hook_output = {"decision": "block", "reason": feedback}
        print(json.dumps(hook_output))
        return 2

    return 0


def format_context_coverage_feedback(coverage: ContextCoverage) -> str:
    """Format context coverage information for Claude consumption.

    Args:
        coverage: Context coverage metrics from the session

    Returns:
        Formatted feedback string highlighting gaps in context reading
    """
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
    """Extract '## Development guidelines' section from CLAUDE.md in the CWD.

    Args:
        working_directory: The current working directory to search for CLAUDE.md

    Returns:
        The extracted dev guidelines content, or empty string if not found

    Raises:
        OSError: If CLAUDE.md exists but cannot be read
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
    """Build set of files related to edited files for blocking smell scoping.

    Only includes edited files and their test files. Does NOT include reverse
    import graph dependents — those files weren't edited, so their pre-existing
    smells are not actionable in the stop hook.

    Args:
        edited_files: Set of files edited in this session
        working_directory: Path to the working directory

    Returns:
        Set of file paths related to edited files (includes edited_files themselves)

    Raises:
        Exception: If import graph analysis fails (no silent fallback)
    """
    from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer

    related = set(edited_files)

    analyzer = ContextCoverageAnalyzer(Path(working_directory))
    analyzer._build_import_graph()

    for edited_file in edited_files:
        test_files = analyzer._find_test_files(edited_file)
        related.update(test_files)

    return related


def _is_file_related_to_edits(smell_file: str, edited_files: set[str], related_files: set[str]) -> bool:
    """Check if a smell file is related to the edited files.

    A file is related if:
    - It is directly in edited_files
    - It is in the related_files set (computed via import graph)

    Args:
        smell_file: Path to a file containing a smell
        edited_files: Set of files edited in this session
        related_files: Set of related files via import graph (required)

    Returns:
        True if the smell file is related to edited files
    """
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

    Args:
        current_metrics: Current complexity metrics with code smell counts
        delta: Optional complexity delta showing changes
        edited_files: Set of files edited in this session
        working_directory: Path to working directory for import graph analysis
        context_coverage: Optional context coverage for detecting unread related tests

    Returns:
        List of ScopedSmell instances classified for this session
    """
    blocking_smell_names = {"test_skip", "swallowed_exception"}

    related_via_imports: set[str] = set()
    if edited_files:
        related_via_imports = _get_related_files_via_imports(edited_files, working_directory)

    result: list[ScopedSmell] = []

    # Synthetic blocking smell: unread related tests
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

        if smell.name in blocking_smell_names and edited_files:
            related_files = [f for f in smell.files if _is_file_related_to_edits(f, edited_files, related_via_imports)]
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
    working_directory: str | None = None,
) -> tuple[str, bool, bool]:
    """Format pre-classified smell data into feedback output.

    Args:
        scoped_smells: Pre-classified smells from scope_smells_for_session
        session_id: Session ID for generating the smell-details command
        working_directory: Path to working directory (unused, kept for caller compatibility)

    Returns:
        Tuple of (formatted feedback string, has_smells, has_blocking_smells)
        - has_smells: whether any code smells were detected
        - has_blocking_smells: whether any BLOCKING smells in edited files were detected
    """
    blocking_smells = [s for s in scoped_smells if s.is_blocking]
    other_smells = [s for s in scoped_smells if not s.is_blocking]

    lines: list[str] = []
    has_blocking = len(blocking_smells) > 0

    # Separate blocking smell increases from decreases
    blocking_increased = [s for s in blocking_smells if s.change > 0]
    blocking_decreased = [s for s in blocking_smells if s.change < 0]
    blocking_unchanged = [s for s in blocking_smells if s.change == 0]

    # Show improvements (decreases) first - don't require action
    if blocking_decreased:
        lines.append("")
        lines.append("**Code Smell Improvements** (decreases - great work!):")
        lines.append("")
        for smell in blocking_decreased:
            change_str = f" ({smell.change})"
            lines.append(f"   • **{smell.label}**: {smell.count} file(s){change_str}")
        lines.append("")

    # Show unchanged and increased blocking smells (require action)
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

    # Separate increases (require review) from decreases (improvements - no review needed)
    smells_increased = [s for s in other_smells if s.change > 0]
    smells_decreased = [s for s in other_smells if s.change < 0]
    other_smells_with_changes = smells_increased + smells_decreased

    if other_smells_with_changes:
        if not blocking_increased:
            lines.append("")

        # Show improvements first (decreases) - these don't require review
        if smells_decreased:
            lines.append("**Code Smell Improvements** (decreases - great work!):")
            lines.append("")
            for smell in smells_decreased:
                change_str = f" ({smell.change})"
                lines.append(f"   • **{smell.label}**: {smell.count}{change_str}")
            lines.append("")

        # Show increases - these require review
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


def format_complexity_metrics_only(
    current_metrics: "ExtendedComplexityMetrics",
    delta: "ComplexityDelta",
    baseline_feedback: str = "",
    context_feedback: str = "",
) -> str:
    """Format complexity metrics feedback (without code smells).

    Args:
        current_metrics: Current complexity metrics
        delta: Complexity changes from previous commit
        baseline_feedback: Optional baseline comparison feedback
        context_feedback: Optional context coverage feedback

    Returns:
        Formatted feedback string for Claude
    """
    lines = []

    lines.append("**Complexity Analysis Summary**")
    lines.append("")

    if delta.total_complexity_change > 0:
        lines.append(
            f"**Complexity increased by +{delta.total_complexity_change}** (now {current_metrics.total_complexity} total)"
        )
    elif delta.total_complexity_change < 0:
        lines.append(
            f"**Complexity decreased by {delta.total_complexity_change}** (now {current_metrics.total_complexity} total)"
        )
    else:
        lines.append(f"**No net complexity change** ({current_metrics.total_complexity} total)")

    if delta.files_added:
        truncated_added = [truncate_path(f, max_width=30) for f in delta.files_added[:3]]
        lines.append(f"**Added {len(delta.files_added)} files**: {', '.join(truncated_added)}")
        if len(delta.files_added) > 3:
            lines.append(f"   ... and {len(delta.files_added) - 3} more")

    if delta.files_removed:
        truncated_removed = [truncate_path(f, max_width=30) for f in delta.files_removed[:3]]
        lines.append(f"**Removed {len(delta.files_removed)} files**: {', '.join(truncated_removed)}")
        if len(delta.files_removed) > 3:
            lines.append(f"   ... and {len(delta.files_removed) - 3} more")

    lines.append("")
    lines.append("**Code Quality**:")
    lines.append(f"   * Type Hint Coverage: {current_metrics.type_hint_coverage:.1f}%")
    lines.append(f"   * Docstring Coverage: {current_metrics.docstring_coverage:.1f}%")
    lines.append(f"   * Any Type Usage: {current_metrics.any_type_percentage:.1f}%")
    lines.append(f"   * str Type Usage: {current_metrics.str_type_percentage:.1f}%")

    if delta.files_changed:
        lines.append("")
        lines.append("**Biggest complexity changes**:")
        sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for file_path, change in sorted_changes:
            truncated = truncate_path(file_path, max_width=50)
            if change > 0:
                lines.append(f"   * {truncated}: +{change}")
            else:
                lines.append(f"   * {truncated}: {change}")

    if baseline_feedback:
        lines.append(baseline_feedback)

    if context_feedback:
        lines.append(context_feedback)

    lines.append("")
    if delta.total_complexity_change > 20:
        lines.append("**Consider**: Breaking down complex functions or refactoring to reduce cognitive load.")
    elif delta.total_complexity_change > 0:
        lines.append("**Note**: Slight complexity increase. Monitor for future refactoring opportunities.")
    elif delta.total_complexity_change < -10:
        lines.append("**Great work**: Complexity reduction makes the code more maintainable!")

    return "\n".join(lines)


def format_complexity_feedback(
    current_metrics: "ExtendedComplexityMetrics",
    delta: "ComplexityDelta",
    baseline_feedback: str = "",
    context_feedback: str = "",
) -> str:
    """Format complexity delta information for Claude consumption.

    Args:
        current_metrics: Current complexity metrics
        delta: Complexity changes from previous commit
        baseline_feedback: Optional baseline comparison feedback
        context_feedback: Optional context coverage feedback

    Returns:
        Formatted feedback string for Claude
    """
    lines = []

    lines.append("**Complexity Analysis Summary**")
    lines.append("")

    if delta.total_complexity_change > 0:
        lines.append(
            f"**Complexity increased by +{delta.total_complexity_change}** (now {current_metrics.total_complexity} total)"
        )
    elif delta.total_complexity_change < 0:
        lines.append(
            f"**Complexity decreased by {delta.total_complexity_change}** (now {current_metrics.total_complexity} total)"
        )
    else:
        lines.append(f"**No net complexity change** ({current_metrics.total_complexity} total)")

    if delta.files_added:
        truncated_added = [truncate_path(f, max_width=30) for f in delta.files_added[:3]]
        lines.append(f"**Added {len(delta.files_added)} files**: {', '.join(truncated_added)}")
        if len(delta.files_added) > 3:
            lines.append(f"   ... and {len(delta.files_added) - 3} more")

    if delta.files_removed:
        truncated_removed = [truncate_path(f, max_width=30) for f in delta.files_removed[:3]]
        lines.append(f"**Removed {len(delta.files_removed)} files**: {', '.join(truncated_removed)}")
        if len(delta.files_removed) > 3:
            lines.append(f"   ... and {len(delta.files_removed) - 3} more")

    lines.append("")
    lines.append("**Code Quality**:")
    lines.append(f"   • Type Hint Coverage: {current_metrics.type_hint_coverage:.1f}%")
    lines.append(f"   • Docstring Coverage: {current_metrics.docstring_coverage:.1f}%")
    lines.append(f"   • Any Type Usage: {current_metrics.any_type_percentage:.1f}%")
    lines.append(f"   • str Type Usage: {current_metrics.str_type_percentage:.1f}%")

    lines.append("")
    lines.append("")
    lines.append("**Code Smells**:")

    def fmt_smell(label: str, count: int, change: int, files: list[str] | None = None) -> str:
        base_msg = ""
        if change > 0:
            base_msg = f"   • {label}: {count} (+{change})"
        elif change < 0:
            base_msg = f"   • {label}: {count} ({change})"
        else:
            base_msg = f"   • {label}: {count}"

        if files and count > 0:
            truncated_files = [truncate_path(f, max_width=25) for f in files[:3]]
            file_list = ", ".join(truncated_files)
            remaining = len(files) - 3
            if remaining > 0:
                return f"{base_msg} [{file_list}, ... +{remaining}]"
            return f"{base_msg} [{file_list}]"
        return base_msg

    lines.append(
        fmt_smell(
            "Orphan Comments - verify if redundant",
            current_metrics.orphan_comment_count,
            delta.orphan_comment_change,
            current_metrics.orphan_comment_files,
        )
    )
    lines.append(
        fmt_smell(
            "Untracked TODOs",
            current_metrics.untracked_todo_count,
            delta.untracked_todo_change,
            current_metrics.untracked_todo_files,
        )
    )
    lines.append(
        fmt_smell(
            "Inline Imports - verify if they can be moved to the top",
            current_metrics.inline_import_count,
            delta.inline_import_change,
            current_metrics.inline_import_files,
        )
    )
    lines.append(
        fmt_smell(
            ".get() with default - may indicate a silent failure",
            current_metrics.dict_get_with_default_count,
            delta.dict_get_with_default_change,
            current_metrics.dict_get_with_default_files,
        )
    )
    lines.append(
        fmt_smell(
            "Dynamic Attr inspection - may indicate a domain modeling gap, i.e. missing BaseModel",
            current_metrics.hasattr_getattr_count,
            delta.hasattr_getattr_change,
            current_metrics.hasattr_getattr_files,
        )
    )
    lines.append(
        fmt_smell(
            "Logic in __init__ - consider if redundant re-exports can be removed",
            current_metrics.nonempty_init_count,
            delta.nonempty_init_change,
            current_metrics.nonempty_init_files,
        )
    )

    if delta.files_changed:
        lines.append("")
        lines.append("**Biggest complexity changes**:")
        sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for file_path, change in sorted_changes:
            truncated = truncate_path(file_path, max_width=50)
            if change > 0:
                lines.append(f"   • {truncated}: +{change}")
            else:
                lines.append(f"   • {truncated}: {change}")

    if baseline_feedback:
        lines.append(baseline_feedback)

    if context_feedback:
        lines.append(context_feedback)

    lines.append("")

    smell_increases = []
    if delta.orphan_comment_change > 0:
        smell_increases.append("orphan comments")
    if delta.untracked_todo_change > 0:
        smell_increases.append("untracked TODOs")
    if delta.inline_import_change > 0:
        smell_increases.append("inline imports")
    if delta.dict_get_with_default_change > 0:
        smell_increases.append("unsafe dict.get()")
    if delta.hasattr_getattr_change > 0:
        smell_increases.append("dynamic attributes")
    if delta.nonempty_init_change > 0:
        smell_increases.append("logic in __init__.py")

    if smell_increases:
        lines.append(f"**⚠️  Quality Alert**: New code smells introduced: {', '.join(smell_increases)}.")
        lines.append("Please review the 'Code Smells' section above and address these issues before stopping.")
    elif delta.total_complexity_change > 20:
        lines.append("**Consider**: Breaking down complex functions or refactoring to reduce cognitive load.")
    elif delta.total_complexity_change > 0:
        lines.append("**Note**: Slight complexity increase. Monitor for future refactoring opportunities.")
    elif delta.total_complexity_change < -10:
        lines.append("**Great work**: Complexity reduction makes the code more maintainable!")

    return "\n".join(lines)


def detect_event_type_from_parsed(parsed_input: HookInputUnion) -> HookEventType:
    """Detect event type from parsed input model."""
    match parsed_input:
        case PreToolUseInput():
            return HookEventType.PRE_TOOL_USE
        case PostToolUseInput():
            return HookEventType.POST_TOOL_USE
        case NotificationInput():
            return HookEventType.NOTIFICATION
        case StopInput():
            return HookEventType.STOP
        case SubagentStopInput():
            return HookEventType.SUBAGENT_STOP
        case _:
            raise ValueError(f"Unknown input type: {type(parsed_input)}")


def main() -> None:
    """Entry point for hook handler."""
    exit_code = handle_hook()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

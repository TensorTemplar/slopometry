"""Hook handler script invoked by Claude Code for each event."""

import json
import logging
import os
import select
import subprocess
import sys
from pathlib import Path

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.git_tracker import GitTracker
from slopometry.core.lock import SlopometryLock
from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import (
    FeedbackCacheState,
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


def _get_feedback_cache_path(working_directory: str) -> Path:
    """Get path to the feedback cache file for a working directory."""
    cache_dir = Path(working_directory) / ".slopometry"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "feedback_cache.json"


def _get_current_commit_sha(working_directory: str) -> str | None:
    """Get current commit SHA with a single git command.

    This is the cheapest possible git operation (~5ms) used to short-circuit
    the expensive _compute_working_tree_cache_key on the cache-hit path.

    Args:
        working_directory: Path to the git working directory.

    Returns:
        Commit SHA string, or None if not a git repo or git fails.
    """
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
        pass
    return None


def _has_source_changes(working_directory: str) -> bool:
    """Check whether the working tree has any in-scope source delta vs the committed state.

    Used by the fast-path: when the commit SHA is unchanged and this returns False,
    the source content is provably identical to the last fire and the hook can stay
    silent without recomputing the full content key.

    Covers two kinds of change so that the fast-path never suppresses a genuine fire:
      1. Tracked modifications — `git diff --quiet -- *.py *.rs` (staged + unstaged),
         cheapest first. Only when git reports a diff do we enumerate via
         `_get_modified_source_files_from_git` to drop false positives from ignored
         dirs (`__pycache__/*.py`, `.venv/site-packages/*.py`, …) and submodules.
      2. Untracked source files — `git ls-files --others` (filtered). A brand-new
         `.py`/`.rs` file is invisible to `git diff`, so without this check the
         fast-path would silently swallow the fire for newly created source.

    Submodule contents are excluded everywhere (--ignore-submodules=all on diffs;
    `git ls-files` never descends into submodules), so dirty submodules, HEAD pointer
    moves, and user git configs (submodule.recurse, diff.submodule=log) cannot trip it.

    Args:
        working_directory: Path to the git working directory.

    Returns:
        True if any non-ignored source files are modified or newly added.
    """
    wt = WorkingTreeStateCalculator(working_directory, languages=None)

    any_diff = False
    for diff_args in [
        ["git", "diff", "--quiet", "--ignore-submodules=all", "--", "*.py", "*.rs"],
        ["git", "diff", "--cached", "--quiet", "--ignore-submodules=all", "--", "*.py", "*.rs"],
    ]:
        try:
            result = subprocess.run(
                diff_args,
                cwd=working_directory,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                any_diff = True
                break
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return True

    if any_diff and wt._get_modified_source_files_from_git():
        return True

    return bool(wt.get_untracked_source_files())


def _compute_working_tree_cache_key(working_directory: str) -> str:
    """Compute a commit-invariant cache key from working-tree source content.

    The key is a digest over the current content of every non-ignored,
    non-submodule .py/.rs file in the working tree (tracked + untracked). It
    deliberately does NOT include the commit SHA: committing already-written
    code, switching branches, pulling, rebasing, or merging does not change
    source *content* and must not re-fire the hook. The key changes iff source
    bytes change — including the addition of a new untracked source file.

    Args:
        working_directory: Path to the working directory

    Returns:
        Cache key string (BLAKE2b hex digest)
    """
    wt_calculator = WorkingTreeStateCalculator(working_directory, languages=None)
    return wt_calculator.calculate_source_content_key()


def _load_feedback_cache(working_directory: str) -> FeedbackCacheState | None:
    """Load the feedback cache state from disk.

    Returns:
        FeedbackCacheState if cache exists and is valid, None otherwise
    """
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
    """Save the feedback cache state with per-file content hashes.

    Args:
        working_directory: Path to the working directory
        cache_key: Working tree cache key
        file_hashes: Per-file content hashes at the time of this cache save
        commit_sha: Current commit SHA for cheap fast-path validation on next run
    """
    cache_path = _get_feedback_cache_path(working_directory)
    try:
        state = FeedbackCacheState(last_key=cache_key, file_hashes=file_hashes, commit_sha=commit_sha)
        cache_path.write_text(state.model_dump_json())
    except OSError as e:
        logger.debug(f"Failed to save feedback cache: {e}")


def _has_analyzable_source_files(working_directory: str) -> bool:
    """Check if the working directory contains any Python or Rust source files.

    Delegates to GitTracker.has_analyzable_source_files() which owns all
    git-file-listing logic.

    Args:
        working_directory: Path to the working directory to check.

    Returns:
        True if at least one .py or .rs file is found via git ls-files.
    """
    tracker = GitTracker(Path(working_directory))
    return tracker.has_analyzable_source_files()


def _resolve_working_directory(stored_wd: str | None) -> str | None:
    """Resolve the effective working_directory for a stop event.

    `stored_wd` is the working_directory recorded on the FIRST event of the
    session. If the user renamed or moved the repo since that event, the
    stored path no longer points to the live repo — falling through to it
    would read/write the cache at a stale location and every Stop would
    invalidate against an absent cache. Fall back to `os.getcwd()` (the
    hook subprocess inherits Claude Code's cwd, which is the live project
    root) whenever the stored path is set but doesn't resolve to an
    existing directory.

    `stored_wd is None` means the session has no recorded events at all
    (unknown session_id); callers should bail in that case, so we
    propagate the None rather than substituting cwd.
    """
    if stored_wd is None:
        return None
    if Path(stored_wd).is_dir():
        return stored_wd
    cwd = os.getcwd()
    if Path(cwd).is_dir():
        return cwd
    return stored_wd


def handle_stop_event(session_id: str, parsed_input: "StopInput | SubagentStopInput") -> int:
    """Handle Stop events with code smell feedback and optional complexity analysis.

    Code smells are always checked (independent of enable_complexity_feedback).
    Complexity metrics are only shown when enable_complexity_feedback is True.
    Dev guidelines are shown when feedback_dev_guidelines is True.

    Feedback is cached - if the same feedback would be shown twice without code changes,
    the second invocation returns silently.

    The firing key is a commit-invariant digest of working-tree source content
    (see _compute_working_tree_cache_key): it fires only when .py/.rs bytes change,
    never on commits, branch switches, pulls, or non-source churn.

    Optimized execution order (cheapest checks first):
      1. stop_hook_active check              (<1ms)
      2. get_session_working_directory        (<1ms, single SQL)
      3. cheap cache fast-path               (commit SHA hint + source-delta probe)
      4. analyzable source files gate         (git ls-files)
      5. full content key computation         (only on fast-path miss)
      6. get_session_statistics               (only when needed)
      7. use stats.complexity_metrics         (no redundant call)

    Args:
        session_id: The session ID
        parsed_input: The stop event input

    Returns:
        Exit code (0 for success, 2 for blocking with feedback)
    """
    if parsed_input.stop_hook_active:
        return 0

    db = EventDatabase()
    working_directory = _resolve_working_directory(db.get_session_working_directory(session_id))
    if not working_directory:
        return 0

    # Fast-path cache check: when the commit SHA is unchanged AND the source tree
    # has no delta (no tracked modifications, no new untracked source files), the
    # content key is provably identical to the last fire — skip the full key
    # computation. commit_sha here is only a cheap hint; it is NOT part of the key,
    # so a bare commit of unchanged content takes the full-key path below and
    # correctly matches last_key (silent) instead of re-firing.
    cached_state = _load_feedback_cache(working_directory)
    if cached_state is not None and cached_state.commit_sha is not None:
        current_sha = _get_current_commit_sha(working_directory)
        if current_sha == cached_state.commit_sha and not _has_source_changes(working_directory):
            return 0

    if not _has_analyzable_source_files(working_directory):
        return 0

    # Full cache key — only reached when source files exist AND fast-path didn't match
    cache_key = _compute_working_tree_cache_key(working_directory)
    if cached_state is not None and cached_state.last_key == cache_key:
        return 0

    stats = db.get_session_statistics(session_id)
    if not stats:
        return 0

    current_metrics = stats.complexity_metrics
    delta = stats.complexity_delta

    # Determine which files changed since the last time feedback was shown.
    # Uses per-file content hashes from the feedback cache to filter out
    # pre-existing uncommitted changes that haven't changed.
    wt_calculator = WorkingTreeStateCalculator(working_directory, languages=None)

    current_file_hashes = wt_calculator.get_source_file_content_hashes()

    if cached_state is not None:
        edited_files = wt_calculator.get_files_changed_since(cached_state.file_hashes)
    else:
        # No cache yet (first run) — treat all modified source files as edited
        edited_files = wt_calculator.get_modified_source_file_paths()

    feedback_parts: list[str] = []

    # Smell feedback: split into code-based (stable) and context-derived (unstable)
    # Context-derived smells (e.g., unread_related_tests) change with every transcript
    # read and must NOT be included in the cache hash to avoid repeated triggers
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

    # Context coverage - informational but NOT stable (changes with every Read/Glob/Grep)
    if settings.enable_complexity_feedback and stats.context_coverage and stats.context_coverage.has_gaps:
        context_feedback = format_context_coverage_feedback(stats.context_coverage)
        if context_feedback:
            feedback_parts.append(context_feedback)

    if settings.feedback_dev_guidelines:
        dev_guidelines = extract_dev_guidelines_from_claude_md(working_directory)
        if dev_guidelines:
            feedback_parts.append(f"\n**Project Development Guidelines:**\n{dev_guidelines}")

    # Save cache with current file hashes regardless of whether feedback is shown.
    # This ensures the next stop event compares against this point in time.
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
    # REASON: acknowledged_silent_except is acceptable individually but blocks on INCREASE — an `# slopometry: allow-silent` marker moves a handler out of swallowed_exception, so a rising marker count is the anti-reward-hack signal that new suppressions need justifying.
    block_on_increase_names = {"acknowledged_silent_except"}

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

        is_blocking_smell = smell.name in blocking_smell_names or (smell.name in block_on_increase_names and change > 0)

        if is_blocking_smell and edited_files:
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
) -> tuple[str, bool, bool]:
    """Format pre-classified smell data into feedback output.

    Args:
        scoped_smells: Pre-classified smells from scope_smells_for_session
        session_id: Session ID for generating the smell-details command

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

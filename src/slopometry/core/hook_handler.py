"""Hook handler script invoked by Claude Code for each event."""

import json
import os
import sys
from pathlib import Path

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.git_tracker import GitTracker
from slopometry.core.models import (
    ComplexityDelta,
    ComplexityMetrics,
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
from slopometry.core.project_tracker import ProjectTracker
from slopometry.core.settings import settings


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
        return SubagentStopInput(**raw_data)

    else:
        raise ValueError(f"Unknown hook input schema with fields: {fields}")


def handle_hook(event_type_override: HookEventType | None = None) -> int:
    """Main hook handler function.

    Args:
        event_type_override: Optional override for the event type, used when called via specific hook entrypoints
    """
    stdin_input = sys.stdin.read().strip()
    if not stdin_input:
        return 0

    raw_data = json.loads(stdin_input)

    parsed_input = parse_hook_input(raw_data)

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

    try:
        db = EventDatabase()
        db.save_event(event)
    except KeyError:
        # This can happen if the project table is not found in the pyproject.toml
        # In this case, we just ignore the event and don't block the user.
        pass

    if event_type in (HookEventType.STOP, HookEventType.SUBAGENT_STOP) and settings.enable_complexity_analysis:
        # mypy: parsed_input is StopInput or SubagentStopInput based on event_type check above
        return handle_stop_event(session_id, parsed_input)  # type: ignore[arg-type]

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


def handle_stop_event(session_id: str, parsed_input: "StopInput | SubagentStopInput") -> int:
    """Handle Stop events with complexity analysis and optional feedback to Claude.

    Args:
        session_id: The session ID
        parsed_input: The stop event input

    Returns:
        Exit code (0 for success, 2 for blocking with feedback)
    """
    if parsed_input.stop_hook_active:
        return 0

    try:
        db = EventDatabase()
        stats = db.get_session_statistics(session_id)

        if not stats:
            return 0

        current_metrics, delta = db.calculate_complexity_metrics(stats.working_directory)

        # TODO: Store complexity metrics in database for analytics

        if not settings.enable_complexity_feedback:
            return 0

        if not delta or not current_metrics:
            return 0

        if abs(delta.total_complexity_change) < 5 and not delta.files_added and not delta.files_removed:
            return 0

        feedback = format_complexity_feedback(current_metrics, delta)

        hook_output = {"decision": "block", "reason": feedback}

        print(json.dumps(hook_output))
        return 2

    except Exception:
        return 0


def format_complexity_feedback(current_metrics: "ComplexityMetrics", delta: "ComplexityDelta") -> str:
    """Format complexity delta information for Claude consumption.

    Args:
        current_metrics: Current complexity metrics
        delta: Complexity changes from previous commit

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
        lines.append(f"**Added {len(delta.files_added)} files**: {', '.join(delta.files_added[:3])}")
        if len(delta.files_added) > 3:
            lines.append(f"   ... and {len(delta.files_added) - 3} more")

    if delta.files_removed:
        lines.append(f"**Removed {len(delta.files_removed)} files**: {', '.join(delta.files_removed[:3])}")
        if len(delta.files_removed) > 3:
            lines.append(f"   ... and {len(delta.files_removed) - 3} more")

    if delta.files_changed:
        lines.append("")
        lines.append("**Biggest complexity changes**:")
        sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for file_path, change in sorted_changes:
            if change > 0:
                lines.append(f"   • {file_path}: +{change}")
            else:
                lines.append(f"   • {file_path}: {change}")

    lines.append("")
    if delta.total_complexity_change > 20:
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


def main() -> None:
    """Entry point for hook handler."""
    exit_code = handle_hook()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

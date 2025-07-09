"""Hook handler script invoked by Claude Code for each event."""

import json
import sys

from .database import EventDatabase, SessionManager
from .git_tracker import GitTracker
from .models import (
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
from .settings import settings


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
        # MCP (Model Context Protocol) tools
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

    # Handle generic MCP tools that don't match specific mappings
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


def handle_hook():
    """Main hook handler function."""
    stdin_input = sys.stdin.read().strip()
    if not stdin_input:
        return 0

    raw_data = json.loads(stdin_input)

    parsed_input = parse_hook_input(raw_data)

    event_type = _detect_event_type_from_parsed(parsed_input)

    session_id = parsed_input.session_id

    session_manager = SessionManager()
    sequence_number = session_manager.get_next_sequence_number(session_id)

    # Track git state for session start and notifications
    git_state = None
    if event_type in (HookEventType.PRE_TOOL_USE, HookEventType.NOTIFICATION) and sequence_number == 1:
        # First event in session - capture initial git state
        git_tracker = GitTracker()
        git_state = git_tracker.get_git_state()
    elif event_type == HookEventType.NOTIFICATION:
        # Notification event - capture current git state
        git_tracker = GitTracker()
        git_state = git_tracker.get_git_state()

    event = HookEvent(
        session_id=session_id,
        event_type=event_type,
        sequence_number=sequence_number,
        metadata=raw_data,
        git_state=git_state,
    )

    if isinstance(parsed_input, PreToolUseInput | PostToolUseInput):
        event.tool_name = parsed_input.tool_name
        event.tool_type = get_tool_type(parsed_input.tool_name)

        # For PostToolUse, extract timing from tool_response
        if isinstance(parsed_input, PostToolUseInput):
            # Handle both string and dictionary responses
            if isinstance(parsed_input.tool_response, dict):
                event.duration_ms = parsed_input.tool_response.get("duration_ms")
                event.exit_code = parsed_input.tool_response.get("exit_code")
                event.error_message = parsed_input.tool_response.get("error")
            else:
                # For string responses, we can't extract timing info
                event.duration_ms = None
                event.exit_code = None
                event.error_message = None

    db = EventDatabase()
    db.save_event(event)

    # Handle Stop/SubagentStop events with complexity delta feedback (if enabled)
    if event_type in (HookEventType.STOP, HookEventType.SUBAGENT_STOP) and settings.enable_stop_feedback:
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
        print(f"Slopometry captured: {json.dumps(debug_info, indent=2)}")

    return 0


def handle_stop_event(session_id: str, parsed_input: "StopInput | SubagentStopInput") -> int:
    """Handle Stop events with complexity delta feedback to Claude.

    Args:
        session_id: The session ID
        parsed_input: The stop event input

    Returns:
        Exit code (0 for success, 2 for blocking with feedback)
    """
    # Check if stop hook is already active to prevent infinite loops
    if parsed_input.stop_hook_active:
        return 0

    try:
        # Calculate session statistics with complexity delta
        db = EventDatabase()
        stats = db.get_session_statistics(session_id)

        if not stats or not stats.complexity_delta:
            # No complexity data available, let Claude stop normally
            return 0

        delta = stats.complexity_delta

        # Check if there are significant complexity changes worth reporting
        if abs(delta.total_complexity_change) < 5 and not delta.files_added and not delta.files_removed:
            # Minor or no changes, let Claude stop normally
            return 0

        # Format complexity feedback for Claude
        feedback = format_complexity_feedback(stats.complexity_metrics, delta)

        # Output JSON feedback to stdout for Claude to read
        hook_output = {"decision": "block", "reason": feedback}

        print(json.dumps(hook_output))
        return 2  # Block Claude from stopping and show feedback

    except Exception:
        # If anything fails, don't block Claude from stopping
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

    # Session impact summary
    lines.append("**Complexity Analysis Summary**")
    lines.append("")

    # Total complexity change
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

    # File changes
    if delta.files_added:
        lines.append(f"**Added {len(delta.files_added)} files**: {', '.join(delta.files_added[:3])}")
        if len(delta.files_added) > 3:
            lines.append(f"   ... and {len(delta.files_added) - 3} more")

    if delta.files_removed:
        lines.append(f"**Removed {len(delta.files_removed)} files**: {', '.join(delta.files_removed[:3])}")
        if len(delta.files_removed) > 3:
            lines.append(f"   ... and {len(delta.files_removed) - 3} more")

    # Biggest complexity changes
    if delta.files_changed:
        lines.append("")
        lines.append("**Biggest complexity changes**:")
        sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for file_path, change in sorted_changes:
            if change > 0:
                lines.append(f"   • {file_path}: +{change}")
            else:
                lines.append(f"   • {file_path}: {change}")

    # Complexity guidance
    lines.append("")
    if delta.total_complexity_change > 20:
        lines.append("**Consider**: Breaking down complex functions or refactoring to reduce cognitive load.")
    elif delta.total_complexity_change > 0:
        lines.append("**Note**: Slight complexity increase. Monitor for future refactoring opportunities.")
    elif delta.total_complexity_change < -10:
        lines.append("**Great work**: Complexity reduction makes the code more maintainable!")

    return "\n".join(lines)


def _detect_event_type_from_parsed(parsed_input: HookInputUnion) -> HookEventType:
    """Detect event type from parsed input model."""
    if isinstance(parsed_input, PreToolUseInput):
        return HookEventType.PRE_TOOL_USE
    elif isinstance(parsed_input, PostToolUseInput):
        return HookEventType.POST_TOOL_USE
    elif isinstance(parsed_input, NotificationInput):
        return HookEventType.NOTIFICATION
    elif isinstance(parsed_input, StopInput):
        return HookEventType.STOP
    elif isinstance(parsed_input, SubagentStopInput):
        return HookEventType.SUBAGENT_STOP
    else:
        # Final fallback for unknown types
        return HookEventType.NOTIFICATION


def main():
    """Entry point for hook handler."""
    exit_code = handle_hook()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

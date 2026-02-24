"""OpenCode hook handler - processes events forwarded by the OpenCode TypeScript plugin.

The OpenCode plugin (plugins/opencode/index.ts) captures in-process events and spawns:
    slopometry hook-opencode --event-type <type>
with JSON on stdin. This module parses the JSON, maps it to HookEvent, and stores it.
"""

import json
import logging
import os
import select
import sys

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.git_tracker import GitTracker
from slopometry.core.lock import SlopometryLock
from slopometry.core.models.hook import (
    EventSource,
    HookEvent,
    HookEventType,
    ToolType,
)
from slopometry.core.models.opencode import (
    OpenCodeMessageEvent,
    OpenCodeSessionEvent,
    OpenCodeStopEvent,
    OpenCodeTodoEvent,
    OpenCodeToolEvent,
)
from slopometry.core.project_tracker import ProjectTracker
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)

# Map OpenCode event type strings to HookEventType
EVENT_TYPE_MAP: dict[str, HookEventType] = {
    "pre_tool_use": HookEventType.PRE_TOOL_USE,
    "post_tool_use": HookEventType.POST_TOOL_USE,
    "stop": HookEventType.STOP,
    "subagent_stop": HookEventType.SUBAGENT_STOP,
    "subagent_start": HookEventType.SUBAGENT_START,
    "todo_updated": HookEventType.TODO_UPDATED,
    "message_updated": HookEventType.MESSAGE_UPDATED,
}


def get_tool_type(tool_name: str) -> ToolType:
    """Map OpenCode tool name to ToolType enum.

    OpenCode uses the same tool names as Claude Code (Bash, Read, Edit, etc.)
    plus some OpenCode-specific ones.
    """
    from slopometry.core.hook_handler import get_tool_type as cc_get_tool_type

    return cc_get_tool_type(tool_name)


def parse_opencode_event(
    event_type: str, raw_data: dict
) -> OpenCodeToolEvent | OpenCodeTodoEvent | OpenCodeMessageEvent | OpenCodeSessionEvent | OpenCodeStopEvent:
    """Parse raw JSON into the appropriate OpenCode event model.

    Args:
        event_type: The event type string from --event-type CLI arg.
        raw_data: Parsed JSON from stdin.

    Returns:
        Typed event model instance.

    Raises:
        ValueError: If event_type is unknown.
    """
    match event_type:
        case "pre_tool_use" | "post_tool_use":
            return OpenCodeToolEvent(**raw_data)
        case "todo_updated":
            return OpenCodeTodoEvent(**raw_data)
        case "message_updated":
            return OpenCodeMessageEvent(**raw_data)
        case "subagent_start":
            return OpenCodeSessionEvent(**raw_data)
        case "stop" | "subagent_stop":
            return OpenCodeStopEvent(**raw_data)
        case _:
            raise ValueError(f"Unknown OpenCode event type: {event_type}")


def _read_stdin_with_timeout(timeout_seconds: float = 5.0) -> str:
    """Read stdin with a timeout to prevent hanging on unclosed pipes."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        return ""
    return sys.stdin.read().strip()


def handle_opencode_hook(event_type: str) -> int:
    """Main entry point for handling OpenCode events.

    Called from CLI: slopometry hook-opencode --event-type <type>

    Reads JSON from stdin, parses it, creates a HookEvent, and stores it.
    For stop events, runs feedback analysis and prints feedback to stdout.

    Args:
        event_type: Event type string (pre_tool_use, post_tool_use, stop, etc.)

    Returns:
        Exit code (0 for success, 2 for blocking feedback).
    """
    try:
        stdin_input = _read_stdin_with_timeout()
    except Exception:
        return 0
    if not stdin_input:
        return 0

    try:
        raw_data = json.loads(stdin_input)
        parsed_event = parse_opencode_event(event_type, raw_data)
    except Exception as e:
        if settings.debug_mode:
            print(f"Slopometry: Failed to parse OpenCode event: {e}", file=sys.stderr)
        return 0

    lock = SlopometryLock(project_dir=os.getcwd())
    with lock.acquire() as acquired:
        if not acquired:
            print("Slopometry: Could not acquire lock, skipping.", file=sys.stderr)
            return 0

        return _handle_opencode_internal(event_type, parsed_event, raw_data)


def _handle_opencode_internal(
    event_type: str,
    parsed_event: (
        OpenCodeToolEvent | OpenCodeTodoEvent | OpenCodeMessageEvent | OpenCodeSessionEvent | OpenCodeStopEvent
    ),
    raw_data: dict,
) -> int:
    """Internal handler for OpenCode events (runs under lock).

    Maps OpenCode event data to the shared HookEvent model and stores it.
    For stop events, runs the same feedback pipeline as Claude Code.

    Returns:
        Exit code.
    """
    try:
        hook_event_type = EVENT_TYPE_MAP.get(event_type)
        if not hook_event_type:
            if settings.debug_mode:
                print(f"Slopometry: Unknown event type '{event_type}'", file=sys.stderr)
            return 0

        # Extract session_id from the parsed event
        session_id = _get_session_id(parsed_event)
        if not session_id:
            return 0

        session_manager = SessionManager()
        sequence_number = session_manager.get_next_sequence_number(session_id)

        working_directory = os.getcwd()
        project_tracker = ProjectTracker(working_dir=__import__("pathlib").Path(working_directory))
        project = project_tracker.get_project()

        # Get git state for first event or stop events
        git_tracker = GitTracker()
        git_state = None
        if sequence_number == 1 or event_type in ("stop", "subagent_stop"):
            git_state = git_tracker.get_git_state()

        # Build the HookEvent
        event = HookEvent(
            session_id=session_id,
            event_type=hook_event_type,
            sequence_number=sequence_number,
            metadata=raw_data,
            git_state=git_state,
            working_directory=working_directory,
            project=project,
            source=EventSource.OPENCODE,
            parent_session_id=_get_parent_id(parsed_event),
        )

        # Set tool-specific fields for tool events
        if isinstance(parsed_event, OpenCodeToolEvent):
            event.tool_name = parsed_event.tool
            event.tool_type = get_tool_type(parsed_event.tool)
            if event_type == "post_tool_use":
                event.duration_ms = parsed_event.duration_ms

        db = EventDatabase()
        db.save_event(event)

        # Handle stop events with feedback
        if event_type in ("stop", "subagent_stop") and settings.enable_complexity_analysis:
            return _handle_opencode_stop(session_id, parsed_event, event_type)

        if settings.debug_mode:
            debug_info = {
                "slopometry_opencode_event": {
                    "session_id": session_id,
                    "event_type": event_type,
                    "hook_event_type": hook_event_type.value,
                    "sequence_number": sequence_number,
                    "source": "opencode",
                }
            }
            print(f"Slopometry captured: {json.dumps(debug_info, indent=2)}", file=sys.stderr)

        return 0

    except Exception as e:
        import traceback

        if settings.debug_mode:
            print(f"Slopometry OpenCode hook error: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return 0


def _handle_opencode_stop(
    session_id: str,
    parsed_event: OpenCodeStopEvent
    | OpenCodeToolEvent
    | OpenCodeTodoEvent
    | OpenCodeMessageEvent
    | OpenCodeSessionEvent,
    event_type: str,
) -> int:
    """Handle stop events from OpenCode with feedback generation.

    Reuses the same feedback pipeline as Claude Code (code smells, context coverage).
    Output is printed to stdout so the OpenCode plugin can capture and inject it.

    Returns:
        Exit code (0 or 2 for blocking feedback).
    """
    if not isinstance(parsed_event, OpenCodeStopEvent):
        return 0

    from slopometry.core.hook_handler import handle_stop_event
    from slopometry.core.models.hook import StopInput

    # Create a StopInput-compatible object for reuse of the existing feedback pipeline
    stop_input = StopInput(
        session_id=session_id,
        transcript_path="",  # OpenCode doesn't use file-based transcripts
        stop_hook_active=False,
    )

    return handle_stop_event(session_id, stop_input)


def _get_session_id(
    event: (OpenCodeToolEvent | OpenCodeTodoEvent | OpenCodeMessageEvent | OpenCodeSessionEvent | OpenCodeStopEvent),
) -> str | None:
    """Extract session_id from any OpenCode event type."""
    return event.session_id


def _get_parent_id(
    event: (OpenCodeToolEvent | OpenCodeTodoEvent | OpenCodeMessageEvent | OpenCodeSessionEvent | OpenCodeStopEvent),
) -> str | None:
    """Extract parent_id from events that have it (session/stop events)."""
    if isinstance(event, OpenCodeSessionEvent | OpenCodeStopEvent):
        return event.parent_id
    return None

"""OpenCode adapter: maps plugin event type strings onto the canonical taxonomy."""

from slopometry.core.protocol.kinds import EventKind

OPENCODE_EVENT_KIND_MAP: dict[str, EventKind] = {
    "pre_tool_use": EventKind.TOOL_CALL,
    "post_tool_use": EventKind.TOOL_RESULT,
    "stop": EventKind.STOP,
    "subagent_stop": EventKind.SUBAGENT_STOP,
    "subagent_start": EventKind.SUBAGENT_START,
    "todo_updated": EventKind.TODO_UPDATED,
    "message_updated": EventKind.MESSAGE_UPDATED,
}

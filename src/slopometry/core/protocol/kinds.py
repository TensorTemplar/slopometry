"""Canonical, harness-independent taxonomy for the extracted hook protocol.

This module defines the open-world vocabulary of the slopometry event
protocol: event kinds are a closed, validated taxonomy, while event sources
are open - any agent tool can ingest events without patching core code.
"""

from enum import StrEnum


class EventKind(StrEnum):
    """Canonical event taxonomy shared by all agent sources.

    Harness-specific event names (Claude Code hook names, OpenCode plugin
    events, third-party trace kinds) are mapped onto these kinds at the
    adapter boundary and never stored or consumed downstream.

    Extension policy: new kinds are added only when a real adapter needs a
    semantic that existing kinds cannot express; kinds are never added
    speculatively. Adapters mapping several source concepts onto one kind
    (e.g. informational messages onto `notification`) should preserve the
    distinction in the stored event's metadata instead.
    """

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NOTIFICATION = "notification"
    STOP = "stop"
    SUBAGENT_STOP = "subagent_stop"
    SUBAGENT_START = "subagent_start"
    TODO_UPDATED = "todo_updated"
    MESSAGE_UPDATED = "message_updated"


class KnownSource(StrEnum):
    """Built-in event sources shipped with slopometry.

    This enum provides constants for the first-party adapters only; the
    protocol itself is open and any source string is valid for ingested
    events. Do NOT gate functionality on membership in this enum.
    """

    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"


LEGACY_EVENT_TYPE_MAP: dict[str, EventKind] = {
    "PreToolUse": EventKind.TOOL_CALL,
    "PostToolUse": EventKind.TOOL_RESULT,
    "Notification": EventKind.NOTIFICATION,
    "Stop": EventKind.STOP,
    "SubagentStop": EventKind.SUBAGENT_STOP,
    "TodoUpdated": EventKind.TODO_UPDATED,
    "MessageUpdated": EventKind.MESSAGE_UPDATED,
    "SubagentStart": EventKind.SUBAGENT_START,
}

# Dialect produced by a parallel abstract-protocol implementation that used
# started/completed suffixes and turn_completed for the stop lifecycle.
# Canonicalized in Migration019 so both dialects coalesce onto one taxonomy.
ALT_DIALECT_EVENT_TYPE_MAP: dict[str, EventKind] = {
    "tool_call_started": EventKind.TOOL_CALL,
    "tool_call_completed": EventKind.TOOL_RESULT,
    "turn_completed": EventKind.STOP,
    "subagent_started": EventKind.SUBAGENT_START,
    "subagent_completed": EventKind.SUBAGENT_STOP,
}

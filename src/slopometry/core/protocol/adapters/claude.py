"""Claude Code adapter: maps harness hook names onto the canonical taxonomy."""

from slopometry.core.protocol.kinds import EventKind

CLAUDE_HOOK_KIND_MAP: dict[str, EventKind] = {
    "PreToolUse": EventKind.TOOL_CALL,
    "PostToolUse": EventKind.TOOL_RESULT,
    "Notification": EventKind.NOTIFICATION,
    "Stop": EventKind.STOP,
    "SubagentStop": EventKind.SUBAGENT_STOP,
}

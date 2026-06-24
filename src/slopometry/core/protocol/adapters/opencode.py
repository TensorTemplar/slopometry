"""OpenCode hook adapter — translates OpenCode plugin JSON into AbstractHookEvent.

OpenCode's TypeScript plugin forwards events with an explicit `--event-type`
discriminator on the CLI; the adapter trusts that rather than inferring from
field shape (since the event-type discriminator is the authoritative source).

OpenCode uses different field names from Claude Code:
  `tool` -> tool_name
  `args` -> input
  `output` -> output
  `duration_ms` is a top-level field, not nested in the response
  `parent_id` -> parent_session_id
"""

from datetime import datetime
from typing import Any

from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
    ToolCallPayload,
)
from slopometry.core.protocol.adapters.claude_code import resolve_tool_type

_OPENCODE_TYPE_TO_ABSTRACT: dict[str, AbstractEventType] = {
    "pre_tool_use": AbstractEventType.TOOL_CALL_STARTED,
    "post_tool_use": AbstractEventType.TOOL_CALL_COMPLETED,
    "stop": AbstractEventType.TURN_COMPLETED,
    "subagent_stop": AbstractEventType.SUBAGENT_COMPLETED,
    "subagent_start": AbstractEventType.SUBAGENT_STARTED,
    "todo_updated": AbstractEventType.TODO_UPDATED,
    "message_updated": AbstractEventType.MESSAGE_UPDATED,
}


def resolve_opencode_event_type(event_type: str) -> AbstractEventType:
    if event_type not in _OPENCODE_TYPE_TO_ABSTRACT:
        raise ValueError(f"Unknown OpenCode event type: {event_type!r}")
    return _OPENCODE_TYPE_TO_ABSTRACT[event_type]


class OpenCodeAdapter:
    source = AbstractEventSource.OPENCODE
    tool_type_map: dict[str, str] = {}

    def detect_event_type(self, raw_payload: dict[str, Any]) -> AbstractEventType:
        event_type = raw_payload.get("event_type")
        if not isinstance(event_type, str):
            raise ValueError("OpenCode payload missing string 'event_type' field")
        return resolve_opencode_event_type(event_type)

    def parse(
        self,
        raw_payload: dict[str, Any],
        *,
        working_directory: str,
        timestamp: datetime | None = None,
        event_type_override: AbstractEventType | None = None,
    ) -> AbstractHookEvent:
        if "session_id" not in raw_payload:
            raise ValueError("OpenCode payload missing required 'session_id' field")

        event_type = event_type_override or self.detect_event_type(raw_payload)
        session_id = raw_payload["session_id"]
        parent_session_id = raw_payload.get("parent_id")

        tool_call: ToolCallPayload | None = None
        if event_type in (AbstractEventType.TOOL_CALL_STARTED, AbstractEventType.TOOL_CALL_COMPLETED):
            tool_name = raw_payload.get("tool")
            if not tool_name:
                raise ValueError(f"OpenCode {event_type.value} payload missing 'tool' field")
            tool_input = raw_payload.get("args") or {}
            tool_output = raw_payload.get("output")
            tool_call = ToolCallPayload(
                tool_name=tool_name,
                tool_type=resolve_tool_type(tool_name),
                input=tool_input,
                output=tool_output,
                duration_ms=raw_payload.get("duration_ms"),
                exit_code=None,
                error_message=None,
            )

        return AbstractHookEvent(
            session_id=session_id,
            parent_session_id=parent_session_id,
            event_type=event_type,
            source=self.source,
            timestamp=timestamp or datetime.now(),
            tool_call=tool_call,
            metadata=dict(raw_payload),
            working_directory=working_directory,
            transcript_location=None,
        )

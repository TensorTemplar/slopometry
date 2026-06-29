"""HookEventAdapter protocol — the contract every harness adapter implements.

An adapter owns:
  - the wire-format field names of its source harness
  - the tool-type vocabulary it recognizes
  - the rule for inferring event_type from a raw payload (when the harness
    doesn't provide an explicit discriminator)
"""

from datetime import datetime
from typing import Any, Protocol

from slopometry.core.models.protocol.events import AbstractEventSource, AbstractEventType, AbstractHookEvent


class HookEventAdapter(Protocol):
    source: AbstractEventSource
    tool_type_map: dict[str, str]

    def parse(
        self,
        raw_payload: dict[str, Any],
        *,
        working_directory: str,
        timestamp: datetime | None = None,
        event_type_override: AbstractEventType | None = None,
    ) -> AbstractHookEvent:
        """Translate a wire payload into a canonical AbstractHookEvent.

        Must populate: session_id, event_type, source, tool_call (when applicable),
        metadata (raw_payload), working_directory, timestamp.

        Raises:
            ValueError: If the payload is not a valid event from this harness.
        """
        ...

    def detect_event_type(self, raw_payload: dict[str, Any]) -> AbstractEventType:
        """Infer event_type from payload shape when the harness doesn't say.

        Used when the harness emits the same payload shape for multiple event
        types (e.g., Claude Code's `session_id + transcript_path + stop_hook_active`
        is ambiguous between Stop and SubagentStop).
        """
        ...


ADAPTERS: dict[AbstractEventSource, HookEventAdapter] = {}


def register_adapter(adapter: HookEventAdapter) -> None:
    """Register an adapter instance for a specific source."""
    ADAPTERS[adapter.source] = adapter

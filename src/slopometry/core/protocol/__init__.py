"""Harness-agnostic hook event protocol — runtime logic layer.

Adapters translate harness wire payloads into the canonical event schema in
`slopometry.core.models.protocol`. This package owns the dispatcher that
persists events and the per-source session manager.

Data types (AbstractHookEvent, AbstractEventType, AbstractEventSource,
ToolCallPayload) live in `slopometry.core.models.protocol`.
"""

from slopometry.core.protocol.dispatch import dispatch_event, emit_event_from_stdin
from slopometry.core.protocol.session import SessionManager

__all__ = ["SessionManager", "dispatch_event", "emit_event_from_stdin"]

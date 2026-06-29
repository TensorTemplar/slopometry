"""Abstract hook event protocol — data-only Pydantic models.

The canonical, harness-agnostic event schema. Adapters translate wire payloads
into these; storage and analytics operate on them.

Runtime logic (adapters, dispatch, session management) lives in
`slopometry.core.protocol`. Models live here so the `core.models` package
holds only pure Pydantic ADTs.
"""

from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
    ToolCallPayload,
)

__all__ = ["AbstractEventSource", "AbstractEventType", "AbstractHookEvent", "ToolCallPayload"]

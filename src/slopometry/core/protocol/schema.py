"""Stable external schema for the hook protocol.

`EventEnvelope` is the public contract for non-Claude-Code / non-OpenCode
agent tools to feed events into slopometry (see issue #46). Collectors emit
JSONL envelopes and hand them to `slopometry ingest` instead of emulating
Claude Code hook payloads.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slopometry.core.protocol.kinds import EventKind

PROTOCOL_SCHEMA_VERSION = 1


class EventEnvelope(BaseModel):
    """One event as ingested over the stable hook protocol.

    Third-party collectors identify events via `source` + `event_id`; the
    combination is unique in storage so re-ingesting a trace file is an
    idempotent backfill.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(default=PROTOCOL_SCHEMA_VERSION, description="Protocol schema version, currently 1")
    source: str = Field(min_length=1, description="Agent tool that produced the event, e.g. 'mmkr'")
    session_id: str = Field(min_length=1, description="Session the event belongs to")
    event_id: str = Field(min_length=1, description="Source-unique event id; required so backfill is always idempotent")
    parent_session_id: str | None = Field(default=None, description="Parent session id for subagent sessions")
    seq: int | None = Field(default=None, ge=1, description="Source-provided sequence number for the event")
    occurred_at: datetime = Field(default_factory=datetime.now, description="When the event occurred at the source")
    kind: EventKind = Field(description="Canonical event kind; validated against the closed taxonomy")
    tool_name: str | None = Field(default=None, description="Tool name for tool_call/tool_result events")
    duration_ms: int | None = Field(default=None, ge=0, description="Tool execution duration for tool_result events")
    exit_code: int | None = Field(default=None, description="Tool exit code for tool_result events")
    error_message: str | None = Field(default=None, description="Tool error message for tool_result events")
    raw: dict[str, Any] = Field(default_factory=dict, description="Source-specific payload kept as event metadata")

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_be_supported(cls, value: int) -> int:
        if value != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {value}; this slopometry version supports 1")
        return value

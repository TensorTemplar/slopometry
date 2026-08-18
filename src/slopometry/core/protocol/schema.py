"""Stable external schema for third-party event ingestion.

`EventEnvelope` is the public contract for agent tools that have no wire-
format adapter (anything beyond Claude Code and OpenCode). Collectors emit
JSONL envelopes and feed them to `slopometry ingest`; the closed
`AbstractEventType` taxonomy and open source strings keep the contract
harness-independent (see issue #46).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slopometry.core.models.protocol.events import AbstractEventType

PROTOCOL_SCHEMA_VERSION = 1


class EventEnvelope(BaseModel):
    """One event as ingested over the stable hook protocol.

    Collectors identify events via `source` + `event_id`; the combination is
    unique in storage, so re-ingesting a trace file is an idempotent backfill.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(default=PROTOCOL_SCHEMA_VERSION, description="Protocol schema version, currently 1")
    source: str = Field(min_length=1, description="Agent tool that produced the event, e.g. 'mmkr'")
    session_id: str = Field(min_length=1, description="Session the event belongs to")
    event_id: str = Field(min_length=1, description="Source-unique event id; required so backfills are always idempotent")
    parent_session_id: str | None = Field(default=None, description="Parent session id for subagent sessions")
    seq: int | None = Field(default=None, ge=1, description="Source-provided sequence number for the event")
    occurred_at: datetime = Field(default_factory=datetime.now, description="When the event occurred at the source")
    kind: AbstractEventType = Field(description="Canonical event kind; validated against the closed taxonomy")
    tool_name: str | None = Field(default=None, description="Tool name for tool_call_started/tool_call_completed events")
    duration_ms: int | None = Field(default=None, ge=0, description="Tool execution duration for tool_call_completed events")
    exit_code: int | None = Field(default=None, description="Tool exit code for tool_call_completed events")
    error_message: str | None = Field(default=None, description="Tool error message for tool_call_completed events")
    raw: dict[str, Any] = Field(default_factory=dict, description="Source-specific payload kept as event metadata")

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_be_supported(cls, value: int) -> int:
        if value != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {value}; this slopometry version supports 1")
        return value

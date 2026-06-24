"""Canonical, harness-agnostic hook event schema.

The persisted event shape. Every adapter produces an AbstractHookEvent; every
analyzer and storage operation consumes one. No field here may reference a
specific agent's vocabulary or wire-format key names.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from slopometry.core.models.hook import GitState, Project


class AbstractEventSource(StrEnum):
    """Identity of the agent harness or collector that produced the event."""

    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"


class AbstractEventType(StrEnum):
    """Canonical event taxonomy — shared across all harnesses.

    Mapping to wire names is the adapter's responsibility:
      tool_call_started    <- Claude Code "PreToolUse"
      tool_call_completed  <- Claude Code "PostToolUse"
      notification         <- Claude Code "Notification"
      turn_completed       <- Claude Code "Stop"
      subagent_completed   <- Claude Code "SubagentStop"
      todo_updated         <- OpenCode bus event
      message_updated      <- OpenCode bus event
      subagent_started     <- OpenCode bus event
    """

    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    NOTIFICATION = "notification"
    TURN_COMPLETED = "turn_completed"
    SUBAGENT_COMPLETED = "subagent_completed"
    TODO_UPDATED = "todo_updated"
    MESSAGE_UPDATED = "message_updated"
    SUBAGENT_STARTED = "subagent_started"


class ToolCallPayload(BaseModel):
    """Harness-agnostic tool-call payload.

    `tool_name` is the raw name from the source harness; `tool_type` is a
    normalized category assigned by the adapter's tool_type_map. `input` and
    `output` preserve the original payload structure from whichever harness
    produced them — adapters are responsible for any reshaping.
    """

    tool_name: str
    tool_type: str | None = Field(
        default=None,
        description="Normalized tool category from the adapter's tool_type_map; None if unknown",
    )
    input: dict[str, Any]
    output: Any | None = Field(
        default=None,
        description="Tool result: dict | str | list depending on the tool's wire shape",
    )
    duration_ms: int | None = None
    exit_code: int | None = Field(
        default=None,
        description="Process exit code; only meaningful for shell-style tools",
    )
    error_message: str | None = None

    model_config = ConfigDict(extra="forbid")


class AbstractHookEvent(BaseModel):
    """A single hook invocation — the canonical stored event.

    `metadata` carries the complete raw wire payload for forensic and
    re-processing use; downstream analyzers should not read from it but go
    through the typed fields instead.
    """

    id: int | None = Field(
        default=None,
        description="Database autoincrement id; None for in-memory events not yet persisted",
    )
    session_id: str
    parent_session_id: str | None = Field(
        default=None,
        description="Parent session ID for subagent/child sessions; None for top-level",
    )
    event_type: AbstractEventType
    source: AbstractEventSource
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_call: ToolCallPayload | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Full raw wire payload — preserves harness-specific shape for forensics",
    )
    git_state: GitState | None = None
    working_directory: str
    project: Project | None = None
    transcript_location: str | None = Field(
        default=None,
        description="Harness-specific path or hint for post-hoc transcript retrieval",
    )
    sequence_number: int = Field(
        default=0,
        description="Monotonic per-session sequence assigned by SessionManager",
    )

    model_config = ConfigDict(extra="forbid")

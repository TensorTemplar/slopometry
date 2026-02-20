"""OpenCode-specific event models for slopometry integration.

OpenCode sends events via its in-process TypeScript plugin system. These models
define the JSON schema that the plugin forwards to `slopometry hook-opencode`.
"""

from pydantic import BaseModel, Field


class OpenCodeTokenUsage(BaseModel):
    """Per-message token usage from OpenCode's message.updated events."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.reasoning


class OpenCodeToolEvent(BaseModel):
    """Event data for tool.execute.before and tool.execute.after hooks.

    Sent by the OpenCode plugin on each tool invocation.
    """

    tool: str = Field(description="Tool name (e.g., 'Bash', 'Read', 'Edit', 'Task')")
    session_id: str = Field(description="OpenCode session ID")
    call_id: str = Field(description="Unique tool call ID within the session")
    args: dict = Field(default_factory=dict, description="Tool arguments")
    output: str | None = Field(default=None, description="Tool output (only on after events)")
    duration_ms: int | None = Field(default=None, description="Tool execution duration (only on after events)")
    title: str | None = Field(default=None, description="Tool output title (only on after events)")
    metadata: dict | None = Field(default=None, description="Tool output metadata (only on after events)")

    model_config = {"extra": "allow"}


class OpenCodeTodoItem(BaseModel):
    """A single todo item from OpenCode's todo.updated event."""

    content: str = Field(description="Brief description of the task")
    status: str = Field(description="Current status: pending, in_progress, completed, cancelled")
    priority: str = Field(description="Priority level: high, medium, low")


class OpenCodeTodoEvent(BaseModel):
    """Event data for todo.updated bus events."""

    session_id: str
    todos: list[OpenCodeTodoItem] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class OpenCodeMessageEvent(BaseModel):
    """Event data for message.updated bus events (assistant messages).

    Provides per-message token and cost tracking.
    """

    session_id: str
    message_id: str
    model_id: str | None = None
    provider_id: str | None = None
    agent: str | None = Field(default=None, description="Agent name: general, explore, plan, etc.")
    tokens: OpenCodeTokenUsage = Field(default_factory=OpenCodeTokenUsage)
    cost: float = 0.0

    model_config = {"extra": "allow"}


class OpenCodeSessionEvent(BaseModel):
    """Event data for session.created and session.idle events.

    Tracks session lifecycle including subagent relationships.
    """

    session_id: str
    parent_id: str | None = Field(default=None, description="Non-None indicates a subagent/child session")
    title: str | None = None
    agent: str | None = Field(default=None, description="Agent type for the session")
    model_id: str | None = None

    model_config = {"extra": "allow"}


class OpenCodeStopEvent(BaseModel):
    """Event data for session.idle (stop) events.

    Includes session summary data and optional transcript.
    """

    session_id: str
    parent_id: str | None = None
    agent: str | None = None
    model_id: str | None = None
    tokens: OpenCodeTokenUsage | None = None
    cost: float | None = None
    todos: list[OpenCodeTodoItem] = Field(default_factory=list)
    transcript: list[dict] | None = Field(
        default=None,
        description="Structured transcript: list of {role, parts, tokens, cost, ...}",
    )
    opencode_version: str | None = Field(
        default=None,
        description="OpenCode version from /global/health endpoint",
    )

    model_config = {"extra": "allow"}

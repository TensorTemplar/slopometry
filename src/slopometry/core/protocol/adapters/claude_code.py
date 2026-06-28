"""Claude Code hook adapter — translates Claude Code's stdin JSON schema into AbstractHookEvent.

Claude Code does not send an explicit hook-type discriminator; the adapter
infers it from field presence. See `docs/claude-hooks-doc.md` for the wire
schema. The adapter owns the Claude-Code + MCP tool vocabulary, exposed as
the `ToolType` enum for downstream typed access.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from slopometry.core.models.protocol.events import (
    AbstractEventSource,
    AbstractEventType,
    AbstractHookEvent,
    ToolCallPayload,
)


class ToolType(StrEnum):
    """Known tool types in Claude Code and OpenCode (shared vocabulary)."""

    BASH = "Bash"
    READ = "Read"
    WRITE = "Write"
    EDIT = "Edit"
    MULTI_EDIT = "MultiEdit"
    GREP = "Grep"
    GLOB = "Glob"
    LS = "LS"
    TASK = "Task"
    TODO_READ = "TodoRead"
    TODO_WRITE = "TodoWrite"
    TASK_CREATE = "TaskCreate"
    TASK_UPDATE = "TaskUpdate"
    TASK_LIST = "TaskList"
    TASK_GET = "TaskGet"
    WEB_FETCH = "WebFetch"
    WEB_SEARCH = "WebSearch"
    NOTEBOOK_READ = "NotebookRead"
    NOTEBOOK_EDIT = "NotebookEdit"
    EXIT_PLAN_MODE = "exit_plan_mode"

    MCP_IDE_GET_DIAGNOSTICS = "mcp__ide__getDiagnostics"
    MCP_IDE_EXECUTE_CODE = "mcp__ide__executeCode"
    MCP_IDE_GET_WORKSPACE_INFO = "mcp__ide__getWorkspaceInfo"
    MCP_IDE_GET_FILE_CONTENTS = "mcp__ide__getFileContents"
    MCP_IDE_CREATE_FILE = "mcp__ide__createFile"
    MCP_IDE_DELETE_FILE = "mcp__ide__deleteFile"
    MCP_IDE_RENAME_FILE = "mcp__ide__renameFile"
    MCP_IDE_SEARCH_FILES = "mcp__ide__searchFiles"
    MCP_FILESYSTEM_READ = "mcp__filesystem__read"
    MCP_FILESYSTEM_WRITE = "mcp__filesystem__write"
    MCP_FILESYSTEM_LIST = "mcp__filesystem__list"
    MCP_DATABASE_QUERY = "mcp__database__query"
    MCP_DATABASE_SCHEMA = "mcp__database__schema"
    MCP_WEB_SCRAPE = "mcp__web__scrape"
    MCP_WEB_SEARCH = "mcp__web__search"
    MCP_GITHUB_GET_REPO = "mcp__github__getRepo"
    MCP_GITHUB_CREATE_ISSUE = "mcp__github__createIssue"
    MCP_GITHUB_LIST_ISSUES = "mcp__github__listIssues"
    MCP_SLACK_SEND_MESSAGE = "mcp__slack__sendMessage"
    MCP_SLACK_LIST_CHANNELS = "mcp__slack__listChannels"
    MCP_OTHER = "mcp__other"

    OTHER = "Other"


_TOOL_NAME_TO_TYPE: dict[str, ToolType] = {
    "bash": ToolType.BASH,
    "read": ToolType.READ,
    "write": ToolType.WRITE,
    "edit": ToolType.EDIT,
    "multiedit": ToolType.MULTI_EDIT,
    "grep": ToolType.GREP,
    "glob": ToolType.GLOB,
    "ls": ToolType.LS,
    "task": ToolType.TASK,
    "todoread": ToolType.TODO_READ,
    "todowrite": ToolType.TODO_WRITE,
    "taskcreate": ToolType.TASK_CREATE,
    "taskupdate": ToolType.TASK_UPDATE,
    "tasklist": ToolType.TASK_LIST,
    "taskget": ToolType.TASK_GET,
    "webfetch": ToolType.WEB_FETCH,
    "websearch": ToolType.WEB_SEARCH,
    "notebookread": ToolType.NOTEBOOK_READ,
    "notebookedit": ToolType.NOTEBOOK_EDIT,
    "exit_plan_mode": ToolType.EXIT_PLAN_MODE,
    "mcp__ide__getdiagnostics": ToolType.MCP_IDE_GET_DIAGNOSTICS,
    "mcp__ide__executecode": ToolType.MCP_IDE_EXECUTE_CODE,
    "mcp__ide__getworkspaceinfo": ToolType.MCP_IDE_GET_WORKSPACE_INFO,
    "mcp__ide__getfilecontents": ToolType.MCP_IDE_GET_FILE_CONTENTS,
    "mcp__ide__createfile": ToolType.MCP_IDE_CREATE_FILE,
    "mcp__ide__deletefile": ToolType.MCP_IDE_DELETE_FILE,
    "mcp__ide__renamefile": ToolType.MCP_IDE_RENAME_FILE,
    "mcp__ide__searchfiles": ToolType.MCP_IDE_SEARCH_FILES,
    "mcp__filesystem__read": ToolType.MCP_FILESYSTEM_READ,
    "mcp__filesystem__write": ToolType.MCP_FILESYSTEM_WRITE,
    "mcp__filesystem__list": ToolType.MCP_FILESYSTEM_LIST,
    "mcp__database__query": ToolType.MCP_DATABASE_QUERY,
    "mcp__database__schema": ToolType.MCP_DATABASE_SCHEMA,
    "mcp__web__scrape": ToolType.MCP_WEB_SCRAPE,
    "mcp__web__search": ToolType.MCP_WEB_SEARCH,
    "mcp__github__getrepo": ToolType.MCP_GITHUB_GET_REPO,
    "mcp__github__createissue": ToolType.MCP_GITHUB_CREATE_ISSUE,
    "mcp__github__listissues": ToolType.MCP_GITHUB_LIST_ISSUES,
    "mcp__slack__sendmessage": ToolType.MCP_SLACK_SEND_MESSAGE,
    "mcp__slack__listchannels": ToolType.MCP_SLACK_LIST_CHANNELS,
}


def resolve_tool_type(tool_name: str) -> str:
    """Map a tool name (any harness) to its PascalCase category string.

    Returns `"Other"` for unknown lowercase names and `"mcp__other"` for
    unknown `mcp__`-prefixed names. The returned string is suitable for
    direct storage in `tool_type` columns.
    """
    lowered = tool_name.lower()
    if lowered in _TOOL_NAME_TO_TYPE:
        return _TOOL_NAME_TO_TYPE[lowered].value
    if lowered.startswith("mcp__"):
        return ToolType.MCP_OTHER.value
    return ToolType.OTHER.value


def _extract_tool_response_fields(tool_response: Any) -> tuple[int | None, int | None, str | None]:
    """Pull the three Bash-style fields from Claude Code's tool_response.

    tool_response shape varies by tool: dict (most), str (Bash), list (NotebookRead).
    Only dicts carry the `duration_ms`/`exit_code`/`error` triple.
    """
    if isinstance(tool_response, dict):
        return (
            tool_response.get("duration_ms"),
            tool_response.get("exit_code"),
            tool_response.get("error"),
        )
    return (None, None, None)


class ClaudeCodeAdapter:
    source = AbstractEventSource.CLAUDE_CODE
    tool_type_map: dict[str, str] = {name: enum.value for name, enum in _TOOL_NAME_TO_TYPE.items()}

    @classmethod
    def map_tool_name(cls, tool_name: str) -> str:
        return resolve_tool_type(tool_name)

    def detect_event_type(self, raw_payload: dict[str, Any]) -> AbstractEventType:
        fields = set(raw_payload.keys())
        if "tool_name" in fields and "tool_input" in fields:
            if "tool_response" in fields:
                return AbstractEventType.TOOL_CALL_COMPLETED
            return AbstractEventType.TOOL_CALL_STARTED
        if "message" in fields:
            return AbstractEventType.NOTIFICATION
        if "stop_hook_active" in fields:
            if raw_payload.get("stop_hook_active"):
                return AbstractEventType.SUBAGENT_COMPLETED
            return AbstractEventType.TURN_COMPLETED
        if "session_id" in fields and "transcript_path" in fields:
            return AbstractEventType.TURN_COMPLETED
        raise ValueError(f"Unknown Claude Code hook payload shape: {sorted(fields)}")

    def parse(
        self,
        raw_payload: dict[str, Any],
        *,
        working_directory: str,
        timestamp: datetime | None = None,
        event_type_override: AbstractEventType | None = None,
    ) -> AbstractHookEvent:
        if "session_id" not in raw_payload:
            raise ValueError("Claude Code payload missing required 'session_id' field")

        event_type = event_type_override or self.detect_event_type(raw_payload)
        session_id = raw_payload["session_id"]
        transcript_path = raw_payload.get("transcript_path")

        tool_call: ToolCallPayload | None = None
        if event_type in (AbstractEventType.TOOL_CALL_STARTED, AbstractEventType.TOOL_CALL_COMPLETED):
            tool_name = raw_payload.get("tool_name")
            if not tool_name:
                raise ValueError(f"Claude Code {event_type.value} payload missing 'tool_name'")
            tool_input = raw_payload.get("tool_input") or {}
            tool_response = raw_payload.get("tool_response")
            duration_ms, exit_code, error_message = _extract_tool_response_fields(tool_response)
            tool_call = ToolCallPayload(
                tool_name=tool_name,
                tool_type=resolve_tool_type(tool_name),
                input=tool_input,
                output=tool_response,
                duration_ms=duration_ms,
                exit_code=exit_code,
                error_message=error_message,
            )

        return AbstractHookEvent(
            session_id=session_id,
            event_type=event_type,
            source=self.source,
            timestamp=timestamp or datetime.now(),
            tool_call=tool_call,
            metadata=dict(raw_payload),
            working_directory=working_directory,
            transcript_location=transcript_path,
        )

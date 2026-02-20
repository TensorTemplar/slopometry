"""Hook-related models for Claude Code integration."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentTool(str, Enum):
    """Agent tool that produced the session."""

    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"


class ProjectLanguage(str, Enum):
    """Supported languages for complexity analysis."""

    PYTHON = "python"
    RUST = "rust"


class ProjectSource(str, Enum):
    """Source of project identification."""

    GIT = "git"
    PYPROJECT = "pyproject"


class Project(BaseModel):
    """Represents a project being worked on."""

    name: str
    source: ProjectSource


class HookEventType(str, Enum):
    """Types of hook events in Claude Code."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"


class ToolType(str, Enum):
    """Known tool types in Claude Code."""

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


class AnalysisSource(str, Enum):
    """Source of the impact analysis."""

    UNCOMMITTED_CHANGES = "uncommitted_changes"
    PREVIOUS_COMMIT = "previous_commit"


class GitState(BaseModel):
    """Represents git repository state at a point in time."""

    commit_count: int = 0
    current_branch: str | None = None
    has_uncommitted_changes: bool = False
    is_git_repo: bool = False
    commit_sha: str | None = None


class HookEvent(BaseModel):
    """Represents a single hook invocation event."""

    id: int | None = None
    session_id: str
    event_type: HookEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence_number: int
    tool_name: str | None = None
    tool_type: ToolType | None = None
    metadata: dict = Field(default_factory=dict)
    duration_ms: int | None = None
    exit_code: int | None = None
    error_message: str | None = None
    git_state: GitState | None = None
    working_directory: str
    project: Project | None = None
    transcript_path: str | None = None


class PreToolUseInput(BaseModel):
    """Input structure for PreToolUse hooks based on Claude Code documentation."""

    session_id: str
    transcript_path: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class PostToolUseInput(BaseModel):
    """Input structure for PostToolUse hooks based on Claude Code documentation."""

    session_id: str
    transcript_path: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_response: dict[str, Any] | str | list[Any] = Field(
        default_factory=dict,
        description="Tool response data. Can be dict (most tools), str (Bash output), or list (NotebookRead cells). Uses Any for list items since different tools return different cell structures.",
    )

    model_config = {"extra": "allow"}


class NotificationInput(BaseModel):
    """Input structure for Notification hooks based on Claude Code documentation."""

    session_id: str
    transcript_path: str
    message: str
    title: str | None = None

    model_config = {"extra": "allow"}


class StopInput(BaseModel):
    """Input structure for Stop hooks based on Claude Code documentation."""

    session_id: str
    transcript_path: str
    stop_hook_active: bool = False

    model_config = {"extra": "allow"}


class SubagentStopInput(BaseModel):
    """Input structure for SubagentStop hooks based on Claude Code documentation."""

    session_id: str
    transcript_path: str
    stop_hook_active: bool = False

    model_config = {"extra": "allow"}


HookInputUnion = PreToolUseInput | PostToolUseInput | NotificationInput | StopInput | SubagentStopInput


class HookOutput(BaseModel):
    """Output structure for hook responses based on Claude Code documentation."""

    continue_: bool | None = Field(None, alias="continue")
    stop_reason: str | None = Field(None, alias="stopReason")
    suppress_output: bool | None = Field(None, alias="suppressOutput")
    decision: str | None = Field(default=None, description="Decision outcome: approve, block, or undefined")
    reason: str | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class LanguageGuardResult(BaseModel):
    """Result of language guard check for complexity analysis features."""

    allowed: bool = Field(description="Whether the required language is available for analysis")
    required_language: ProjectLanguage = Field(description="The language required by the feature")
    detected_supported: set[ProjectLanguage] = Field(
        default_factory=set, description="Languages detected in repo that are supported"
    )
    detected_unsupported: set[str] = Field(
        default_factory=set, description="Language names detected but not supported (e.g., 'Rust', 'Go')"
    )

    def format_warning(self) -> str | None:
        """Return warning message if unsupported languages found, else None."""
        if not self.detected_unsupported:
            return None
        return f"Found {', '.join(sorted(self.detected_unsupported))} files but analysis not yet supported"

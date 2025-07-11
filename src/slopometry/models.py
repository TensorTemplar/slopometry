"""Data models for tracking Claude Code hook events."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


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


class GitState(BaseModel):
    """Represents git repository state at a point in time."""

    commit_count: int = 0
    current_branch: str | None = None
    has_uncommitted_changes: bool = False
    is_git_repo: bool = False


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


class ComplexityMetrics(BaseModel):
    """Cognitive complexity metrics for Python files."""

    total_files_analyzed: int = 0
    total_complexity: int = 0
    average_complexity: float = 0.0
    max_complexity: int = 0
    min_complexity: int = 0
    files_by_complexity: dict[str, int] = Field(
        default_factory=dict, description="Mapping of filename to complexity score"
    )


class ComplexityDelta(BaseModel):
    """Complexity change comparison between two versions."""

    total_complexity_change: int = 0
    files_added: list[str] = Field(default_factory=list)
    files_removed: list[str] = Field(default_factory=list)
    files_changed: dict[str, int] = Field(default_factory=dict, description="Mapping of filename to complexity delta")
    net_files_change: int = Field(default=0, description="Net change in number of files (files_added - files_removed)")
    avg_complexity_change: float = 0.0

    total_volume_change: float = 0.0
    avg_volume_change: float = 0.0
    total_difficulty_change: float = 0.0
    avg_difficulty_change: float = 0.0
    total_effort_change: float = 0.0
    total_mi_change: float = 0.0
    avg_mi_change: float = 0.0


class TodoItem(BaseModel):
    """Represents a single todo item."""

    id: str
    content: str
    status: str = Field(description="Status of the todo item: pending, in_progress, or completed")
    priority: str = Field(description="Priority level: high, medium, or low")


class PlanStep(BaseModel):
    """Represents a planning step between TodoWrite events."""

    step_number: int
    events_in_step: int = Field(description="Number of events between this and previous TodoWrite")
    todos_added: list[str] = Field(default_factory=list, description="Content of new todos added in this step")
    todos_removed: list[str] = Field(default_factory=list, description="Content of todos removed in this step")
    todos_status_changed: dict[str, tuple[str, str]] = Field(
        default_factory=dict, description="Mapping of todo content to (old_status, new_status) for status changes"
    )
    todos_content_changed: dict[str, tuple[str, str]] = Field(
        default_factory=dict, description="Mapping of old_content to new_content for content changes"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    search_events: int = Field(default=0, description="Number of search-type tool events")
    implementation_events: int = Field(default=0, description="Number of implementation-type tool events")
    search_to_implementation_ratio: float = Field(
        default=0.0, description="Ratio of search events to implementation events (0 if no implementation events)"
    )


class PlanEvolution(BaseModel):
    """Tracks how the plan evolves through TodoWrite events."""

    total_plan_steps: int = 0
    total_todos_created: int = 0
    total_todos_completed: int = 0
    average_events_per_step: float = 0.0
    plan_steps: list[PlanStep] = Field(default_factory=list)
    final_todo_count: int = 0
    planning_efficiency: float = Field(default=0.0, description="Ratio of completed todos to total todos created")
    total_search_events: int = 0
    total_implementation_events: int = 0
    overall_search_to_implementation_ratio: float = 0.0


class SessionStatistics(BaseModel):
    """Aggregated statistics for a Claude Code session."""

    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    total_events: int = 0
    working_directory: str
    events_by_type: dict[HookEventType, int] = Field(default_factory=dict)
    tool_usage: dict[ToolType, int] = Field(default_factory=dict)
    error_count: int = 0
    total_duration_ms: int = 0
    average_tool_duration_ms: float = 0.0
    initial_git_state: GitState | None = None
    final_git_state: GitState | None = None
    commits_made: int = 0
    complexity_metrics: "ExtendedComplexityMetrics | None" = None
    complexity_delta: ComplexityDelta | None = None
    plan_evolution: PlanEvolution | None = None
    project: Project | None = None


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
    tool_response: dict[str, Any] | str = Field(default_factory=dict)

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


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UserStory(BaseModel):
    """Represents a single user story for feature development."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(description="Short title of the user story")
    description: str = Field(description="Detailed description of the user story")
    acceptance_criteria: list[str] = Field(default_factory=list, description="List of acceptance criteria")
    priority: int = Field(default=1, description="Priority level (1=highest, 5=lowest)")
    estimated_complexity: int = Field(default=0, description="Estimated complexity points")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class NextFeaturePrediction(BaseModel):
    """Next Feature Prediction objective containing user stories."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target_commit: str = Field(description="The commit SHA this NFP targets (e.g., HEAD)")
    base_commit: str = Field(description="The starting commit SHA (e.g., HEAD~1)")
    repository_path: Path = Field(description="Path to the repository this NFP belongs to")
    title: str = Field(description="Overall title for this feature set")
    description: str = Field(description="High-level description of the feature development")
    user_stories: list[UserStory] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def total_estimated_complexity(self) -> int:
        """Calculate total estimated complexity across all user stories."""
        return sum(story.estimated_complexity for story in self.user_stories)

    @property
    def story_count(self) -> int:
        """Get total number of user stories."""
        return len(self.user_stories)

    def get_stories_by_priority(self, priority: int) -> list[UserStory]:
        """Get all user stories with specific priority level."""
        return [story for story in self.user_stories if story.priority == priority]

    def get_high_priority_stories(self) -> list[UserStory]:
        """Get high priority user stories (priority 1-2)."""
        return [story for story in self.user_stories if story.priority <= 2]


class ExtendedComplexityMetrics(BaseModel):
    """Extended metrics including Halstead and Maintainability Index."""

    total_complexity: int = 0
    average_complexity: float = 0.0
    max_complexity: int = 0
    min_complexity: int = 0

    total_volume: float = 0.0
    total_difficulty: float = 0.0
    total_effort: float = 0.0
    average_volume: float = 0.0
    average_difficulty: float = 0.0

    total_mi: float = 0.0
    average_mi: float = Field(default=0.0, description="Higher is better (0-100 scale)")

    total_files_analyzed: int = 0
    files_by_complexity: dict[str, int] = Field(default_factory=dict)


class ExperimentRun(BaseModel):
    """Represents a single experiment run."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    repository_path: Path
    start_commit: str  # SHA of starting commit (e.g., HEAD~1)
    target_commit: str  # SHA of target commit (e.g., HEAD)
    process_id: int
    worktree_path: Path | None = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    nfp_objective: NextFeaturePrediction | None = None  # Feature objectives for this experiment


class ExperimentProgress(BaseModel):
    """Tracks real-time progress with CLI metric."""

    experiment_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    current_metrics: ExtendedComplexityMetrics
    target_metrics: ExtendedComplexityMetrics  # From HEAD commit

    cli_score: float = Field(
        default=0.0, description="Numeric objective: 1.0 = perfect match, <0 = overshooting target"
    )

    complexity_score: float = 0.0
    halstead_score: float = 0.0
    maintainability_score: float = 0.0


class CommitComplexitySnapshot(BaseModel):
    """Complexity metrics for a specific commit."""

    commit_sha: str
    commit_message: str
    timestamp: datetime
    complexity_metrics: ExtendedComplexityMetrics
    parent_commit_sha: str | None = None
    complexity_delta: ComplexityDelta | None = None  # Delta from parent


class CommitChain(BaseModel):
    """Represents a chain of commits with complexity evolution."""

    repository_path: Path
    base_commit: str  # Starting point (e.g., HEAD~10)
    head_commit: str  # End point (e.g., HEAD)
    commits: list[CommitComplexitySnapshot] = Field(default_factory=list)
    total_complexity_growth: int = 0
    average_complexity_per_commit: float = 0.0


class ComplexityEvolution(BaseModel):
    """Tracks how complexity evolves across commits."""

    commit_sha: str
    cumulative_complexity: int  # Total complexity up to this commit
    incremental_complexity: int  # Complexity added in this commit
    files_modified: int
    functions_added: int
    functions_removed: int
    functions_modified: int

"""Data models for tracking Claude Code hook events."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentTool(str, Enum):
    """Agent tool that produced the session."""

    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"


class SmellCategory(str, Enum):
    """Category of code smell for organization and filtering."""

    GENERAL = "general"
    PYTHON = "python"


class SmellDefinition(BaseModel):
    """Definition of a code smell with all metadata."""

    model_config = ConfigDict(frozen=True)

    internal_name: str
    label: str
    category: SmellCategory
    weight: float
    guidance: str
    count_field: str
    files_field: str


SMELL_REGISTRY: dict[str, SmellDefinition] = {
    # General smells (language-agnostic concepts)
    "orphan_comment": SmellDefinition(
        internal_name="orphan_comment",
        label="Orphan Comments",
        category=SmellCategory.GENERAL,
        weight=0.01,
        guidance="Make sure inline code comments add meaningful information about non-obvious design tradeoffs or explain tech debt or performance implications. Consider if these could be docstrings or field descriptors instead",
        count_field="orphan_comment_count",
        files_field="orphan_comment_files",
    ),
    "untracked_todo": SmellDefinition(
        internal_name="untracked_todo",
        label="Untracked TODOs",
        category=SmellCategory.GENERAL,
        weight=0.02,
        guidance="Untracked TODOs should include ticket references (JIRA-123, #123) or URLs",
        count_field="untracked_todo_count",
        files_field="untracked_todo_files",
    ),
    "swallowed_exception": SmellDefinition(
        internal_name="swallowed_exception",
        label="Swallowed Exceptions",
        category=SmellCategory.GENERAL,
        weight=0.15,
        guidance="BLOCKING: You MUST present a table with columns [Location | Purpose | Justification ] for each and ask user to confirm silent failure is acceptable",
        count_field="swallowed_exception_count",
        files_field="swallowed_exception_files",
    ),
    "test_skip": SmellDefinition(
        internal_name="test_skip",
        label="Test Skips",
        category=SmellCategory.GENERAL,
        weight=0.10,
        guidance="BLOCKING: You MUST present a table with columns [Test Name | Intent] for each skip and ask user to confirm skipping is acceptable",
        count_field="test_skip_count",
        files_field="test_skip_files",
    ),
    "type_ignore": SmellDefinition(
        internal_name="type_ignore",
        label="Type Ignores",
        category=SmellCategory.GENERAL,
        weight=0.08,
        guidance="Review type: ignore comments - consider fixing the underlying type issue",
        count_field="type_ignore_count",
        files_field="type_ignore_files",
    ),
    "dynamic_execution": SmellDefinition(
        internal_name="dynamic_execution",
        label="Dynamic Execution",
        category=SmellCategory.GENERAL,
        weight=0.12,
        guidance="Review usage of eval/exec/compile - ensure this is necessary and secure",
        count_field="dynamic_execution_count",
        files_field="dynamic_execution_files",
    ),
    "inline_import": SmellDefinition(
        internal_name="inline_import",
        label="Inline Imports",
        category=SmellCategory.PYTHON,
        weight=0.01,
        guidance="Verify if these can be moved to the top of the file (except TYPE_CHECKING guards)",
        count_field="inline_import_count",
        files_field="inline_import_files",
    ),
    "dict_get_with_default": SmellDefinition(
        internal_name="dict_get_with_default",
        label="Dict .get() Defaults",
        category=SmellCategory.PYTHON,
        weight=0.05,
        guidance="Consider if these are justified or just indicate modeling gaps - replace with Pydantic BaseSettings or BaseModel with narrower field types or raise explicit errors instead",
        count_field="dict_get_with_default_count",
        files_field="dict_get_with_default_files",
    ),
    "hasattr_getattr": SmellDefinition(
        internal_name="hasattr_getattr",
        label="hasattr/getattr",
        category=SmellCategory.PYTHON,
        weight=0.10,
        guidance="Consider if these indicate missing domain models - replace with Pydantic BaseModel objects with explicit fields or raise explicit errors instead",
        count_field="hasattr_getattr_count",
        files_field="hasattr_getattr_files",
    ),
    "nonempty_init": SmellDefinition(
        internal_name="nonempty_init",
        label="Non-empty __init__",
        category=SmellCategory.PYTHON,
        weight=0.03,
        guidance="Consider if implementation code should be moved out of __init__.py files",
        count_field="nonempty_init_count",
        files_field="nonempty_init_files",
    ),
    # Abstraction smells (unnecessary complexity)
    "single_method_class": SmellDefinition(
        internal_name="single_method_class",
        label="Single-Method Classes",
        category=SmellCategory.PYTHON,
        weight=0.05,
        guidance="Consider using a function instead of a class with only one method besides __init__",
        count_field="single_method_class_count",
        files_field="single_method_class_files",
    ),
    "deep_inheritance": SmellDefinition(
        internal_name="deep_inheritance",
        label="Deep Inheritance",
        category=SmellCategory.PYTHON,
        weight=0.08,
        guidance="Prefer composition over inheritance; >2 base classes increases complexity",
        count_field="deep_inheritance_count",
        files_field="deep_inheritance_files",
    ),
    "passthrough_wrapper": SmellDefinition(
        internal_name="passthrough_wrapper",
        label="Pass-Through Wrappers",
        category=SmellCategory.PYTHON,
        weight=0.02,
        guidance="Function that just delegates to another with same args; consider removing indirection",
        count_field="passthrough_wrapper_count",
        files_field="passthrough_wrapper_files",
    ),
    "sys_path_manipulation": SmellDefinition(
        internal_name="sys_path_manipulation",
        label="sys.path Manipulation",
        category=SmellCategory.PYTHON,
        weight=0.10,
        guidance="sys.path mutations bypass the package system — restructure package boundaries and use absolute imports from installed packages instead",
        count_field="sys_path_manipulation_count",
        files_field="sys_path_manipulation_files",
    ),
}


def get_smell_label(internal_name: str) -> str:
    """Get display label for a smell from registry."""
    defn = SMELL_REGISTRY.get(internal_name)
    return defn.label if defn else internal_name.replace("_", " ").title()


def get_smells_by_category(category: SmellCategory) -> list[SmellDefinition]:
    """Get all smells in a category, sorted by weight (highest first)."""
    return sorted(
        [d for d in SMELL_REGISTRY.values() if d.category == category],
        key=lambda d: d.weight,
        reverse=True,
    )


def SmellField(
    default: int = 0,
    *,
    label: str,
    files_field: str,
    guidance: str,
) -> Any:
    """Create a Field for a code smell metric with embedded metadata.

    Args:
        default: Default value (always 0 for counts)
        label: Display label for the smell (e.g., "Orphan Comments")
        files_field: Name of the corresponding files list field
        guidance: Actionable message shown in feedback
    """
    return Field(
        default=default,
        description=guidance,
        json_schema_extra={
            "is_smell": True,
            "label": label,
            "files_field": files_field,
        },
    )


class BaselineStrategy(str, Enum):
    """How to select commits for building the historic quality baseline.

    MERGE_ANCHORED: Follows first-parent (trunk) history, so each delta represents
    the net quality effect of one accepted merge/PR. Best for repos using merge
    workflows where merges are quality checkpoints (code review happened).

    TIME_SAMPLED: Samples commits at regular time intervals within a bounded
    lookback window. Prevents the 'N commits = 2 days' problem in active repos.
    Best for repos with linear history (squash merges, rebase workflows).

    AUTO: Examines recent commit history to compute merge ratio. If merges are
    frequent enough (above configurable threshold), uses MERGE_ANCHORED.
    Otherwise falls back to TIME_SAMPLED.
    """

    MERGE_ANCHORED = "merge_anchored"
    TIME_SAMPLED = "time_sampled"
    AUTO = "auto"


class ResolvedBaselineStrategy(BaseModel):
    """Records which baseline strategy was actually used after AUTO resolution.

    AUTO never appears as the resolved strategy -- it always resolves to one of
    the concrete strategies. This model is stored with the cached baseline so
    we can invalidate the cache when the user changes strategy settings.
    """

    model_config = ConfigDict(frozen=True)

    requested: BaselineStrategy = Field(description="Strategy requested via settings (may be AUTO)")
    resolved: BaselineStrategy = Field(
        description="Concrete strategy actually used (never AUTO). "
        "MERGE_ANCHORED uses first-parent trunk history at merge points. "
        "TIME_SAMPLED samples commits at regular time intervals within a bounded lookback window."
    )
    merge_ratio: float = Field(
        description="Fraction of merge commits in the detection sample (0.0-1.0). "
        "Used by AUTO to decide strategy: above threshold -> MERGE_ANCHORED, below -> TIME_SAMPLED."
    )
    total_commits_sampled: int = Field(description="Number of recent commits examined during strategy auto-detection")

    @field_validator("resolved")
    @classmethod
    def resolved_must_be_concrete(cls, v: BaselineStrategy) -> BaselineStrategy:
        if v == BaselineStrategy.AUTO:
            raise ValueError("resolved strategy cannot be AUTO")
        return v


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


class TokenCountError(BaseModel):
    """Error that occurred during token counting."""

    model_config = ConfigDict(frozen=True)

    message: str
    path: str


class CacheUpdateError(BaseModel):
    """Error that occurred during cache update operation."""

    model_config = ConfigDict(frozen=True)

    message: str
    session_id: str
    operation: str = "update_coverage"


class FileAnalysisResult(BaseModel):
    """Result from analyzing a single Python file for complexity metrics."""

    path: str
    complexity: int
    volume: float
    difficulty: float
    effort: float
    mi: float
    tokens: int | TokenCountError | None = None
    error: str | None = None


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
    files_by_effort: dict[str, float] = Field(
        default_factory=dict, description="Mapping of filename to Halstead effort"
    )
    files_with_parse_errors: dict[str, str] = Field(
        default_factory=dict, description="Files that failed to parse: {filepath: error_message}"
    )

    total_tokens: int = 0
    average_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    files_by_token_count: dict[str, int] = Field(default_factory=dict, description="Mapping of filename to token count")


class ComplexityDelta(BaseModel):
    """Complexity change comparison between two versions."""

    total_complexity_change: int = 0
    files_added: list[str] = Field(default_factory=list)
    files_removed: list[str] = Field(default_factory=list)
    files_changed: dict[str, int] = Field(default_factory=dict, description="Mapping of filename to complexity delta")
    files_effort_changed: dict[str, float] = Field(
        default_factory=dict, description="Mapping of filename to effort delta"
    )
    net_files_change: int = Field(default=0, description="Net change in number of files (files_added - files_removed)")
    avg_complexity_change: float = 0.0

    total_volume_change: float = 0.0
    avg_volume_change: float = 0.0
    avg_effort_change: float = 0.0
    avg_difficulty_change: float = 0.0
    total_difficulty_change: float = 0.0
    total_effort_change: float = 0.0
    total_mi_change: float = 0.0

    avg_mi_change: float = 0.0

    total_tokens_change: int = 0
    avg_tokens_change: float = 0.0

    qpe_change: float = Field(default=0.0, description="QPE delta (current - baseline)")

    type_hint_coverage_change: float = 0.0
    docstring_coverage_change: float = 0.0
    deprecation_change: int = 0

    any_type_percentage_change: float = 0.0
    str_type_percentage_change: float = 0.0

    orphan_comment_change: int = 0
    untracked_todo_change: int = 0
    inline_import_change: int = 0
    dict_get_with_default_change: int = 0
    hasattr_getattr_change: int = 0
    nonempty_init_change: int = 0
    test_skip_change: int = 0
    swallowed_exception_change: int = 0
    type_ignore_change: int = 0
    dynamic_execution_change: int = 0
    single_method_class_change: int = 0
    deep_inheritance_change: int = 0
    passthrough_wrapper_change: int = 0
    sys_path_manipulation_change: int = 0

    def get_smell_changes(self) -> dict[str, int]:
        """Return smell name to change value mapping for direct access."""
        return {
            "orphan_comment": self.orphan_comment_change,
            "untracked_todo": self.untracked_todo_change,
            "inline_import": self.inline_import_change,
            "dict_get_with_default": self.dict_get_with_default_change,
            "hasattr_getattr": self.hasattr_getattr_change,
            "nonempty_init": self.nonempty_init_change,
            "test_skip": self.test_skip_change,
            "swallowed_exception": self.swallowed_exception_change,
            "type_ignore": self.type_ignore_change,
            "dynamic_execution": self.dynamic_execution_change,
            "single_method_class": self.single_method_class_change,
            "deep_inheritance": self.deep_inheritance_change,
            "passthrough_wrapper": self.passthrough_wrapper_change,
            "sys_path_manipulation": self.sys_path_manipulation_change,
        }


class TodoItem(BaseModel):
    """Represents a single todo item from Claude Code's TodoWrite tool."""

    content: str = Field(description="The todo item description")
    status: str = Field(description="Status: pending, in_progress, or completed")
    activeForm: str = Field(description="Present continuous form shown during execution")


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
    search_events: int = Field(default=0, description="Number of exploration-type tool events")
    implementation_events: int = Field(default=0, description="Number of implementation-type tool events")
    exploration_percentage: float = Field(default=0.0, description="Percentage of events that were exploration (0-100)")


class TokenUsage(BaseModel):
    """Token usage metrics categorized by exploration vs implementation."""

    total_input_tokens: int = Field(default=0, description="Total input tokens across all messages")
    total_output_tokens: int = Field(default=0, description="Total output tokens across all messages")
    exploration_input_tokens: int = Field(default=0, description="Input tokens for exploration tools")
    exploration_output_tokens: int = Field(default=0, description="Output tokens for exploration tools")
    implementation_input_tokens: int = Field(default=0, description="Input tokens for implementation tools")
    implementation_output_tokens: int = Field(default=0, description="Output tokens for implementation tools")
    subagent_tokens: int = Field(default=0, description="Total tokens from subagent (Task tool) executions")

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def exploration_tokens(self) -> int:
        """Total exploration tokens (input + output)."""
        return self.exploration_input_tokens + self.exploration_output_tokens

    @property
    def implementation_tokens(self) -> int:
        """Total implementation tokens (input + output)."""
        return self.implementation_input_tokens + self.implementation_output_tokens

    @property
    def exploration_token_percentage(self) -> float:
        """Percentage of tokens used for exploration (0-100)."""
        total = self.exploration_tokens + self.implementation_tokens
        return (self.exploration_tokens / total * 100) if total > 0 else 0.0


class SessionMetadata(BaseModel):
    """Structured metadata for a saved session, agent-tool-agnostic."""

    session_id: str
    agent_tool: AgentTool
    agent_version: str | None = None
    model: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    total_events: int = 0
    working_directory: str
    git_branch: str | None = None
    token_usage: TokenUsage | None = None


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
    exploration_percentage: float = Field(default=0.0, description="Percentage of events that were exploration (0-100)")
    token_usage: TokenUsage | None = Field(
        default=None, description="Token usage breakdown by exploration vs implementation"
    )
    plan_files_created: int = Field(default=0, description="Number of plan files written to ~/.claude/plans/")
    plan_file_paths: list[str] = Field(default_factory=list, description="Paths to plan files created during session")
    final_todos: list[TodoItem] = Field(default_factory=list, description="Final state of todos at session end")


class CompactEvent(BaseModel):
    """Represents a compact event from Claude Code transcript.

    Compact events occur when the conversation is compacted to save context.
    They consist of a compact_boundary system event followed by an isCompactSummary user message.
    """

    line_number: int = Field(description="Line number in transcript where compact occurred")
    trigger: str = Field(description="Trigger type: 'auto' or 'manual'")
    pre_tokens: int = Field(description="Token count before this compact")
    summary_content: str = Field(description="The compact summary content")
    timestamp: datetime = Field(description="When the compact occurred")
    uuid: str = Field(description="UUID of the compact_boundary event")
    version: str = Field(default="n/a", description="Claude Code version at compact time")
    git_branch: str = Field(default="n/a", description="Git branch at compact time")


class SavedCompact(BaseModel):
    """Saved compact event with instructions and results for export."""

    transcript_path: str = Field(description="Path to source transcript")
    line_number: int = Field(description="Line number in transcript")
    timestamp: datetime
    trigger: str
    pre_tokens: int
    summary_content: str
    instructions: str | None = Field(default=None, description="Compact instructions if found")
    version: str = Field(default="n/a", description="Claude Code version at compact time")
    git_branch: str = Field(default="n/a", description="Git branch at compact time")


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
    context_coverage: "ContextCoverage | None" = None
    project: Project | None = None
    transcript_path: str | None = None
    compact_events: list[CompactEvent] = Field(
        default_factory=list, description="Compacts that occurred during session"
    )


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


class ScopedSmell(BaseModel):
    """A smell classified for a specific session context."""

    model_config = ConfigDict(frozen=True)

    label: str
    name: str
    count: int
    change: int
    actionable_files: list[str]
    guidance: str
    is_blocking: bool


class SmellData(BaseModel):
    """Structured smell data with direct field access (no getattr needed)."""

    model_config = ConfigDict(frozen=True)

    name: str
    count: int
    files: list[str]

    @property
    def definition(self) -> SmellDefinition:
        """Get the smell definition from registry."""
        return SMELL_REGISTRY[self.name]

    @property
    def label(self) -> str:
        """Get display label from registry."""
        return self.definition.label

    @property
    def category(self) -> SmellCategory:
        """Get category from registry."""
        return self.definition.category

    @property
    def weight(self) -> float:
        """Get weight from registry."""
        return self.definition.weight


class ExtendedComplexityMetrics(ComplexityMetrics):
    """Extended metrics including Halstead and Maintainability Index.

    Core Halstead metrics are required to catch missing parameter bugs early.
    """

    total_volume: float
    total_effort: float
    total_difficulty: float
    average_volume: float
    average_effort: float
    average_difficulty: float

    total_mi: float
    average_mi: float = Field(description="Higher is better (0-100 scale)")
    files_by_mi: dict[str, float] = Field(default_factory=dict, description="Mapping of filename to MI score")

    type_hint_coverage: float = Field(default=0.0, description="Percentage of functions/args with type hints (0-100)")
    docstring_coverage: float = Field(
        default=0.0, description="Percentage of functions/classes with docstrings (0-100)"
    )
    deprecation_count: int = Field(default=0, description="Number of deprecation warnings/markers found")

    any_type_percentage: float = Field(
        default=0.0, description="Percentage of type annotations using Any (0-100). Lower is better."
    )
    str_type_percentage: float = Field(
        default=0.0,
        description="Percentage of type annotations using str (0-100). Consider enums for constrained strings.",
    )

    test_coverage_percent: float | None = Field(
        default=None, description="Pytest test coverage percentage (0-100). None if unavailable."
    )
    test_coverage_source: str | None = Field(
        default=None, description="Source file for coverage data (e.g., 'coverage.xml')"
    )

    orphan_comment_count: int = SmellField(
        label="Orphan Comments",
        files_field="orphan_comment_files",
        guidance="Make sure inline code comments add meaningful information about non-obvious design tradeoffs or explain tech debt or performance implications. Consider if these could be docstrings or field descriptors instead",
    )
    untracked_todo_count: int = SmellField(
        label="Untracked TODOs",
        files_field="untracked_todo_files",
        guidance="Untracked TODOs should include ticket references (JIRA-123, #123) or URLs",
    )
    inline_import_count: int = SmellField(
        label="Inline Imports",
        files_field="inline_import_files",
        guidance="Verify if these can be moved to the top of the file (except TYPE_CHECKING guards)",
    )
    dict_get_with_default_count: int = SmellField(
        label="Modeling Gaps (.get() defaults)",
        files_field="dict_get_with_default_files",
        guidance="Consider if these are justified or just indicate modeling gaps - replace with Pydantic BaseSettings or BaseModel with narrower field types or raise explicit errors instead",
    )
    hasattr_getattr_count: int = SmellField(
        label="Modeling Gaps (hasattr/getattr)",
        files_field="hasattr_getattr_files",
        guidance="Consider if these indicate missing domain models - replace with Pydantic BaseModel objects with explicit fields or raise explicit errors instead",
    )
    nonempty_init_count: int = SmellField(
        label="Logic in __init__.py",
        files_field="nonempty_init_files",
        guidance="Consider if implementation code should be moved out of __init__.py files",
    )
    test_skip_count: int = SmellField(
        label="Test Skips",
        files_field="test_skip_files",
        guidance="BLOCKING: You MUST present a table with columns [Test Name | Intent] for each skip and ask user to confirm skipping is acceptable",
    )
    swallowed_exception_count: int = SmellField(
        label="Swallowed Exceptions",
        files_field="swallowed_exception_files",
        guidance="BLOCKING: You MUST present a table with columns [Location | Purpose | Justification ] for each and ask user to confirm silent failure is acceptable",
    )
    type_ignore_count: int = SmellField(
        label="Type Ignores",
        files_field="type_ignore_files",
        guidance="Review type: ignore comments - consider fixing the underlying type issue",
    )
    dynamic_execution_count: int = SmellField(
        label="Dynamic Execution",
        files_field="dynamic_execution_files",
        guidance="Review usage of eval/exec/compile - ensure this is necessary and secure",
    )
    single_method_class_count: int = SmellField(
        label="Single-Method Classes",
        files_field="single_method_class_files",
        guidance="Consider using a function instead of a class with only one method besides __init__",
    )
    deep_inheritance_count: int = SmellField(
        label="Deep Inheritance",
        files_field="deep_inheritance_files",
        guidance="Prefer composition over inheritance; >2 base classes increases complexity",
    )
    passthrough_wrapper_count: int = SmellField(
        label="Pass-Through Wrappers",
        files_field="passthrough_wrapper_files",
        guidance="Function that just delegates to another with same args; consider removing indirection",
    )
    sys_path_manipulation_count: int = SmellField(
        label="sys.path Manipulation",
        files_field="sys_path_manipulation_files",
        guidance="sys.path mutations bypass the package system — restructure package boundaries and use absolute imports from installed packages instead",
    )

    # LOC metrics (for file filtering in QPE)
    total_loc: int = Field(default=0, description="Total lines of code across all files")
    code_loc: int = Field(default=0, description="Non-blank, non-comment lines")
    files_by_loc: dict[str, int] = Field(
        default_factory=dict, description="Mapping of filepath to code LOC for file filtering"
    )

    orphan_comment_files: list[str] = Field(default_factory=list, description="Files with orphan comments")
    untracked_todo_files: list[str] = Field(default_factory=list, description="Files with untracked TODOs")
    inline_import_files: list[str] = Field(default_factory=list, description="Files with inline imports")
    dict_get_with_default_files: list[str] = Field(default_factory=list, description="Files with .get() defaults")
    hasattr_getattr_files: list[str] = Field(default_factory=list, description="Files with hasattr/getattr")
    nonempty_init_files: list[str] = Field(default_factory=list, description="Files with nonempty __init__")
    test_skip_files: list[str] = Field(default_factory=list, description="Files with test skips")
    swallowed_exception_files: list[str] = Field(default_factory=list, description="Files with swallowed exceptions")
    type_ignore_files: list[str] = Field(default_factory=list, description="Files with type: ignore")
    dynamic_execution_files: list[str] = Field(default_factory=list, description="Files with eval/exec/compile")
    single_method_class_files: list[str] = Field(default_factory=list, description="Files with single-method classes")
    deep_inheritance_files: list[str] = Field(
        default_factory=list, description="Files with deep inheritance (>2 bases)"
    )
    passthrough_wrapper_files: list[str] = Field(default_factory=list, description="Files with pass-through wrappers")
    sys_path_manipulation_files: list[str] = Field(default_factory=list, description="Files with sys.path mutations")

    def get_smells(self) -> list["SmellData"]:
        """Return all smell data as structured objects with direct field access."""
        return [
            SmellData(
                name="orphan_comment",
                count=self.orphan_comment_count,
                files=self.orphan_comment_files,
            ),
            SmellData(
                name="untracked_todo",
                count=self.untracked_todo_count,
                files=self.untracked_todo_files,
            ),
            SmellData(
                name="swallowed_exception",
                count=self.swallowed_exception_count,
                files=self.swallowed_exception_files,
            ),
            SmellData(
                name="test_skip",
                count=self.test_skip_count,
                files=self.test_skip_files,
            ),
            SmellData(
                name="type_ignore",
                count=self.type_ignore_count,
                files=self.type_ignore_files,
            ),
            SmellData(
                name="dynamic_execution",
                count=self.dynamic_execution_count,
                files=self.dynamic_execution_files,
            ),
            SmellData(
                name="inline_import",
                count=self.inline_import_count,
                files=self.inline_import_files,
            ),
            SmellData(
                name="dict_get_with_default",
                count=self.dict_get_with_default_count,
                files=self.dict_get_with_default_files,
            ),
            SmellData(
                name="hasattr_getattr",
                count=self.hasattr_getattr_count,
                files=self.hasattr_getattr_files,
            ),
            SmellData(
                name="nonempty_init",
                count=self.nonempty_init_count,
                files=self.nonempty_init_files,
            ),
            SmellData(
                name="single_method_class",
                count=self.single_method_class_count,
                files=self.single_method_class_files,
            ),
            SmellData(
                name="deep_inheritance",
                count=self.deep_inheritance_count,
                files=self.deep_inheritance_files,
            ),
            SmellData(
                name="passthrough_wrapper",
                count=self.passthrough_wrapper_count,
                files=self.passthrough_wrapper_files,
            ),
            SmellData(
                name="sys_path_manipulation",
                count=self.sys_path_manipulation_count,
                files=self.sys_path_manipulation_files,
            ),
        ]

    def get_smell_files(self) -> dict[str, list[str]]:
        """Return smell name to files mapping for filtering."""
        return {smell.name: smell.files for smell in self.get_smells()}

    def get_smell_counts(self) -> "SmellCounts":
        """Return smell counts as a typed model for QPE and display."""
        return SmellCounts(**{smell.name: smell.count for smell in self.get_smells()})


class ExperimentRun(BaseModel):
    """Represents a single experiment run."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    repository_path: Path
    start_commit: str = Field(description="SHA of starting commit (e.g., HEAD~1)")
    target_commit: str = Field(description="SHA of target commit (e.g., HEAD)")
    process_id: int
    worktree_path: Path | None = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    nfp_objective: NextFeaturePrediction | None = Field(
        default=None, description="Feature objectives for this experiment"
    )


class ExperimentProgress(BaseModel):
    """Tracks real-time progress with CLI and QPE metrics."""

    experiment_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    current_metrics: ExtendedComplexityMetrics
    target_metrics: ExtendedComplexityMetrics = Field(description="Metrics from HEAD commit")

    # Legacy CLI metrics (deprecated - use qpe_score instead)
    cli_score: float = Field(
        default=0.0, description="DEPRECATED: Use qpe_score. 1.0 = perfect match, <0 = overshooting"
    )
    complexity_score: float = 0.0
    halstead_score: float = 0.0
    maintainability_score: float = 0.0

    # QPE metrics (principled replacement for CLI)
    qpe_score: float | None = Field(default=None, description="Quality-per-effort score (higher is better)")
    smell_penalty: float | None = Field(default=None, description="Penalty from code smells (0-0.5 range)")


class CommitComplexitySnapshot(BaseModel):
    """Complexity metrics for a specific commit."""

    commit_sha: str
    commit_message: str
    timestamp: datetime
    complexity_metrics: ExtendedComplexityMetrics
    parent_commit_sha: str | None = None
    complexity_delta: ComplexityDelta | None = Field(default=None, description="Delta from parent commit")


class CommitChain(BaseModel):
    """Represents a chain of commits with complexity evolution."""

    repository_path: Path
    base_commit: str = Field(description="Starting point (e.g., HEAD~10)")
    head_commit: str = Field(description="End point (e.g., HEAD)")
    commits: list[CommitComplexitySnapshot] = Field(default_factory=list)
    total_complexity_growth: int = 0
    average_complexity_per_commit: float = 0.0


class ComplexityEvolution(BaseModel):
    """Tracks how complexity evolves across commits."""

    commit_sha: str
    cumulative_complexity: int = Field(description="Total complexity up to this commit")
    incremental_complexity: int = Field(description="Complexity added in this commit")
    files_modified: int
    functions_added: int
    functions_removed: int
    functions_modified: int


class MergeCommit(BaseModel):
    """Information about a merge commit in git history."""

    hash: str = Field(description="The commit hash")
    parents: list[str] = Field(description="Parent commit hashes")
    message: str = Field(description="Commit message")
    feature_branch: str = Field(description="The feature branch commit (second parent)")


class FeatureBoundary(BaseModel):
    """Represents a feature's boundary commits."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the feature")
    base_commit: str = Field(description="Common ancestor of the merge")
    head_commit: str = Field(description="Feature branch tip commit")
    merge_commit: str = Field(description="The merge commit hash")
    merge_message: str = Field(description="Message from the merge commit")
    feature_message: str = Field(description="Message from the feature branch tip")
    repository_path: Path = Field(description="Path to the repository this feature belongs to")

    @property
    def short_id(self) -> str:
        """Get the first 8 characters of the feature ID for display."""
        return self.id[:8]


class UserStoryEntry(BaseModel):
    """User story entry for diff <> user story pairs with ratings."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)

    base_commit: str = Field(description="Base commit reference")
    head_commit: str = Field(description="Head commit reference")
    diff_content: str = Field(description="The git diff content")
    stride_size: int = Field(default=1, description="Number of intermediate commits spanned")

    user_stories: str = Field(description="Generated user stories markdown")

    rating: int = Field(ge=1, le=5, description="User rating from 1-5")
    guidelines_for_improving: str = Field(default="", description="Guidelines for improving the user story generation")

    model_used: str = Field(default="o3", description="Model used for generation")
    prompt_template: str = Field(default="", description="Template used for prompt")
    repository_path: str = Field(default="", description="Repository path")

    @property
    def short_id(self) -> str:
        """Get the first 8 characters of the user story entry ID for display."""
        return self.id[:8]


class UserStoryStatistics(BaseModel):
    """Statistics about user story entries."""

    total_entries: int = Field(description="Total number of user story entries")
    avg_rating: float = Field(description="Average rating across all entries")
    unique_models: int = Field(description="Number of unique models used")
    unique_repos: int = Field(description="Number of unique repositories")
    rating_distribution: dict[str, int] = Field(description="Distribution of ratings")


class UserStoryDisplayData(BaseModel):
    """Display data for user story entries in tables."""

    entry_id: str = Field(description="Short ID of the entry")
    date: str = Field(description="Formatted creation date")
    commits: str = Field(description="Short commit range display")
    rating: str = Field(description="Formatted rating display")
    model: str = Field(description="Model used for generation")
    repository: str = Field(description="Repository name")


class ExperimentDisplayData(BaseModel):
    """Display data for experiment runs in tables."""

    id: str = Field(description="Experiment ID")
    repository_name: str = Field(description="Name of the repository")
    commits_display: str = Field(description="Formatted commit range (e.g., 'abc123 → def456')")
    start_time: str = Field(description="Formatted start time")
    duration: str = Field(description="Formatted duration or 'Running...'")
    status: str = Field(description="Current status (running, completed, failed)")


class ProgressDisplayData(BaseModel):
    """Display data for experiment progress rows."""

    timestamp: str = Field(description="Formatted timestamp (HH:MM:SS)")
    cli_score: str = Field(description="Formatted CLI score")
    complexity_score: str = Field(description="Formatted complexity score")
    halstead_score: str = Field(description="Formatted Halstead score")
    maintainability_score: str = Field(description="Formatted maintainability score")


class NFPObjectiveDisplayData(BaseModel):
    """Display data for NFP objectives in tables."""

    id: str = Field(description="Objective ID")
    title: str = Field(description="Objective title")
    commits: str = Field(description="Formatted commit range")
    story_count: int = Field(description="Number of associated user stories")
    complexity: int = Field(description="Complexity metric")
    created_date: str = Field(description="Formatted creation date")


class CodeQualityCache(BaseModel):
    """Cached code quality metrics for a specific session/repository/commit combination."""

    id: int | None = None
    session_id: str = Field(description="Session ID this cache entry belongs to")
    repository_path: str = Field(description="Absolute path to the repository")
    commit_sha: str = Field(description="Git commit SHA when metrics were calculated")
    calculated_at: datetime = Field(default_factory=datetime.now, description="When metrics were calculated")
    complexity_metrics: ExtendedComplexityMetrics = Field(description="Cached complexity metrics")
    complexity_delta: ComplexityDelta | None = Field(
        default=None, description="Cached complexity delta from previous commit"
    )
    working_tree_hash: str | None = Field(
        default=None, description="Hash of working tree state for uncommitted changes. NULL for clean repos."
    )


class ImpactCategory(str, Enum):
    """Categories for staged changes impact assessment."""

    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MINOR_IMPROVEMENT = "minor_improvement"
    NEUTRAL = "neutral"
    MINOR_DEGRADATION = "minor_degradation"
    SIGNIFICANT_DEGRADATION = "significant_degradation"


class HistoricalMetricStats(BaseModel):
    """Statistical summary of a metric across repository history."""

    metric_name: str = Field(description="Name of the metric (e.g., 'cc_delta', 'effort_delta')")
    mean: float = Field(description="Mean value across all commits")
    std_dev: float = Field(description="Standard deviation")
    median: float = Field(description="Median value")
    min_value: float = Field(description="Minimum observed value")
    max_value: float = Field(description="Maximum observed value")
    sample_count: int = Field(description="Number of commits analyzed")
    trend_coefficient: float = Field(
        default=0.0, description="Linear regression slope indicating improvement/degradation trend"
    )


GALEN_TOKENS_PER_MONTH = 1_000_000
GALEN_TOKENS_PER_DAY = GALEN_TOKENS_PER_MONTH / 30  # ~33,333 tokens/day


class GalenMetrics(BaseModel):
    """Developer productivity metrics based on code token throughput.

    Named after Galen Hunt at Microsoft, who calculated that rewriting C++ to Rust
    requires approximately 1 million tokens per developer per month.

    1 Galen = 1 million code tokens per developer per month.
    Based on Microsoft's calculation for C++ to Rust migration effort.
    """

    tokens_changed: int = Field(description="Net tokens changed during the period")
    period_days: float = Field(description="Duration of the analysis period in days")
    tokens_per_day: float = Field(description="Average tokens changed per day")
    galen_rate: float = Field(description="Productivity rate (1.0 = on track for 1M tokens/month)")
    tokens_per_day_to_reach_one_galen: float | None = Field(
        default=None,
        description="Additional tokens/day needed to reach 1 Galen (None if already >= 1 Galen)",
    )

    @classmethod
    def calculate(cls, tokens_changed: int, period_days: float) -> "GalenMetrics":
        """Calculate Galen metrics from token change and time period."""
        if period_days <= 0:
            return cls(
                tokens_changed=tokens_changed,
                period_days=0.0,
                tokens_per_day=0.0,
                galen_rate=0.0,
                tokens_per_day_to_reach_one_galen=GALEN_TOKENS_PER_DAY,
            )

        tokens_per_day = abs(tokens_changed) / period_days
        galen_rate = tokens_per_day / GALEN_TOKENS_PER_DAY

        tokens_needed = None
        if galen_rate < 1.0:
            tokens_needed = GALEN_TOKENS_PER_DAY - tokens_per_day

        return cls(
            tokens_changed=tokens_changed,
            period_days=period_days,
            tokens_per_day=tokens_per_day,
            galen_rate=galen_rate,
            tokens_per_day_to_reach_one_galen=tokens_needed,
        )


class RepoBaseline(BaseModel):
    """Baseline statistics computed from entire repository history.

    Delta statistics use TOTAL metrics (sums across all files), not averages.
    This allows correct comparison even when baseline is computed from only
    changed files per commit (an optimization that works because sums are additive).
    """

    repository_path: str = Field(description="Absolute path to the repository")
    computed_at: datetime = Field(default_factory=datetime.now)
    head_commit_sha: str = Field(description="HEAD commit when baseline was computed")
    total_commits_analyzed: int = Field(description="Number of commits in baseline calculation")

    cc_delta_stats: HistoricalMetricStats = Field(description="Total CC delta statistics per commit")
    effort_delta_stats: HistoricalMetricStats = Field(description="Total Effort delta statistics per commit")
    mi_delta_stats: HistoricalMetricStats = Field(description="Total MI delta statistics per commit")

    current_metrics: ExtendedComplexityMetrics = Field(description="Metrics at HEAD commit")

    oldest_commit_date: datetime | None = Field(
        default=None, description="Timestamp of the oldest commit in the analysis"
    )
    newest_commit_date: datetime | None = Field(
        default=None, description="Timestamp of the newest commit in the analysis"
    )
    oldest_commit_tokens: int | None = Field(
        default=None, description="Total tokens in codebase at oldest analyzed commit"
    )

    qpe_stats: HistoricalMetricStats | None = Field(default=None, description="QPE statistics from commit history")
    current_qpe: "QPEScore | None" = Field(default=None, description="QPE score at HEAD")

    strategy: ResolvedBaselineStrategy | None = Field(
        default=None,
        description="Which baseline computation strategy produced this baseline. "
        "None for legacy baselines computed before strategy support was added. "
        "Used for cache invalidation: strategy mismatch with current settings triggers recomputation.",
    )

    qpe_weight_version: str | None = Field(
        default=None,
        description="QPE_WEIGHT_VERSION at time of computation. None = pre-versioning entry.",
    )


class ZScoreInterpretation(str, Enum):
    """Human-readable interpretation of Z-score values."""

    MUCH_BETTER = "much better than avg"
    BETTER = "better than avg"
    ABOUT_AVERAGE = "about avg"
    WORSE = "worse than avg"
    MUCH_WORSE = "much worse than avg"

    @classmethod
    def from_z_score(cls, normalized_z: float, verbose: bool = False) -> "ZScoreInterpretation":
        """Interpret a normalized Z-score (positive = good).

        Args:
            normalized_z: Z-score where positive values indicate improvement
            verbose: If True, uses wider thresholds (1.5/0.5) for more nuanced output

        Returns:
            ZScoreInterpretation enum value
        """
        if verbose:
            if normalized_z > 1.5:
                return cls.MUCH_BETTER
            elif normalized_z > 0.5:
                return cls.BETTER
            elif normalized_z > -0.5:
                return cls.ABOUT_AVERAGE
            elif normalized_z > -1.5:
                return cls.WORSE
            else:
                return cls.MUCH_WORSE
        else:
            if normalized_z > 1.0:
                return cls.MUCH_BETTER
            elif normalized_z > 0.3:
                return cls.BETTER
            elif normalized_z > -0.3:
                return cls.ABOUT_AVERAGE
            elif normalized_z > -1.0:
                return cls.WORSE
            else:
                return cls.MUCH_WORSE


class ImpactAssessment(BaseModel):
    """Assessment of staged changes impact against repo baseline."""

    cc_z_score: float = Field(description="Z-score for CC change (positive = above avg increase)")
    effort_z_score: float = Field(description="Z-score for Effort change (positive = above avg increase)")
    mi_z_score: float = Field(description="Z-score for MI change (positive = above avg increase, which is good)")

    impact_score: float = Field(
        description="Composite score: positive = above-average quality improvement, negative = below-average"
    )
    impact_category: ImpactCategory = Field(description="Categorical assessment of impact")

    cc_delta: float = Field(description="Total CC change (sum across all files)")
    effort_delta: float = Field(description="Total Effort change (sum across all files)")
    mi_delta: float = Field(description="Total MI change (sum across all files)")

    qpe_delta: float = Field(default=0.0, description="QPE change between baseline and current")
    qpe_z_score: float = Field(default=0.0, description="Z-score for QPE change (positive = above avg improvement)")

    def interpret_cc(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret CC z-score (lower CC is better, so invert)."""
        return ZScoreInterpretation.from_z_score(-self.cc_z_score, verbose)

    def interpret_effort(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret Effort z-score (lower effort is better, so invert)."""
        return ZScoreInterpretation.from_z_score(-self.effort_z_score, verbose)

    def interpret_mi(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret MI z-score (higher MI is better, no inversion)."""
        return ZScoreInterpretation.from_z_score(self.mi_z_score, verbose)

    def interpret_qpe(self, verbose: bool = False) -> ZScoreInterpretation:
        """Interpret QPE z-score (higher QPE is better, no inversion)."""
        return ZScoreInterpretation.from_z_score(self.qpe_z_score, verbose)


class SmellCounts(BaseModel):
    """Per-smell occurrence counts from complexity analysis.

    Every field corresponds to a smell in SMELL_REGISTRY. Using explicit fields
    instead of dict[str, int] so access is validated at construction time and
    there is no need for .get() defaults or key existence checks.
    """

    model_config = ConfigDict(frozen=True)

    orphan_comment: int = 0
    untracked_todo: int = 0
    swallowed_exception: int = 0
    test_skip: int = 0
    type_ignore: int = 0
    dynamic_execution: int = 0
    inline_import: int = 0
    dict_get_with_default: int = 0
    hasattr_getattr: int = 0
    nonempty_init: int = 0
    single_method_class: int = 0
    deep_inheritance: int = 0
    passthrough_wrapper: int = 0
    sys_path_manipulation: int = 0


class QPEScore(BaseModel):
    """Quality score for principled code quality comparison.

    Single metric: adjusted quality = MI * (1 - smell_penalty) + bonuses.
    Used for temporal tracking (delta between commits), cross-project comparison,
    and GRPO rollout advantage computation.
    """

    qpe: float = Field(description="Adjusted quality score (higher is better)")
    mi_normalized: float = Field(description="Maintainability Index normalized to 0-1")
    smell_penalty: float = Field(description="Penalty from code smells (sigmoid-saturated, 0-0.9 range)")
    adjusted_quality: float = Field(description="MI after smell penalty applied")

    smell_counts: SmellCounts = Field(
        default_factory=SmellCounts, description="Individual smell counts contributing to penalty"
    )


class SmellAdvantage(BaseModel):
    """Per-smell contribution to the GRPO advantage signal.

    The aggregate grpo_advantage() collapses all quality into a single scalar.
    SmellAdvantage decomposes that into individual smell contributions, enabling:
    - Interpretability: which specific smells drove the advantage/disadvantage
    - Per-smell reward shaping: downstream GRPO training can weight individual
      smell improvements differently
    - Debugging: understand why two implementations scored differently

    The weighted_delta uses the same weights from SMELL_REGISTRY that feed into
    QPE's smell_penalty calculation, ensuring consistency between the aggregate
    and decomposed signals.
    """

    model_config = ConfigDict(frozen=True)

    smell_name: str = Field(
        description="Internal name from SMELL_REGISTRY (e.g., 'swallowed_exception', 'hasattr_getattr')"
    )
    baseline_count: int = Field(description="Number of this smell in the baseline/reference implementation")
    candidate_count: int = Field(description="Number of this smell in the candidate implementation being evaluated")
    weight: float = Field(
        description="Smell weight from SMELL_REGISTRY (0.02-0.15). "
        "Higher weight means this smell has more impact on QPE penalty."
    )
    weighted_delta: float = Field(
        description="(candidate_count - baseline_count) * weight. "
        "Negative = candidate improved (fewer smells). "
        "Positive = candidate regressed (more smells). "
        "Zero = no change for this smell type."
    )


class ImplementationComparison(BaseModel):
    """Result of comparing two parallel implementations via their subtree prefixes.

    This is the primary output for GRPO-based quality comparison. Two code subtrees
    (e.g., two implementations of the same feature living side-by-side in the repo)
    are analyzed independently and compared using QPE + smell decomposition.

    The aggregate_advantage is the main GRPO reward signal: positive means B is
    better, negative means A is better, bounded to (-1, 1) via tanh.
    """

    prefix_a: str = Field(
        description="Subtree path prefix for implementation A (e.g., 'vendor/lib-a'). "
        "All Python files under this prefix are analyzed as a unit."
    )
    prefix_b: str = Field(
        description="Subtree path prefix for implementation B (e.g., 'vendor/lib-b'). "
        "Compared against prefix_a to determine which implementation is better."
    )
    ref: str = Field(
        description="Git ref both subtrees were extracted from (e.g., 'HEAD', 'main', commit SHA). "
        "Both prefixes are analyzed at this same point in history."
    )
    qpe_a: "QPEScore" = Field(
        description="Full QPE score for implementation A including MI, smell penalty, and per-smell counts"
    )
    qpe_b: "QPEScore" = Field(
        description="Full QPE score for implementation B including MI, smell penalty, and per-smell counts"
    )
    aggregate_advantage: float = Field(
        description="GRPO advantage of B over A, bounded (-1, 1) via tanh. "
        "Positive = B is better quality. Negative = A is better."
    )
    smell_advantages: list[SmellAdvantage] = Field(
        default_factory=list, description="Per-smell advantage breakdown sorted by impact magnitude."
    )
    winner: str = Field(
        description="Which prefix produced better code: prefix_a value, prefix_b value, or 'tie'. "
        "Tie declared when |aggregate_advantage| < 0.01 (within deadband)."
    )


class ProjectQPEResult(BaseModel):
    """QPE result for a single project, used in cross-project comparison."""

    project_path: str = Field(description="Path to the project")
    project_name: str = Field(description="Name of the project")
    qpe_score: QPEScore = Field(description="QPE score for this project")
    metrics: ExtendedComplexityMetrics = Field(description="Full metrics for this project")


class CrossProjectComparison(BaseModel):
    """Result of comparing multiple projects using QPE.

    Projects are ranked by QPE from highest to lowest.
    """

    compared_at: datetime = Field(default_factory=datetime.now)
    total_projects: int = Field(description="Total number of projects compared")

    rankings: list[ProjectQPEResult] = Field(default_factory=list, description="Projects ranked by QPE, highest first")


class LeaderboardEntry(BaseModel):
    """A persistent record of a project's quality score at a specific commit.

    Used for cross-project quality comparison and temporal tracking.
    """

    id: int | None = Field(default=None, description="Database ID")
    project_name: str = Field(description="Name of the project")
    project_path: str = Field(description="Absolute path to the project")
    commit_sha_short: str = Field(description="7-character short git hash")
    commit_sha_full: str = Field(description="Full git hash for deduplication")
    measured_at: datetime = Field(default_factory=datetime.now, description="Date of the analyzed commit")
    qpe_score: float = Field(description="Quality score for cross-project comparison")
    mi_normalized: float = Field(description="Maintainability Index normalized to 0-1")
    smell_penalty: float = Field(description="Penalty from code smells")
    adjusted_quality: float = Field(description="MI × (1 - smell_penalty) + bonuses")
    effort_factor: float = Field(description="log(total_halstead_effort + 1)")
    total_effort: float = Field(description="Total Halstead Effort")
    metrics_json: str = Field(description="Full ExtendedComplexityMetrics as JSON")
    qpe_weight_version: str | None = Field(
        default=None,
        description="QPE_WEIGHT_VERSION at time of computation. None = pre-versioning entry.",
    )


class StagedChangesAnalysis(BaseModel):
    """Complete analysis of staged changes against repository baseline.

    Deprecated: Use CurrentChangesAnalysis instead for analyzing uncommitted changes.
    """

    repository_path: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    staged_files: list[str] = Field(description="List of staged Python files")
    staged_metrics: ExtendedComplexityMetrics = Field(description="Metrics if staged changes were applied")
    baseline_metrics: ExtendedComplexityMetrics = Field(description="Metrics at current HEAD")

    assessment: ImpactAssessment
    baseline: RepoBaseline


class CurrentChangesAnalysis(BaseModel):
    """Complete analysis of changes against repository baseline."""

    repository_path: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    source: AnalysisSource = Field(
        default=AnalysisSource.UNCOMMITTED_CHANGES,
        description="Whether analyzing uncommitted changes or previous commit",
    )
    analyzed_commit_sha: str | None = Field(
        default=None,
        description="SHA of the commit being analyzed (when source is PREVIOUS_COMMIT)",
    )
    base_commit_sha: str | None = Field(
        default=None,
        description="SHA of the base commit for comparison (when source is PREVIOUS_COMMIT)",
    )

    changed_files: list[str] = Field(description="List of changed Python files")
    current_metrics: ExtendedComplexityMetrics = Field(description="Metrics with uncommitted changes applied")
    baseline_metrics: ExtendedComplexityMetrics = Field(description="Metrics at current HEAD")

    assessment: ImpactAssessment
    baseline: RepoBaseline

    blind_spots: list[str] = Field(
        default_factory=list, description="Files dependent on changed files but not in changed set"
    )
    filtered_coverage: dict[str, float] | None = Field(default=None, description="Coverage % for changed files")

    blind_spot_tokens: int = Field(default=0, description="Total tokens in blind spot files")
    changed_files_tokens: int = Field(default=0, description="Total tokens in changed files")
    complete_picture_context_size: int = Field(default=0, description="Sum of tokens in changed files + blind spots")

    galen_metrics: GalenMetrics | None = Field(
        default=None, description="Developer productivity metrics based on token throughput"
    )

    smell_advantages: list["SmellAdvantage"] = Field(
        default_factory=list,
        description="Per-smell advantage breakdown between baseline and current QPE. "
        "Shows which specific smells changed and their weighted impact.",
    )


class CurrentImpactSummary(BaseModel):
    """Compact JSON output of current-impact analysis for CI consumption.

    Extracts the essential fields from CurrentChangesAnalysis, omitting
    the large nested baseline and full metrics objects.
    """

    source: AnalysisSource = Field(description="Whether analyzing uncommitted changes or previous commit")
    analyzed_commit_sha: str | None = Field(
        default=None, description="SHA of the analyzed commit (when source is previous_commit)"
    )
    base_commit_sha: str | None = Field(
        default=None, description="SHA of the base commit (when source is previous_commit)"
    )
    impact_score: float = Field(description="Weighted composite impact score")
    impact_category: ImpactCategory = Field(description="Human-readable impact category")
    qpe_delta: float = Field(description="Change in QPE score")
    cc_delta: float = Field(description="Change in cyclomatic complexity")
    effort_delta: float = Field(description="Change in Halstead effort")
    mi_delta: float = Field(description="Change in maintainability index")
    changed_files_count: int = Field(description="Number of changed code files")
    blind_spots_count: int = Field(description="Number of dependent files not in changed set")
    smell_advantages: list["SmellAdvantage"] = Field(
        default_factory=list, description="Per-smell advantage breakdown"
    )

    @staticmethod
    def from_analysis(analysis: "CurrentChangesAnalysis") -> "CurrentImpactSummary":
        """Create compact summary from full analysis."""
        return CurrentImpactSummary(
            source=analysis.source,
            analyzed_commit_sha=analysis.analyzed_commit_sha,
            base_commit_sha=analysis.base_commit_sha,
            impact_score=analysis.assessment.impact_score,
            impact_category=analysis.assessment.impact_category,
            qpe_delta=analysis.assessment.qpe_delta,
            cc_delta=analysis.assessment.cc_delta,
            effort_delta=analysis.assessment.effort_delta,
            mi_delta=analysis.assessment.mi_delta,
            changed_files_count=len(analysis.changed_files),
            blind_spots_count=len(analysis.blind_spots),
            smell_advantages=analysis.smell_advantages,
        )


class FileCoverageStatus(BaseModel):
    """Coverage status for a single edited file showing what context was read."""

    file_path: str = Field(description="Relative path to the edited file")
    was_read_before_edit: bool = Field(description="Whether the file was read before being edited")
    imports: list[str] = Field(default_factory=list, description="Files this file imports")
    imports_read: list[str] = Field(default_factory=list, description="Imported files that were read")
    dependents: list[str] = Field(default_factory=list, description="Files that import this file")
    dependents_read: list[str] = Field(default_factory=list, description="Dependent files that were read")
    test_files: list[str] = Field(default_factory=list, description="Related test files")
    test_files_read: list[str] = Field(default_factory=list, description="Related test files that were read")

    @property
    def imports_coverage(self) -> float:
        """Percentage of imports that were read (0-100)."""
        if not self.imports:
            return 100.0
        return len(self.imports_read) / len(self.imports) * 100

    @property
    def dependents_coverage(self) -> float:
        """Percentage of dependents that were read (0-100)."""
        if not self.dependents:
            return 100.0
        return len(self.dependents_read) / len(self.dependents) * 100

    @property
    def test_coverage(self) -> float:
        """Percentage of test files that were read (0-100)."""
        if not self.test_files:
            return 100.0
        return len(self.test_files_read) / len(self.test_files) * 100


class ContextCoverage(BaseModel):
    """Tracks whether Claude read enough context before editing files."""

    files_edited: list[str] = Field(default_factory=list, description="All files that were edited")
    files_read: list[str] = Field(default_factory=list, description="All files that were read")
    file_coverage: list[FileCoverageStatus] = Field(
        default_factory=list, description="Coverage status for each edited file"
    )
    blind_spots: list[str] = Field(default_factory=list, description="Files related to edits that were never read")

    @property
    def files_read_before_edit_ratio(self) -> float:
        """Ratio of edited files that were read first (0-1)."""
        if not self.file_coverage:
            return 1.0
        read_first = sum(1 for f in self.file_coverage if f.was_read_before_edit)
        return read_first / len(self.file_coverage)

    @property
    def overall_imports_coverage(self) -> float:
        """Average imports coverage across all edited files (0-100)."""
        if not self.file_coverage:
            return 100.0
        return sum(f.imports_coverage for f in self.file_coverage) / len(self.file_coverage)

    @property
    def overall_dependents_coverage(self) -> float:
        """Average dependents coverage across all edited files (0-100)."""
        if not self.file_coverage:
            return 100.0
        return sum(f.dependents_coverage for f in self.file_coverage) / len(self.file_coverage)

    @property
    def total_blind_spots(self) -> int:
        """Total number of related files that were never read."""
        return len(self.blind_spots)

    @property
    def has_gaps(self) -> bool:
        """Whether there are any coverage gaps requiring attention."""
        return (
            self.files_read_before_edit_ratio < 1.0
            or self.overall_imports_coverage < 100
            or self.overall_dependents_coverage < 100
            or bool(self.blind_spots)
        )


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

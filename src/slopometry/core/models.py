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
    files_with_parse_errors: dict[str, str] = Field(
        default_factory=dict, description="Files that failed to parse: {filepath: error_message}"
    )

    # Token metrics
    total_tokens: int = 0
    average_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    files_by_token_count: dict[str, int] = Field(
        default_factory=dict, description="Mapping of filename to token count"
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
    avg_effort_change: float = 0.0
    avg_difficulty_change: float = 0.0
    total_difficulty_change: float = 0.0
    total_effort_change: float = 0.0
    total_mi_change: float = 0.0

    avg_mi_change: float = 0.0

    total_tokens_change: int = 0
    avg_tokens_change: float = 0.0

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

    orphan_comment_count: int = Field(
        default=0, description="Comments outside docstrings that aren't TODOs or explanatory URLs"
    )
    untracked_todo_count: int = Field(
        default=0, description="TODO comments without ticket references (JIRA-123, #123) or URLs"
    )
    inline_import_count: int = Field(
        default=0, description="Import statements not at module level (excluding TYPE_CHECKING guards)"
    )
    dict_get_with_default_count: int = Field(
        default=0, description=".get() calls with defaults (indicates missing typed config)"
    )
    hasattr_getattr_count: int = Field(
        default=0, description="hasattr()/getattr() calls (indicates missing domain models)"
    )
    nonempty_init_count: int = Field(
        default=0, description="__init__.py files with implementation code (beyond imports/__all__)"
    )

    orphan_comment_files: list[str] = Field(default_factory=list, description="Files with orphan comments")
    untracked_todo_files: list[str] = Field(default_factory=list, description="Files with untracked TODOs")
    inline_import_files: list[str] = Field(default_factory=list, description="Files with inline imports")
    dict_get_with_default_files: list[str] = Field(default_factory=list, description="Files with .get() defaults")
    hasattr_getattr_files: list[str] = Field(default_factory=list, description="Files with hasattr/getattr")
    nonempty_init_files: list[str] = Field(default_factory=list, description="Files with nonempty __init__")


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


class RepoBaseline(BaseModel):
    """Baseline statistics computed from entire repository history."""

    repository_path: str = Field(description="Absolute path to the repository")
    computed_at: datetime = Field(default_factory=datetime.now)
    head_commit_sha: str = Field(description="HEAD commit when baseline was computed")
    total_commits_analyzed: int = Field(description="Number of commits in baseline calculation")

    cc_delta_stats: HistoricalMetricStats = Field(description="Cyclomatic complexity delta statistics")
    effort_delta_stats: HistoricalMetricStats = Field(description="Halstead effort delta statistics")
    mi_delta_stats: HistoricalMetricStats = Field(description="Maintainability index delta statistics")

    current_metrics: ExtendedComplexityMetrics = Field(description="Metrics at HEAD commit")


class ImpactAssessment(BaseModel):
    """Assessment of staged changes impact against repo baseline."""

    cc_z_score: float = Field(description="Z-score for CC change (positive = above avg increase)")
    effort_z_score: float = Field(description="Z-score for Effort change (positive = above avg increase)")
    mi_z_score: float = Field(description="Z-score for MI change (positive = above avg increase, which is good)")

    impact_score: float = Field(
        description="Composite score: positive = above-average quality improvement, negative = below-average"
    )
    impact_category: ImpactCategory = Field(description="Categorical assessment of impact")

    cc_delta: float = Field(description="Raw CC change from staged files")
    effort_delta: float = Field(description="Raw Effort change from staged files")
    mi_delta: float = Field(description="Raw MI change from staged files")


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
    """Complete analysis of uncommitted changes against repository baseline."""

    repository_path: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    changed_files: list[str] = Field(description="List of changed Python files (staged and unstaged)")
    current_metrics: ExtendedComplexityMetrics = Field(description="Metrics with uncommitted changes applied")
    baseline_metrics: ExtendedComplexityMetrics = Field(description="Metrics at current HEAD")

    assessment: ImpactAssessment
    baseline: RepoBaseline

    blind_spots: list[str] = Field(
        default_factory=list, description="Files dependent on changed files but not in changed set"
    )
    filtered_coverage: dict[str, float] | None = Field(default=None, description="Coverage % for changed files")

    # Token impact metrics
    blind_spot_tokens: int = Field(default=0, description="Total tokens in blind spot files")
    changed_files_tokens: int = Field(default=0, description="Total tokens in changed files")
    complete_picture_context_size: int = Field(
        default=0, description="Sum of tokens in changed files + blind spots"
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

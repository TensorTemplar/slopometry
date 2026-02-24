"""Session tracking and plan evolution models."""

from datetime import datetime

from pydantic import BaseModel, Field

from slopometry.core.models.complexity import ComplexityDelta, ExtendedComplexityMetrics
from slopometry.core.models.hook import AgentTool, GitState, HookEventType, Project, ToolType


class TodoItem(BaseModel):
    """Represents a single todo item from Claude Code's TodoWrite/TaskCreate or OpenCode's TodoWrite."""

    content: str = Field(
        description="Task description. Maps to 'content' (OpenCode TodoWrite) or 'subject' (Claude Code TaskCreate)"
    )
    status: str = Field(
        default="pending", description="Status: pending, in_progress, completed, cancelled. Used by both sources."
    )
    activeForm: str = Field(
        default="",
        description="Present continuous form shown during execution (Claude Code TaskCreate only, empty for OpenCode)",
    )
    priority: str = Field(
        default="", description="Priority level: high, medium, low (OpenCode TodoWrite only, empty for Claude Code)"
    )


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
    changeset_tokens: int = Field(default=0, description="Tokenized git diff of uncommitted changes")
    final_context_input_tokens: int = Field(
        default=0, description="Raw input_tokens from the last assistant message (final context window size)"
    )
    explore_subagent_tokens: int = Field(default=0, description="Subagent tokens from Explore Task invocations")
    non_explore_subagent_tokens: int = Field(default=0, description="Subagent tokens from non-Explore Task invocations")

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def exploration_tokens(self) -> int:
        """Total tokens spent understanding the problem.

        Includes main agent exploration tool tokens plus Explore subagent work.
        """
        return self.exploration_input_tokens + self.exploration_output_tokens + self.explore_subagent_tokens

    @property
    def implementation_tokens(self) -> int:
        """Total tokens spent on implementation work.

        Includes incremental input/output tokens from implementation-classified
        tool invocations plus non-Explore subagent work.
        """
        return self.implementation_input_tokens + self.implementation_output_tokens + self.non_explore_subagent_tokens

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
    complexity_delta: "ComplexityDelta | None" = None
    plan_evolution: PlanEvolution | None = None
    context_coverage: "ContextCoverage | None" = None
    project: Project | None = None
    transcript_path: str | None = None
    compact_events: list[CompactEvent] = Field(
        default_factory=list, description="Compacts that occurred during session"
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

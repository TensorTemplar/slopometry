"""Hook-related base models — non-protocol types shared across the system.

The protocol-layer types (AbstractHookEvent, AbstractEventType, AbstractEventSource,
ToolCallPayload) live in `core/protocol/events.py`. The Claude-Code-specific tool
vocabulary (ToolType enum) lives in `core/protocol/adapters/claude_code.py`.

Models kept here are harness-agnostic: project identification, git state, language
guard, feedback cache, hook response shape.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentTool(StrEnum):
    """Agent tool that produced the session."""

    CLAUDE_CODE = "claude_code"
    OPENCODE = "opencode"


class ProjectLanguage(StrEnum):
    """Supported languages for complexity analysis."""

    PYTHON = "python"
    RUST = "rust"


class ProjectSource(StrEnum):
    """Source of project identification."""

    GIT = "git"
    PYPROJECT = "pyproject"


class Project(BaseModel):
    """Represents a project being worked on."""

    name: str
    source: ProjectSource


class GitState(BaseModel):
    """Represents git repository state at a point in time."""

    commit_count: int = 0
    current_branch: str | None = None
    has_uncommitted_changes: bool = False
    is_git_repo: bool = False
    commit_sha: str | None = None


class AnalysisSource(StrEnum):
    """Source of the impact analysis."""

    UNCOMMITTED_CHANGES = "uncommitted_changes"
    PREVIOUS_COMMIT = "previous_commit"


class FeedbackCacheState(BaseModel):
    """Persisted state of the feedback cache for change-based firing.

    Stored in .slopometry/feedback_cache.json. The hook only fires when
    the working tree state changes since the last time feedback was shown.
    Per-file content hashes enable computing which specific files changed.
    """

    last_key: str = Field(description="Cache key from last fire: commit_sha:working_tree_hash")
    file_hashes: dict[str, str] = Field(
        default_factory=dict,
        description="Per-file content hashes (rel_path -> BLAKE2b hex) at time of last fire",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Commit SHA at time of last fire, enables cheap cache validation via single git rev-parse",
    )


class HookOutput(BaseModel):
    """Output structure for hook responses.

    Mirrors Claude Code's hook output schema. The `decision`/`reason` pair is
    the canonical blocking feedback shape; `continue`/`stopReason`/`suppressOutput`
    are Claude-Code-specific extensions tolerated for compatibility.
    """

    continue_: bool | None = Field(None, alias="continue")
    stop_reason: str | None = Field(None, alias="stopReason")
    suppress_output: bool | None = Field(None, alias="suppressOutput")
    decision: str | None = Field(default=None, description="Decision outcome: approve, block, or undefined")
    reason: str | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


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
        if not self.detected_unsupported:
            return None
        return f"Found {', '.join(sorted(self.detected_unsupported))} files but analysis not yet supported"

"""Display data models for CLI tables and output formatting.

These are pure display models with no cross-module dependencies.
Models like QPEScore, ProjectQPEResult, etc. have been moved to baseline.py
to avoid circular imports.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ExperimentDisplayData(BaseModel):
    """Display data for experiment runs in tables."""

    id: str = Field(description="Experiment ID")
    repository_name: str = Field(description="Name of the repository")
    commits_display: str = Field(description="Formatted commit range (e.g., 'abc123 → def456')")
    start_time: str = Field(description="Formatted start time")
    duration: str = Field(description="Formatted duration or 'Running...'")
    status: str = Field(description="Current status (running, completed, failed)")


class NFPObjectiveDisplayData(BaseModel):
    """Display data for NFP objectives in tables."""

    id: str = Field(description="Objective ID")
    title: str = Field(description="Objective title")
    commits: str = Field(description="Formatted commit range")
    story_count: int = Field(description="Number of associated user stories")
    complexity: int = Field(description="Complexity metric")
    created_date: str = Field(description="Formatted creation date")


class LeaderboardEntry(BaseModel):
    """A persistent record of a project's quality score at a specific commit.

    Note: This stores metrics_json as a string to avoid circular imports.
    Use RepoBaseline for actual complexity metrics.
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

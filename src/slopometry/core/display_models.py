"""Display models for presenting data in tables and CLI output.

These models are simple data containers for formatting and presentation,
used primarily by formatters.py and CLI commands.
"""

from pydantic import BaseModel, Field


class UserStoryDisplayData(BaseModel):
    """Display data for user story entries in tables."""

    entry_id: str = Field(description="Short ID of the entry")
    date: str = Field(description="Formatted creation date")
    commits: str = Field(description="Short commit range display")
    rating: str = Field(description="Formatted rating display")
    model: str = Field(description="Model used for generation")
    repository: str = Field(description="Repository name")


class SessionDisplayData(BaseModel):
    """Display data for session entries in tables."""

    session_id: str = Field(description="Session identifier")
    start_time: str = Field(description="Formatted session start time")
    total_events: int = Field(description="Total number of events in session")
    tools_used: int = Field(description="Number of unique tools used")
    project_name: str | None = Field(description="Project name if detected")
    project_source: str | None = Field(description="Project source/path")


class FeatureDisplayData(BaseModel):
    """Display data for feature boundary entries in tables."""

    feature_id: str = Field(description="Short feature identifier")
    feature_message: str = Field(description="Feature title/message")
    commits_display: str = Field(description="Base → Head commit display")
    best_entry_id: str = Field(description="Best user story entry ID or 'N/A'")
    merge_message: str = Field(description="Merge commit message")


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

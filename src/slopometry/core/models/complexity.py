"""Complexity analysis models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Re-export core types for backwards compatibility and convenience
# These types are defined in core.py to avoid circular imports
from slopometry.core.models.core import (  # noqa: F401
    CacheUpdateError,
    ComplexityDelta,
    ComplexityMetrics,
    ExtendedComplexityMetrics,
    TokenCountError,
)


class FileAnalysisResult(BaseModel):
    """Result from analyzing a single Python file for complexity metrics."""

    model_config = {"arbitrary_types_allowed": True}

    path: str
    complexity: int
    volume: float
    difficulty: float
    effort: float
    mi: float
    tokens: int | TokenCountError | None = None
    error: str | None = None


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

"""Experiment tracking and progress models."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from slopometry.core.models.complexity import ExtendedComplexityMetrics
from slopometry.core.models.user_story import NextFeaturePrediction


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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


class ProgressDisplayData(BaseModel):
    """Display data for experiment progress rows."""

    timestamp: str = Field(description="Formatted timestamp (HH:MM:SS)")
    cli_score: str = Field(description="Formatted CLI score")
    complexity_score: str = Field(description="Formatted complexity score")
    halstead_score: str = Field(description="Formatted Halstead score")
    maintainability_score: str = Field(description="Formatted maintainability score")


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

"""User story generation and tracking models."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


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

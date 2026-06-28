"""Memory models for tracking durable facts across sessions."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Types of memories that can be stored."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class FreshnessAction(StrEnum):
    """The four reconciliation verdicts an LLM judge can return for a memory pair."""

    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    DEDUPE = "dedupe"

    @property
    def color(self) -> str:
        """Rich console color for display."""
        match self:
            case FreshnessAction.KEEP_BOTH:
                return "green"
            case FreshnessAction.MERGE:
                return "cyan"
            case FreshnessAction.SUPERSEDE:
                return "yellow"
            case FreshnessAction.DEDUPE:
                return "magenta"


class MemoryEntry(BaseModel):
    """Represents a stored memory entry."""

    id: str
    session_id: str
    project_dir: str
    memory_type: MemoryType
    content: str
    source_context: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    retained: bool = False
    superseded_by: str | None = None
    embedding: list[float] | None = None
    metadata: dict | None = None


class MemoryCandidate(BaseModel):
    """A candidate memory extracted from a transcript, before saving."""

    memory_type: MemoryType
    content: str
    source_context: str | None = None
    embedding: list[float] | None = None
    metadata: dict | None = None


class MemoryCreateRequest(BaseModel):
    """Request to create multiple memory entries from a session."""

    session_id: str
    project_dir: str
    candidates: list[MemoryCandidate]


class LLMMemoryCandidate(BaseModel):
    """Raw LLM-extracted memory candidate before enrichment.

    The extraction LLM returns a JSON array of these. ``memory_type`` is
    validated against the canonical ``MemoryType`` enum — invalid types
    cause the candidate to be skipped rather than raising.
    """

    memory_type: MemoryType
    content: str
    source_context: str | None = None


class FreshnessVerdict(BaseModel):
    """Structured LLM judge response for a single memory reconciliation pair.

    ``merged_content`` is only present when ``action == merge``.
    """

    action: FreshnessAction
    reason: str = ""
    merged_content: str | None = Field(
        default=None,
        description="Only present when action == merge",
    )

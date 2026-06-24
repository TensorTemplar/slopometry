"""Memory models for tracking durable facts across sessions."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class MemoryType(StrEnum):
    """Types of memories that can be stored."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


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

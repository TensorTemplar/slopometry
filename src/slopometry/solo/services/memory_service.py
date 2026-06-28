"""Memory management service for solo-leveler features."""

import uuid
from datetime import datetime

from slopometry.core.database import EventDatabase
from slopometry.core.models.memory import MemoryCreateRequest, MemoryEntry, MemoryType


class MemoryService:
    """Handles memory CRUD operations and session tracking."""

    def __init__(self, db: EventDatabase | None = None):
        self.db = db or EventDatabase()

    def save_memory(self, memory: MemoryEntry) -> None:
        """Save a single memory entry."""
        self.db.save_memory(memory)

    def save_memories(self, request: MemoryCreateRequest) -> list[MemoryEntry]:
        """Save multiple memory entries from a request.

        Does not mark the session as processed — the caller owns that
        decision (and must pass the correct ``source`` to avoid
        cross-harness key collisions in ``processed_memory_sessions``).

        Returns:
            List of saved MemoryEntry objects
        """
        created_at = datetime.now()
        saved_memories: list[MemoryEntry] = []

        for candidate in request.candidates:
            memory = MemoryEntry(
                id=str(uuid.uuid4()),
                session_id=request.session_id,
                project_dir=request.project_dir,
                memory_type=candidate.memory_type,
                content=candidate.content,
                source_context=candidate.source_context,
                embedding=candidate.embedding,
                metadata=candidate.metadata,
                created_at=created_at,
            )
            self.db.save_memory(memory)
            saved_memories.append(memory)

        return saved_memories

    def get_memories(
        self,
        project_dir: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 50,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """Get memories with optional filters.

        Args:
            project_dir: Filter by project directory
            memory_type: Filter by memory type
            limit: Maximum number of results
            include_superseded: When False (default), exclude memories that
                have been superseded by a newer replacement.

        Returns:
            List of matching MemoryEntry objects
        """
        return self.db.get_memories(
            project_dir=project_dir,
            memory_type=memory_type.value if memory_type else None,
            limit=limit,
            include_superseded=include_superseded,
        )

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Returns:
            True if deleted, False if not found
        """
        return self.db.delete_memory(memory_id)

    def delete_all_memories(self) -> int:
        """Delete all memories.

        Returns:
            Number of memories deleted
        """
        return self.db.delete_all_memories()

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        retained: bool | None = None,
        superseded_by: str | None = None,
        source_context: str | None = None,
        embedding: list[float] | None = None,
    ) -> bool:
        """Update a memory entry.

        Returns:
            True if updated, False if not found
        """
        return self.db.update_memory(
            memory_id,
            content=content,
            retained=retained,
            superseded_by=superseded_by,
            source_context=source_context,
            embedding=embedding,
        )

    def retire_memory(self, memory_id: str, reason: str) -> bool:
        """Retire a memory (mark as stale without a direct replacement).

        Returns:
            True if retired, False if not found
        """
        return self.db.retire_memory(memory_id, reason)

    def mark_session_processed(
        self, session_id: str, project_dir: str, memory_count: int, source: str
    ) -> None:
        """Mark a session as processed for memory extraction."""
        self.db.mark_session_processed(session_id, project_dir, memory_count, source=source)

    def is_session_processed(self, session_id: str, project_dir: str, source: str) -> bool:
        """Check if a session has already been processed."""
        return self.db.is_session_processed(session_id, project_dir, source=source)

    def get_memory_stats(self, project_dir: str | None = None) -> dict:
        """Get statistics about stored memories.

        Returns:
            Dict with total count and breakdown by type
        """
        return self.db.get_memory_stats(project_dir=project_dir)

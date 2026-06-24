"""Test memory models."""

from datetime import datetime

from slopometry.core.models.memory import MemoryCandidate, MemoryCreateRequest, MemoryEntry, MemoryType


def test_memory_entry_model() -> None:
    """Validates MemoryEntry creation."""
    entry = MemoryEntry(
        id="test-id-123",
        session_id="session-abc",
        project_dir="/path/to/project",
        memory_type=MemoryType.USER,
        content="User prefers dark mode",
        source_context="From conversation about UI settings",
        created_at=datetime.now(),
        embedding=[0.1, 0.2, 0.3],
        retained=False,
    )

    assert entry.id == "test-id-123"
    assert entry.session_id == "session-abc"
    assert entry.project_dir == "/path/to/project"
    assert entry.memory_type == MemoryType.USER
    assert entry.content == "User prefers dark mode"
    assert entry.source_context == "From conversation about UI settings"
    assert entry.embedding == [0.1, 0.2, 0.3]
    assert entry.retained is False
    assert entry.metadata is None


def test_memory_candidate_model() -> None:
    """Validates MemoryCandidate creation."""
    candidate = MemoryCandidate(
        memory_type=MemoryType.PROJECT,
        content="Project uses pytest for testing",
        source_context="From analysis of project structure",
        embedding=[0.5, 0.6, 0.7],
    )

    assert candidate.memory_type == MemoryType.PROJECT
    assert candidate.content == "Project uses pytest for testing"
    assert candidate.source_context == "From analysis of project structure"
    assert candidate.embedding == [0.5, 0.6, 0.7]


def test_memory_create_request() -> None:
    """Validates MemoryCreateRequest."""
    candidates = [
        MemoryCandidate(
            memory_type=MemoryType.USER,
            content="User likes tab indentation",
        ),
        MemoryCandidate(
            memory_type=MemoryType.FEEDBACK,
            content="Previous code was too complex",
            embedding=[0.1, 0.2, 0.3],
        ),
    ]

    request = MemoryCreateRequest(
        session_id="session-xyz",
        project_dir="/path/to/project",
        candidates=candidates,
    )

    assert request.session_id == "session-xyz"
    assert request.project_dir == "/path/to/project"
    assert len(request.candidates) == 2
    assert request.candidates[0].memory_type == MemoryType.USER
    assert request.candidates[1].memory_type == MemoryType.FEEDBACK


def test_memory_entry_defaults() -> None:
    """Test MemoryEntry default values."""
    entry = MemoryEntry(
        id="id-1",
        session_id="session-1",
        project_dir="/project",
        memory_type=MemoryType.REFERENCE,
        content="Some content",
        created_at=datetime.now(),
    )

    assert entry.retained is False
    assert entry.source_context is None
    assert entry.embedding is None
    assert entry.metadata is None
    assert entry.updated_at is None


def test_memory_type_enum_values() -> None:
    """Test MemoryType enum values."""
    assert MemoryType.USER == "user"
    assert MemoryType.FEEDBACK == "feedback"
    assert MemoryType.PROJECT == "project"
    assert MemoryType.REFERENCE == "reference"

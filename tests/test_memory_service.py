"""Test memory service."""

import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from slopometry.core.database import EventDatabase
from slopometry.core.models.memory import MemoryCandidate, MemoryCreateRequest, MemoryEntry, MemoryType
from slopometry.solo.services.memory_service import MemoryService


@pytest.fixture
def temp_db() -> Iterator[EventDatabase]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        db = EventDatabase(db_path)
        yield db
    finally:
        if db_path.exists():
            db_path.unlink()


@pytest.fixture
def memory_service(temp_db: EventDatabase) -> MemoryService:
    """Create a MemoryService with temporary database."""
    return MemoryService(db=temp_db)


def test_save_memory__persists_single_memory_entry(memory_service: MemoryService) -> None:
    memory = MemoryEntry(
        id="mem-001",
        session_id="session-abc",
        project_dir="/test/project",
        memory_type=MemoryType.USER,
        content="Test memory content",
        created_at=datetime.now(),
    )

    memory_service.save_memory(memory)

    memories = memory_service.get_memories()
    assert len(memories) == 1
    assert memories[0].id == "mem-001"
    assert memories[0].content == "Test memory content"


def test_save_memories__saves_all_candidates_from_request(memory_service: MemoryService) -> None:
    request = MemoryCreateRequest(
        session_id="session-xyz",
        project_dir="/test/project",
        candidates=[
            MemoryCandidate(
                memory_type=MemoryType.USER,
                content="First memory",
            ),
            MemoryCandidate(
                memory_type=MemoryType.PROJECT,
                content="Second memory",
            ),
        ],
    )

    saved = memory_service.save_memories(request)

    assert len(saved) == 2
    assert saved[0].content == "First memory"
    assert saved[1].content == "Second memory"
    assert saved[0].session_id == "session-xyz"


def test_save_memories__does_not_mark_session_as_processed(memory_service: MemoryService) -> None:
    request = MemoryCreateRequest(
        session_id="session-no-mark",
        project_dir="/test/project",
        candidates=[
            MemoryCandidate(
                memory_type=MemoryType.USER,
                content="A memory",
            ),
        ],
    )

    memory_service.save_memories(request)

    assert not memory_service.is_session_processed("session-no-mark", "/test/project", source="claude_code")


def test_get_memories__filters_by_project_dir_and_memory_type(memory_service: MemoryService) -> None:
    memory_service.save_memory(
        MemoryEntry(
            id="mem-001",
            session_id="session-1",
            project_dir="/project1",
            memory_type=MemoryType.USER,
            content="User memory",
            created_at=datetime.now(),
        )
    )
    memory_service.save_memory(
        MemoryEntry(
            id="mem-002",
            session_id="session-2",
            project_dir="/project2",
            memory_type=MemoryType.PROJECT,
            content="Project memory",
            created_at=datetime.now(),
        )
    )

    memories = memory_service.get_memories()
    assert len(memories) == 2

    project_memories = memory_service.get_memories(project_dir="/project2")
    assert len(project_memories) == 1
    assert project_memories[0].memory_type == MemoryType.PROJECT

    type_memories = memory_service.get_memories(memory_type=MemoryType.USER)
    assert len(type_memories) == 1
    assert type_memories[0].id == "mem-001"


def test_get_memories__respects_limit_parameter(memory_service: MemoryService) -> None:
    for i in range(10):
        memory_service.save_memory(
            MemoryEntry(
                id=f"mem-{i}",
                session_id="session-1",
                project_dir="/test/project",
                memory_type=MemoryType.USER,
                content=f"Memory {i}",
                created_at=datetime.now(),
            )
        )

    memories = memory_service.get_memories(limit=5)
    assert len(memories) == 5


def test_get_memories__excludes_superseded_by_default(memory_service: MemoryService) -> None:
    old = MemoryEntry(
        id="mem-old",
        session_id="session-1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="Old memory",
        created_at=datetime.now(),
    )
    new = MemoryEntry(
        id="mem-new",
        session_id="session-2",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="New memory that supersedes old",
        created_at=datetime.now(),
    )
    memory_service.save_memory(old)
    memory_service.save_memory(new)
    memory_service.update_memory(old.id, superseded_by=new.id)

    visible = memory_service.get_memories(project_dir="/proj", limit=100)
    assert len(visible) == 1
    assert visible[0].id == "mem-new"


def test_get_memories__includes_superseded_when_flag_set(memory_service: MemoryService) -> None:
    old = MemoryEntry(
        id="mem-old",
        session_id="session-1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="Old memory",
        created_at=datetime.now(),
    )
    new = MemoryEntry(
        id="mem-new",
        session_id="session-2",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="New memory that supersedes old",
        created_at=datetime.now(),
    )
    memory_service.save_memory(old)
    memory_service.save_memory(new)
    memory_service.update_memory(old.id, superseded_by=new.id)

    all_memories = memory_service.get_memories(project_dir="/proj", limit=100, include_superseded=True)
    assert len(all_memories) == 2
    ids = {m.id for m in all_memories}
    assert ids == {"mem-old", "mem-new"}


def test_delete_memory__returns_true_when_exists_false_when_not(memory_service: MemoryService) -> None:
    memory = MemoryEntry(
        id="mem-to-delete",
        session_id="session-1",
        project_dir="/test/project",
        memory_type=MemoryType.REFERENCE,
        content="Memory to delete",
        created_at=datetime.now(),
    )
    memory_service.save_memory(memory)

    result = memory_service.delete_memory("mem-to-delete")
    assert result is True

    memories = memory_service.get_memories()
    assert len(memories) == 0

    result = memory_service.delete_memory("non-existent")
    assert result is False


def test_delete_all_memories__clears_memories_and_processed_sessions(memory_service: MemoryService) -> None:
    memory_service.save_memory(
        MemoryEntry(
            id="mem-1",
            session_id="session-1",
            project_dir="/project1",
            memory_type=MemoryType.USER,
            content="Memory 1",
            created_at=datetime.now(),
        )
    )
    memory_service.save_memory(
        MemoryEntry(
            id="mem-2",
            session_id="session-2",
            project_dir="/project2",
            memory_type=MemoryType.PROJECT,
            content="Memory 2",
            created_at=datetime.now(),
        )
    )
    memory_service.mark_session_processed("session-1", "/project1", 1, source="claude_code")

    assert memory_service.is_session_processed("session-1", "/project1", source="claude_code") is True

    count = memory_service.delete_all_memories()
    assert count == 2

    memories = memory_service.get_memories()
    assert len(memories) == 0

    assert memory_service.is_session_processed("session-1", "/project1", source="claude_code") is False


def test_mark_session_processed__marks_session_for_source(memory_service: MemoryService) -> None:
    memory_service.mark_session_processed("session-test", "/test/project", 5, source="claude_code")

    assert memory_service.is_session_processed("session-test", "/test/project", source="claude_code") is True


def test_is_session_processed__returns_false_before_true_after_marking(memory_service: MemoryService) -> None:
    assert memory_service.is_session_processed("unprocessed-session", "/any/project", source="claude_code") is False

    memory_service.mark_session_processed("processed-session", "/any/project", 3, source="claude_code")

    assert memory_service.is_session_processed("processed-session", "/any/project", source="claude_code") is True


def _make_memory(mem_id: str, content: str = "content", project: str = "/proj") -> MemoryEntry:
    return MemoryEntry(
        id=mem_id,
        session_id="s1",
        project_dir=project,
        memory_type=MemoryType.PROJECT,
        content=content,
        created_at=datetime.now(),
    )


def test_retire_memory__marks_memory_with_retired_reason_and_hides_from_default_query(
    memory_service: MemoryService,
) -> None:
    memory_service.save_memory(_make_memory("mem-active"))
    memory_service.save_memory(_make_memory("mem-stale", content="describes a fixed bug"))

    result = memory_service.retire_memory("mem-stale", reason="bug was fixed in session abc")

    assert result is True
    visible = memory_service.get_memories(project_dir="/proj", limit=100)
    assert {m.id for m in visible} == {"mem-active"}


def test_retire_memory__returns_false_when_memory_id_does_not_exist(memory_service: MemoryService) -> None:
    result = memory_service.retire_memory("nonexistent-id", reason="no such memory")
    assert result is False


def test_get_memories__includes_retired_when_include_superseded_is_true(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-active"))
    memory_service.save_memory(_make_memory("mem-retired", content="stale"))
    memory_service.retire_memory("mem-retired", reason="no longer relevant")

    all_memories = memory_service.get_memories(project_dir="/proj", limit=100, include_superseded=True)
    ids = {m.id for m in all_memories}
    assert ids == {"mem-active", "mem-retired"}


def test_get_memories__excludes_both_superseded_and_retired_by_default(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-active"))
    memory_service.save_memory(_make_memory("mem-superseded", content="old version"))
    memory_service.save_memory(_make_memory("mem-retired", content="fixed bug"))
    memory_service.update_memory("mem-superseded", superseded_by="mem-active")
    memory_service.retire_memory("mem-retired", reason="bug was fixed")

    visible = memory_service.get_memories(project_dir="/proj", limit=100)
    assert {m.id for m in visible} == {"mem-active"}


def test_get_memories__retired_memory_carries_retired_reason_when_included(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-retired", content="stale"))
    memory_service.retire_memory("mem-retired", reason="work was completed")

    all_memories = memory_service.get_memories(project_dir="/proj", limit=100, include_superseded=True)
    retired = next(m for m in all_memories if m.id == "mem-retired")
    assert retired.retired_reason == "work was completed"


def test_get_memory_stats__excludes_retired_memories_from_count(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-active", content="active"))
    memory_service.save_memory(_make_memory("mem-retired", content="stale"))
    memory_service.retire_memory("mem-retired", reason="stale")

    stats = memory_service.get_memory_stats(project_dir="/proj")
    assert stats["total"] == 1


def test_retire_memory__does_not_interfere_with_supersede_chain(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-old", content="version 1"))
    memory_service.save_memory(_make_memory("mem-new", content="version 2"))
    memory_service.update_memory("mem-old", superseded_by="mem-new")
    memory_service.retire_memory("mem-new", reason="superseded work was completed")

    visible = memory_service.get_memories(project_dir="/proj", limit=100)
    assert visible == []


def test_retire_memory__can_be_called_multiple_times_on_same_memory(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-stale"))
    assert memory_service.retire_memory("mem-stale", reason="first reason") is True
    assert memory_service.retire_memory("mem-stale", reason="updated reason") is True

    all_memories = memory_service.get_memories(project_dir="/proj", limit=100, include_superseded=True)
    retired = next(m for m in all_memories if m.id == "mem-stale")
    assert retired.retired_reason == "updated reason"


def test_retire_memory__does_not_clobber_superseded_by_field(memory_service: MemoryService) -> None:
    memory_service.save_memory(_make_memory("mem-old", content="old version"))
    memory_service.save_memory(_make_memory("mem-new", content="new version"))
    memory_service.update_memory("mem-old", superseded_by="mem-new")
    memory_service.retire_memory("mem-old", reason="also stale")

    all_memories = memory_service.get_memories(project_dir="/proj", limit=100, include_superseded=True)
    old = next(m for m in all_memories if m.id == "mem-old")
    assert old.superseded_by == "mem-new"
    assert old.retired_reason == "also stale"

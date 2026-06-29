"""Test memory models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from slopometry.core.models.memory import (
    FreshnessAction,
    FreshnessVerdict,
    LLMMemoryCandidate,
    MemoryCandidate,
    MemoryCreateRequest,
    MemoryEntry,
    MemoryType,
    StalenessVerdict,
)


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
    assert entry.superseded_by is None
    assert entry.retired_reason is None


def test_memory_type_enum_values() -> None:
    """Test MemoryType enum values."""
    assert MemoryType.USER == "user"
    assert MemoryType.FEEDBACK == "feedback"
    assert MemoryType.PROJECT == "project"
    assert MemoryType.REFERENCE == "reference"


def test_memory_entry__defaults_retired_reason_and_superseded_by_to_none() -> None:
    entry = MemoryEntry(
        id="id-retired",
        session_id="s1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="content",
        created_at=datetime.now(),
    )
    assert entry.retired_reason is None
    assert entry.superseded_by is None


def test_memory_entry__accepts_retired_reason_when_stale() -> None:
    entry = MemoryEntry(
        id="id-stale",
        session_id="s1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content="describes a bug that was fixed",
        created_at=datetime.now(),
        retired_reason="bug was fixed in session abc-123",
    )
    assert entry.retired_reason == "bug was fixed in session abc-123"


def test_freshness_verdict__defaults_merged_content_to_none_for_non_merge_actions() -> None:
    verdict = FreshnessVerdict(action=FreshnessAction.SUPERSEDE, reason="newer version wins")
    assert verdict.merged_content is None


def test_freshness_verdict__accepts_merged_content_when_action_is_merge() -> None:
    verdict = FreshnessVerdict(
        action=FreshnessAction.MERGE,
        reason="old was outdated",
        merged_content="uses rust-code-analysis since 2026",
    )
    assert verdict.merged_content == "uses rust-code-analysis since 2026"


def test_freshness_verdict__defaults_reason_to_empty_string() -> None:
    verdict = FreshnessVerdict(action=FreshnessAction.KEEP_BOTH)
    assert verdict.reason == ""


def test_freshness_action__has_four_distinct_values() -> None:
    actions = {FreshnessAction.KEEP_BOTH, FreshnessAction.MERGE, FreshnessAction.SUPERSEDE, FreshnessAction.DEDUPE}
    assert len(actions) == 4


def test_freshness_action__color_property_returns_valid_color_for_each_action() -> None:
    assert FreshnessAction.KEEP_BOTH.color == "green"
    assert FreshnessAction.MERGE.color == "cyan"
    assert FreshnessAction.SUPERSEDE.color == "yellow"
    assert FreshnessAction.DEDUPE.color == "magenta"


def test_staleness_verdict__accepts_positive_ref_and_reason() -> None:
    verdict = StalenessVerdict(ref=1, reason="bug was fixed in this session")
    assert verdict.ref == 1
    assert verdict.reason == "bug was fixed in this session"


def test_staleness_verdict__defaults_reason_to_empty_string() -> None:
    verdict = StalenessVerdict(ref=3)
    assert verdict.reason == ""


def test_freshness_verdict__rejects_invalid_action_string_via_validation() -> None:
    with pytest.raises(ValidationError):
        FreshnessVerdict.model_validate({"action": "invalid_action"})


def test_llm_memory_candidate__validates_memory_type_against_enum() -> None:
    candidate = LLMMemoryCandidate(
        memory_type=MemoryType.PROJECT,
        content="uses rust-code-analysis",
    )
    assert candidate.memory_type == MemoryType.PROJECT
    assert candidate.source_context is None

"""Integration test: real OpenCode session → memory extraction → freshness → save.

Reads the actual OpenCode storage tree on the test host (if present),
extracts conversation text via MemoryExtractor.extract_memories_from_opencode_session,
then runs the full freshness + save pipeline with stubbed LLM candidates to
verify the end-to-end wiring of CLI/logic without depending on a live chat LLM.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.database import EventDatabase
from slopometry.core.models.memory import MemoryCandidate, MemoryEntry, MemoryType
from slopometry.solo.services.memory_extractor import MemoryExtractor
from slopometry.solo.services.memory_freshness import MemoryFreshnessValidator
from slopometry.solo.services.memory_service import MemoryService


OPENCODE_STORAGE = Path(os.environ.get("OPENCODE_STORAGE", "/home/tensor-templar/.local/share/opencode/storage"))


@pytest.fixture
def real_opencode_session() -> tuple[str, Path]:
    """Pick the first OpenCode session that has a message directory, or skip."""
    if not OPENCODE_STORAGE.is_dir():
        pytest.skip(f"OpenCode storage not found at {OPENCODE_STORAGE}")
    message_root = OPENCODE_STORAGE / "message"
    if not message_root.is_dir():
        pytest.skip(f"No message directory at {message_root}")
    sessions = sorted(p.name for p in message_root.iterdir() if p.is_dir())
    if not sessions:
        pytest.skip("No OpenCode sessions with messages found")
    return sessions[0], OPENCODE_STORAGE


@pytest.fixture
def fresh_memory_service(tmp_path: Path) -> MemoryService:
    """MemoryService backed by a fresh tmp database, isolated from the global one."""
    db = EventDatabase(db_path=tmp_path / "test.db")
    return MemoryService(db=db)


class TestRealOpenCodeSessionExtraction:
    def test_extract_produces_non_empty_conversation(self, real_opencode_session: tuple[str, Path]):
        session_id, storage_root = real_opencode_session
        extractor = MemoryExtractor("https://llm.example/v1", "model-x")
        text = extractor.extract_memories_from_opencode_session(session_id, storage_root)
        assert text.strip()
        assert "USER:" in text or "ASSISTANT:" in text, (
            f"Expected USER:/ASSISTANT: markers in reconstructed text from {session_id}, "
            f"got first 200 chars: {text[:200]!r}"
        )

    def test_extracted_text_contains_some_tool_markers(self, real_opencode_session: tuple[str, Path]):
        session_id, storage_root = real_opencode_session
        extractor = MemoryExtractor("https://llm.example/v1", "model-x")
        text = extractor.extract_memories_from_opencode_session(session_id, storage_root)
        has_tool = "TOOL:" in text
        if not has_tool:
            pytest.skip(f"Session {session_id} has no tool parts (text-only conversation)")


class TestEndToEndFreshnessPipeline:
    """Full pipeline: stubbed LLM candidates → freshness validator → save + superseded_by."""

    def _stub_judge(self, action: str, merged_content: str | None = None) -> MagicMock:
        payload = {"action": action, "reason": "stub"}
        if merged_content:
            payload["merged_content"] = merged_content
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
        return mock

    def test_supersede_links_old_to_new_via_superseded_by(self, fresh_memory_service: MemoryService):
        existing = MemoryEntry(
            id="old-1",
            session_id="claude_code:s_prev",
            project_dir="/test/proj",
            memory_type=MemoryType.PROJECT,
            content="Project uses radon for Python complexity metrics",
            embedding=[1.0, 0.0, 0.0],
            created_at=datetime.now(),
        )
        fresh_memory_service.save_memory(existing)

        new_candidate = MemoryCandidate(
            memory_type=MemoryType.PROJECT,
            content="Project uses rust-code-analysis (switched from radon in 2026)",
            embedding=[0.99, 0.14, 0.0],
        )

        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._stub_judge("supersede")
            decisions, _ = validator.validate([new_candidate], [existing])

        assert len(decisions) == 1
        assert decisions[0].action == "supersede"

        from slopometry.core.models.memory import MemoryCreateRequest

        saved = fresh_memory_service.save_memories(
            MemoryCreateRequest(
                session_id="claude_code:test",
                project_dir="/test/proj",
                candidates=[new_candidate],
            )
        )
        new_id = saved[0].id

        fresh_memory_service.update_memory(existing.id, superseded_by=new_id)

        all_memories = fresh_memory_service.get_memories(project_dir="/test/proj", limit=100)
        old_updated = next(m for m in all_memories if m.id == existing.id)
        new_loaded = next(m for m in all_memories if m.id == new_id)
        assert old_updated.superseded_by == new_id
        assert new_loaded.superseded_by is None

    def test_merge_action_rewrites_candidate_content(self, fresh_memory_service: MemoryService):
        existing = MemoryEntry(
            id="old-2",
            session_id="claude_code:s_prev2",
            project_dir="/test/proj",
            memory_type=MemoryType.PROJECT,
            content="Project uses radon",
            embedding=[1.0, 0.0, 0.0],
            created_at=datetime.now(),
        )
        fresh_memory_service.save_memory(existing)

        new_candidate = MemoryCandidate(
            memory_type=MemoryType.PROJECT,
            content="Project switched to rust-code-analysis in 2026",
            embedding=[0.99, 0.14, 0.0],
        )

        merged_text = "Project uses rust-code-analysis (switched from radon in 2026)"

        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._stub_judge(
                "merge", merged_content=merged_text
            )
            decisions, _ = validator.validate([new_candidate], [existing])

        assert decisions[0].action == "merge"
        assert decisions[0].merged_content == merged_text

        decisions[0].new_candidate.content = decisions[0].merged_content

        from slopometry.core.models.memory import MemoryCreateRequest

        saved = fresh_memory_service.save_memories(
            MemoryCreateRequest(
                session_id="claude_code:test",
                project_dir="/test/proj",
                candidates=[decisions[0].new_candidate],
            )
        )
        assert saved[0].content == merged_text
        all_memories = fresh_memory_service.get_memories(project_dir="/test/proj", limit=100)
        old_loaded = next(m for m in all_memories if m.id == existing.id)
        assert old_loaded.superseded_by is None

    def test_dedupe_action_skips_new_save(self, fresh_memory_service: MemoryService):
        existing = MemoryEntry(
            id="old-3",
            session_id="claude_code:s_prev3",
            project_dir="/test/proj",
            memory_type=MemoryType.PROJECT,
            content="User prefers pyright",
            embedding=[1.0, 0.0, 0.0],
            created_at=datetime.now(),
        )
        fresh_memory_service.save_memory(existing)

        new_candidate = MemoryCandidate(
            memory_type=MemoryType.PROJECT,
            content="User uses pyright type checker",
            embedding=[0.99, 0.14, 0.0],
        )

        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._stub_judge("dedupe")
            decisions, _ = validator.validate([new_candidate], [existing])

        assert len(decisions) == 1
        assert decisions[0].action == "dedupe"

        deduped_candidates: list[MemoryCandidate] = []
        for d in decisions:
            if d.action == "dedupe":
                if d.new_candidate.metadata is None:
                    d.new_candidate.metadata = {}
                d.new_candidate.metadata["deduped_against"] = d.existing_memory.id
                continue
            deduped_candidates.append(d.new_candidate)

        assert deduped_candidates == []
        all_memories = fresh_memory_service.get_memories(project_dir="/test/proj", limit=100)
        assert len(all_memories) == 1
        assert all_memories[0].id == existing.id

    def test_keep_both_saves_both_independently(self, fresh_memory_service: MemoryService):
        existing = MemoryEntry(
            id="old-4",
            session_id="claude_code:s_prev4",
            project_dir="/test/proj",
            memory_type=MemoryType.PROJECT,
            content="Project uses rust-code-analysis",
            embedding=[1.0, 0.0, 0.0],
            created_at=datetime.now(),
        )
        fresh_memory_service.save_memory(existing)

        new_candidate = MemoryCandidate(
            memory_type=MemoryType.PROJECT,
            content="User prefers dark mode",
            embedding=[0.95, 0.31, 0.0],
        )

        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = self._stub_judge("keep_both")
            decisions, _ = validator.validate([new_candidate], [existing])

        assert len(decisions) == 1
        assert decisions[0].action == "keep_both"

        from slopometry.core.models.memory import MemoryCreateRequest

        fresh_memory_service.save_memories(
            MemoryCreateRequest(
                session_id="claude_code:test",
                project_dir="/test/proj",
                candidates=[decisions[0].new_candidate],
            )
        )

        all_memories = fresh_memory_service.get_memories(project_dir="/test/proj", limit=100)
        assert len(all_memories) == 2
        old_loaded = next(m for m in all_memories if m.id == existing.id)
        assert old_loaded.superseded_by is None

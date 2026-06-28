"""Integration tests for the prune-memories CLI command.

These tests exercise the full click command pipeline, mocking only the
external boundaries (LLM calls, preflight endpoint checks, transcript
discovery, and memory extraction) to verify the command's behavior.
"""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from slopometry.cli import cli
from slopometry.core.database import EventDatabase
from slopometry.core.models.memory import MemoryEntry, MemoryType
from slopometry.solo.services.memory_service import MemoryService


@pytest.fixture
def temp_db() -> Iterator[EventDatabase]:
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
    return MemoryService(db=temp_db)


def _make_memory(mem_id: str, content: str = "content", project: str = "/proj") -> MemoryEntry:
    return MemoryEntry(
        id=mem_id,
        session_id="s1",
        project_dir=project,
        memory_type=MemoryType.PROJECT,
        content=content,
        created_at=datetime.now(),
    )


@contextmanager
def patch_prune_externals(memory_service: MemoryService):
    """Patch all external boundaries of prune-memories."""
    api_key_mock = MagicMock(get_secret_value=lambda: "key")
    with (
        patch("slopometry.core.settings.settings.offline_mode", False),
        patch("slopometry.core.settings.settings.memory_llm_endpoint", "https://llm.example/v1"),
        patch("slopometry.core.settings.settings.memory_llm_model", "model-x"),
        patch("slopometry.core.settings.settings.memory_llm_api_key", api_key_mock),
        patch("slopometry.core.settings.settings.memory_embedding_endpoint", "https://embed.example/v1"),
        patch("slopometry.core.settings.settings.memory_embedding_model", "embed-model"),
        patch("slopometry.core.settings.settings.memory_embedding_api_key", api_key_mock),
        patch("slopometry.solo.cli.preflight._check_endpoint", return_value=None),
        patch("slopometry.solo.services.memory_service.MemoryService", return_value=memory_service),
    ):
        yield


class TestPruneMemoriesCli:
    def test_prune_memories__returns_early_when_no_active_memories(
        self, memory_service: MemoryService, tmp_path: Path
    ):
        with patch_prune_externals(memory_service):
            runner = CliRunner()
            result = runner.invoke(cli, ["solo", "prune-memories", "--project-dir", str(tmp_path)])
            assert result.exit_code == 0
            assert "No active memories to audit" in result.output

    def test_prune_memories__returns_early_when_no_transcripts_found(
        self, memory_service: MemoryService, tmp_path: Path
    ):
        memory_service.save_memory(_make_memory("mem-1", content="stale memory", project=str(tmp_path)))

        with (
            patch_prune_externals(memory_service),
            patch("slopometry.solo.services.transcript_finder.TranscriptFinder") as mock_tf,
        ):
            mock_tf.return_value.discover_transcripts.return_value = []

            runner = CliRunner()
            result = runner.invoke(cli, ["solo", "prune-memories", "--project-dir", str(tmp_path)])
            assert result.exit_code == 0
            assert "No transcripts found to audit against" in result.output

    def test_prune_memories__retires_stale_memories_identified_by_llm(
        self, memory_service: MemoryService, tmp_path: Path
    ):
        memory_service.save_memory(
            _make_memory("mem-active", content="user prefers dark mode", project=str(tmp_path))
        )
        memory_service.save_memory(
            _make_memory("mem-stale", content="describes a fixed bug", project=str(tmp_path))
        )

        mock_transcript = MagicMock()
        mock_transcript.session_id = "s1"
        mock_transcript.transcript_path = tmp_path / "transcript.jsonl"
        mock_transcript.project_dir = tmp_path
        mock_transcript.source = MagicMock(value="claude_code")

        mock_extractor = MagicMock()
        mock_extractor.extract_memories_from_transcript.return_value = "transcript showing bug fix"

        staleness_response = MagicMock()
        staleness_response.choices = [
            MagicMock(message=MagicMock(content='[{"ref": 1, "reason": "bug was fixed in session"}]'))
        ]

        with (
            patch_prune_externals(memory_service),
            patch("slopometry.solo.services.transcript_finder.TranscriptFinder") as mock_tf_class,
            patch("slopometry.solo.services.memory_extractor.MemoryExtractor", return_value=mock_extractor),
            patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai,
        ):
            mock_tf_class.return_value.discover_transcripts.return_value = [mock_transcript]
            mock_tf_class.return_value.find_opencode_storage_root.return_value = tmp_path / "opencode"
            mock_openai.return_value.chat.completions.create.return_value = staleness_response

            runner = CliRunner()
            result = runner.invoke(cli, ["solo", "prune-memories", "--project-dir", str(tmp_path)])
            assert result.exit_code == 0
            assert "Retired 1 memor" in result.output

            visible = memory_service.get_memories(project_dir=str(tmp_path), limit=100)
            assert {m.id for m in visible} == {"mem-active"}

    def test_prune_memories__dry_run_does_not_persist_retirements(
        self, memory_service: MemoryService, tmp_path: Path
    ):
        memory_service.save_memory(
            _make_memory("mem-stale", content="describes a fixed bug", project=str(tmp_path))
        )

        mock_transcript = MagicMock()
        mock_transcript.session_id = "s1"
        mock_transcript.transcript_path = tmp_path / "transcript.jsonl"
        mock_transcript.project_dir = tmp_path
        mock_transcript.source = MagicMock(value="claude_code")

        mock_extractor = MagicMock()
        mock_extractor.extract_memories_from_transcript.return_value = "transcript showing bug fix"

        staleness_response = MagicMock()
        staleness_response.choices = [
            MagicMock(message=MagicMock(content='[{"ref": 1, "reason": "bug was fixed"}]'))
        ]

        with (
            patch_prune_externals(memory_service),
            patch("slopometry.solo.services.transcript_finder.TranscriptFinder") as mock_tf_class,
            patch("slopometry.solo.services.memory_extractor.MemoryExtractor", return_value=mock_extractor),
            patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai,
        ):
            mock_tf_class.return_value.discover_transcripts.return_value = [mock_transcript]
            mock_tf_class.return_value.find_opencode_storage_root.return_value = tmp_path / "opencode"
            mock_openai.return_value.chat.completions.create.return_value = staleness_response

            runner = CliRunner()
            result = runner.invoke(
                cli, ["solo", "prune-memories", "--project-dir", str(tmp_path), "--dry-run"]
            )
            assert result.exit_code == 0
            assert "--dry-run: would retire 1" in result.output

            all_memories = memory_service.get_memories(
                project_dir=str(tmp_path), limit=100, include_superseded=True
            )
            assert len(all_memories) == 1
            assert all_memories[0].retired_reason is None

    def test_prune_memories__prints_no_stale_when_llm_returns_empty_array(
        self, memory_service: MemoryService, tmp_path: Path
    ):
        memory_service.save_memory(
            _make_memory("mem-active", content="stable preference", project=str(tmp_path))
        )

        mock_transcript = MagicMock()
        mock_transcript.session_id = "s1"
        mock_transcript.transcript_path = tmp_path / "transcript.jsonl"
        mock_transcript.project_dir = tmp_path
        mock_transcript.source = MagicMock(value="claude_code")

        mock_extractor = MagicMock()
        mock_extractor.extract_memories_from_transcript.return_value = "transcript about unrelated work"

        staleness_response = MagicMock()
        staleness_response.choices = [MagicMock(message=MagicMock(content="[]"))]

        with (
            patch_prune_externals(memory_service),
            patch("slopometry.solo.services.transcript_finder.TranscriptFinder") as mock_tf_class,
            patch("slopometry.solo.services.memory_extractor.MemoryExtractor", return_value=mock_extractor),
            patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai,
        ):
            mock_tf_class.return_value.discover_transcripts.return_value = [mock_transcript]
            mock_tf_class.return_value.find_opencode_storage_root.return_value = tmp_path / "opencode"
            mock_openai.return_value.chat.completions.create.return_value = staleness_response

            runner = CliRunner()
            result = runner.invoke(cli, ["solo", "prune-memories", "--project-dir", str(tmp_path)])
            assert result.exit_code == 0
            assert "No stale memories found" in result.output

            visible = memory_service.get_memories(project_dir=str(tmp_path), limit=100)
            assert len(visible) == 1

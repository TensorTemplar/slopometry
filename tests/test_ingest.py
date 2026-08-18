"""Tests for envelope ingestion into slopometry storage."""

from datetime import datetime
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from slopometry.cli import cli
from slopometry.core.database import EventDatabase
from slopometry.core.protocol.ingest import ingest_envelopes
from slopometry.core.protocol.kinds import EventKind
from slopometry.core.protocol.schema import EventEnvelope


@pytest.fixture
def isolated_storage(tmp_path):
    """Isolated database for ingestion tests."""
    return EventDatabase(db_path=tmp_path / "test.db")


def _mmkr_envelope(
    session_id: str,
    event_id: str,
    kind: str,
    seq: int | None = None,
    tool_name: str | None = None,
    raw: dict | None = None,
    **tool_result_fields,
) -> EventEnvelope:
    return EventEnvelope(
        source="mmkr",
        session_id=session_id,
        event_id=event_id,
        kind=kind,
        seq=seq,
        tool_name=tool_name,
        occurred_at=datetime(2026, 6, 30, 12, 0, 0),
        raw=raw or {},
        **tool_result_fields,
    )


class TestIngestEnvelopes:
    def test_ingest_envelopes__stores_third_party_events_with_canonical_kinds(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope("s-1", "evt-1", "tool_call", seq=1, tool_name="check_issue_responses", raw={"tick": 31}),
            _mmkr_envelope("s-1", "evt-2", "tool_result", seq=2, tool_name="check_issue_responses", raw={"ok": True}),
            _mmkr_envelope("s-1", "evt-3", "stop", seq=3),
        ]

        report = ingest_envelopes(envelopes, db=db, working_directory="/repo")

        assert report.inserted == 3
        events = db.get_session_events("s-1")
        assert [e.event_type for e in events] == [EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.STOP]
        assert all(e.source == "mmkr" for e in events)
        assert [e.sequence_number for e in events] == [1, 2, 3]
        assert events[0].tool_name == "check_issue_responses"
        assert events[0].metadata == {"tick": 31}
        assert all(e.id is not None for e in events)

    def test_ingest_envelopes__stores_tool_result_timing_and_error_fields(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope(
                "s-1",
                "evt-1",
                "tool_result",
                seq=1,
                tool_name="Bash",
                duration_ms=120,
                exit_code=1,
                error_message="command failed",
            ),
        ]

        ingest_envelopes(envelopes, db=db)

        event = db.get_session_events("s-1")[0]
        assert event.duration_ms == 120
        assert event.exit_code == 1
        assert event.error_message == "command failed"

    def test_ingest_envelopes__assigns_sequence_numbers_when_envelope_lacks_seq(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope("s-2", "evt-1", "tool_call"),
            _mmkr_envelope("s-2", "evt-2", "tool_result"),
        ]

        ingest_envelopes(envelopes, db=db)

        events = db.get_session_events("s-2")
        assert [e.sequence_number for e in events] == [1, 2]

    def test_ingest_envelopes__skips_duplicates_when_backfilling_identical_trace(self, isolated_storage):
        db = isolated_storage
        envelopes = [_mmkr_envelope("s-3", "evt-1", "stop", seq=1)]

        first = ingest_envelopes(envelopes, db=db)
        second = ingest_envelopes(envelopes, db=db)

        assert (first.inserted, first.skipped_duplicates) == (1, 0)
        assert (second.inserted, second.skipped_duplicates) == (0, 1)
        assert len(db.get_session_events("s-3")) == 1

    def test_ingest_envelopes__interleaves_source_and_assigned_sequences_without_collision(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope("s-4", "evt-1", "tool_call", seq=1),
            _mmkr_envelope("s-4", "evt-2", "tool_result"),
            _mmkr_envelope("s-4", "evt-3", "stop", seq=7),
            _mmkr_envelope("s-4", "evt-4", "notification"),
        ]

        ingest_envelopes(envelopes, db=db)

        events = db.get_session_events("s-4")
        assert [e.sequence_number for e in sorted(events, key=lambda e: e.event_id)] == [1, 2, 7, 8]

    def test_ingest_envelopes__continues_after_highest_stored_sequence_for_existing_session(self, isolated_storage):
        db = isolated_storage
        ingest_envelopes([_mmkr_envelope("s-5", "evt-1", "stop", seq=5)], db=db)

        ingest_envelopes([_mmkr_envelope("s-5", "evt-2", "notification")], db=db)

        events = db.get_session_events("s-5")
        assert [e.sequence_number for e in events] == [5, 6]


class TestIngestCli:
    def test_ingest__ingests_jsonl_from_stdin_for_third_party_source(self, tmp_path):
        jsonl = (
            '{"source": "mmkr", "session_id": "s-cli", "event_id": "e1", "kind": "tool_call", "seq": 1}\n'
            '{"source": "mmkr", "session_id": "s-cli", "event_id": "e2", "kind": "stop", "seq": 2}\n'
        )
        db_path = tmp_path / "test.db"

        with patch("slopometry.core.settings.settings.database_path", db_path):
            result = CliRunner().invoke(cli, ["ingest", "--source", "mmkr"], input=jsonl)

        assert result.exit_code == 0
        assert "Ingested 2 events" in result.output
        events = EventDatabase(db_path=db_path).get_session_events("s-cli")
        assert [e.event_type for e in events] == [EventKind.TOOL_CALL, EventKind.STOP]

    def test_ingest__fails_loudly_when_envelope_source_mismatches_flag(self, tmp_path):
        jsonl = '{"source": "mmkr", "session_id": "s-cli", "event_id": "e1", "kind": "stop"}\n'

        with patch("slopometry.core.settings.settings.database_path", tmp_path / "test.db"):
            result = CliRunner().invoke(cli, ["ingest", "--source", "other-agent"], input=jsonl)

        assert result.exit_code == 2
        assert "does not match --source" in result.output

    def test_ingest__fails_loudly_when_envelope_is_invalid(self, tmp_path):
        jsonl = '{"source": "mmkr", "session_id": "s-cli", "kind": "tick_complete"}\n'

        with patch("slopometry.core.settings.settings.database_path", tmp_path / "test.db"):
            result = CliRunner().invoke(cli, ["ingest", "--source", "mmkr"], input=jsonl)

        assert result.exit_code == 2
        assert "Invalid envelope at line 1" in result.output

    def test_ingest__records_explicit_working_directory_on_envelope_events(self, tmp_path):
        jsonl = '{"source": "mmkr", "session_id": "s-wd", "event_id": "e1", "kind": "stop", "seq": 1}\n'
        db_path = tmp_path / "test.db"

        with patch("slopometry.core.settings.settings.database_path", db_path):
            result = CliRunner().invoke(
                cli, ["ingest", "--source", "mmkr", "--working-directory", str(tmp_path)], input=jsonl
            )

        assert result.exit_code == 0
        events = EventDatabase(db_path=db_path).get_session_events("s-wd")
        assert events[0].working_directory == str(tmp_path)

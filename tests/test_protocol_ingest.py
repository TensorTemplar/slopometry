"""Tests for envelope ingestion into slopometry storage."""

from datetime import datetime
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from slopometry.cli import cli
from slopometry.core.database import EventDatabase
from slopometry.core.models.protocol.events import AbstractEventType
from slopometry.core.protocol.ingest import ingest_envelopes
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
            _mmkr_envelope("s-1", "evt-1", "tool_call_started", seq=1, tool_name="check_issue_responses", raw={"tick": 31}),
            _mmkr_envelope(
                "s-1", "evt-2", "tool_call_completed", seq=2, tool_name="check_issue_responses", raw={"ok": True}
            ),
            _mmkr_envelope("s-1", "evt-3", "turn_completed", seq=3),
        ]

        report = ingest_envelopes(envelopes, db=db, working_directory="/repo")

        assert report.inserted == 3
        events = db.get_session_events("s-1")
        assert [e.event_type for e in events] == [
            AbstractEventType.TOOL_CALL_STARTED,
            AbstractEventType.TOOL_CALL_COMPLETED,
            AbstractEventType.TURN_COMPLETED,
        ]
        assert all(e.source == "mmkr" for e in events)
        assert [e.sequence_number for e in events] == [1, 2, 3]
        assert events[0].tool_call is not None
        assert events[0].tool_call.tool_name == "check_issue_responses"
        assert events[0].metadata == {"tick": 31}
        assert all(e.event_id is not None for e in events)
        assert all(e.id is not None for e in events)

    def test_ingest_envelopes__stores_tool_result_timing_and_error_fields(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope(
                "s-1",
                "evt-1",
                "tool_call_completed",
                seq=1,
                tool_name="Bash",
                duration_ms=120,
                exit_code=1,
                error_message="command failed",
            ),
        ]

        ingest_envelopes(envelopes, db=db)

        event = db.get_session_events("s-1")[0]
        assert event.tool_call is not None
        assert event.tool_call.duration_ms == 120
        assert event.tool_call.exit_code == 1
        assert event.tool_call.error_message == "command failed"

    def test_ingest_envelopes__preserves_open_source_strings_through_storage_round_trip(self, isolated_storage):
        db = isolated_storage
        envelopes = [_mmkr_envelope("s-7", "evt-1", "turn_completed", seq=1)]
        envelopes[0].source = "totally-unknown-agent"

        ingest_envelopes(envelopes, db=db)
        reloaded = EventDatabase(db_path=db.db_path).get_session_events("s-7")

        assert [e.source for e in reloaded] == ["totally-unknown-agent"]

    def test_ingest_envelopes__assigns_sequence_numbers_when_envelope_lacks_seq(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope("s-2", "evt-1", "tool_call_started"),
            _mmkr_envelope("s-2", "evt-2", "tool_call_completed"),
        ]

        ingest_envelopes(envelopes, db=db)

        events = db.get_session_events("s-2")
        assert [e.sequence_number for e in events] == [1, 2]

    def test_ingest_envelopes__interleaves_source_and_assigned_sequences_without_collision(self, isolated_storage):
        db = isolated_storage
        envelopes = [
            _mmkr_envelope("s-4", "evt-1", "tool_call_started", seq=1),
            _mmkr_envelope("s-4", "evt-2", "tool_call_completed"),
            _mmkr_envelope("s-4", "evt-3", "turn_completed", seq=7),
            _mmkr_envelope("s-4", "evt-4", "notification"),
        ]

        ingest_envelopes(envelopes, db=db)

        events = sorted(db.get_session_events("s-4"), key=lambda e: e.event_id)
        assert [e.sequence_number for e in events] == [1, 2, 7, 8]

    def test_ingest_envelopes__continues_after_highest_stored_sequence_for_existing_session(self, isolated_storage):
        db = isolated_storage
        ingest_envelopes([_mmkr_envelope("s-5", "evt-1", "turn_completed", seq=5)], db=db)

        ingest_envelopes([_mmkr_envelope("s-5", "evt-2", "notification")], db=db)

        events = db.get_session_events("s-5")
        assert [e.sequence_number for e in events] == [5, 6]

    def test_ingest_envelopes__skips_duplicates_when_backfilling_identical_trace(self, isolated_storage):
        db = isolated_storage
        envelopes = [_mmkr_envelope("s-3", "evt-1", "turn_completed", seq=1)]

        first = ingest_envelopes(envelopes, db=db)
        second = ingest_envelopes(envelopes, db=db)

        assert (first.inserted, first.skipped_duplicates) == (1, 0)
        assert (second.inserted, second.skipped_duplicates) == (0, 1)
        assert len(db.get_session_events("s-3")) == 1


class TestIngestCli:
    def test_ingest__ingests_jsonl_from_stdin_for_third_party_source(self, tmp_path):
        jsonl = (
            '{"source": "mmkr", "session_id": "s-cli", "event_id": "e1", "kind": "tool_call_started", "seq": 1}\n'
            '{"source": "mmkr", "session_id": "s-cli", "event_id": "e2", "kind": "turn_completed", "seq": 2}\n'
        )
        db_path = tmp_path / "test.db"

        with patch("slopometry.core.settings.settings.database_path", db_path):
            result = CliRunner().invoke(cli, ["ingest", "--source", "mmkr"], input=jsonl)

        assert result.exit_code == 0
        assert "Ingested 2 events" in result.output
        events = EventDatabase(db_path=db_path).get_session_events("s-cli")
        assert [e.event_type for e in events] == [AbstractEventType.TOOL_CALL_STARTED, AbstractEventType.TURN_COMPLETED]

    def test_ingest__reads_from_file_when_flag_given(self, tmp_path):
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text('{"source": "mmkr", "session_id": "s-file", "event_id": "e1", "kind": "notification"}\n')
        db_path = tmp_path / "test.db"

        with patch("slopometry.core.settings.settings.database_path", db_path):
            result = CliRunner().invoke(
                cli, ["ingest", "--source", "mmkr", "--file", str(trace_file), "--working-directory", str(tmp_path)]
            )

        assert result.exit_code == 0
        events = EventDatabase(db_path=db_path).get_session_events("s-file")
        assert [e.event_type for e in events] == [AbstractEventType.NOTIFICATION]
        assert events[0].working_directory == str(tmp_path)

    def test_ingest__fails_loudly_when_envelope_source_mismatches_flag(self, tmp_path):
        jsonl = '{"source": "mmkr", "session_id": "s-cli", "event_id": "e1", "kind": "turn_completed"}\n'

        with patch("slopometry.core.settings.settings.database_path", tmp_path / "test.db"):
            result = CliRunner().invoke(cli, ["ingest", "--source", "other-agent"], input=jsonl)

        assert result.exit_code == 2
        assert "does not match --source" in result.output

    def test_ingest__fails_loudly_when_envelope_is_invalid(self, tmp_path):
        jsonl = '{"source": "mmkr", "session_id": "s-cli", "event_id": "e1", "kind": "tick_complete"}\n'

        with patch("slopometry.core.settings.settings.database_path", tmp_path / "test.db"):
            result = CliRunner().invoke(cli, ["ingest", "--source", "mmkr"], input=jsonl)

        assert result.exit_code == 2
        assert "Invalid envelope at line 1" in result.output

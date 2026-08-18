"""Tests for the external envelope schema used by `slopometry ingest`."""

import pytest
from pydantic import ValidationError

from slopometry.core.models.protocol.events import AbstractEventType
from slopometry.core.protocol.schema import EventEnvelope


class TestEventEnvelopeValidation:
    def test_event_envelope__accepts_minimal_envelope_with_defaults(self):
        envelope = EventEnvelope(source="mmkr", session_id="s1", event_id="e1", kind="tool_call_started")

        assert envelope.kind is AbstractEventType.TOOL_CALL_STARTED
        assert envelope.schema_version == 1
        assert envelope.seq is None
        assert envelope.duration_ms is None
        assert envelope.exit_code is None
        assert envelope.error_message is None
        assert envelope.raw == {}

    def test_event_envelope__accepts_arbitrary_third_party_source_strings(self):
        envelope = EventEnvelope(source="my-custom-agent", session_id="s1", event_id="e1", kind="notification")

        assert envelope.source == "my-custom-agent"

    def test_event_envelope__requires_event_id_for_idempotent_backfill(self):
        with pytest.raises(ValidationError):
            EventEnvelope(source="mmkr", session_id="s1", kind="tool_call_started")

    def test_event_envelope__rejects_unsupported_schema_version(self):
        with pytest.raises(ValidationError, match="Unsupported schema_version"):
            EventEnvelope(source="mmkr", session_id="s1", event_id="e1", kind="tool_call_started", schema_version=2)

    def test_event_envelope__rejects_unknown_kind_with_helpful_message(self):
        with pytest.raises(ValidationError) as exc_info:
            EventEnvelope(source="mmkr", session_id="s1", event_id="e1", kind="tick_complete")

        assert "tool_call_started" in str(exc_info.value)

    def test_event_envelope__rejects_empty_source(self):
        with pytest.raises(ValidationError):
            EventEnvelope(source="", session_id="s1", event_id="e1", kind="tool_call_started")

    def test_event_envelope__rejects_sequence_below_one(self):
        with pytest.raises(ValidationError):
            EventEnvelope(source="mmkr", session_id="s1", event_id="e1", kind="tool_call_started", seq=0)

    def test_event_envelope__preserves_unknown_fields_for_forward_compatibility(self):
        envelope = EventEnvelope(
            source="mmkr", session_id="s1", event_id="e1", kind="turn_completed", harness_trace_url="https://example.com"
        )

        assert envelope.model_extra == {"harness_trace_url": "https://example.com"}

    def test_event_envelope__round_trips_through_jsonl_line(self):
        payload = (
            '{"source": "mmkr", "schema_version": 1, "session_id": "s-42", "event_id": "tick-31",'
            ' "parent_session_id": "s-1", "seq": 31, "occurred_at": "2026-06-30T12:00:00",'
            ' "kind": "tool_call_started", "tool_name": "check_issue_responses", "raw": {"tick": 31}}'
        )

        envelope = EventEnvelope.model_validate_json(payload)

        assert envelope.parent_session_id == "s-1"
        assert envelope.tool_name == "check_issue_responses"
        assert envelope.seq == 31
        assert envelope.raw == {"tick": 31}

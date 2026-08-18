"""Idempotent ingestion of `EventEnvelope` batches into slopometry storage.

This is the third-party counterpart to `protocol/dispatch.py`: dispatch
runs a wire payload through a registered adapter (live harness paths), while
ingest accepts already-canonical envelopes from collectors without adapters.
Session context (git state, project detection) is deliberately not captured:
the repository the ingester runs in is unrelated to the traced session.
"""

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from slopometry.core.database import EventDatabase
from slopometry.core.models.protocol.events import AbstractHookEvent, ToolCallPayload
from slopometry.core.protocol.schema import EventEnvelope


class IngestReport(BaseModel):
    """Outcome of ingesting a batch of envelopes."""

    inserted: int = 0
    skipped_duplicates: int = 0


def envelope_to_event(envelope: EventEnvelope, working_directory: str, sequence_number: int) -> AbstractHookEvent:
    """Expand a validated envelope into the canonical stored event model."""
    tool_call = (
        ToolCallPayload(
            tool_name=envelope.tool_name,
            input=envelope.raw,
            duration_ms=envelope.duration_ms,
            exit_code=envelope.exit_code,
            error_message=envelope.error_message,
        )
        if envelope.tool_name
        else None
    )
    return AbstractHookEvent(
        session_id=envelope.session_id,
        parent_session_id=envelope.parent_session_id,
        event_type=envelope.kind,
        source=envelope.source,
        timestamp=envelope.occurred_at,
        tool_call=tool_call,
        metadata=envelope.raw,
        working_directory=working_directory,
        sequence_number=sequence_number,
        event_id=envelope.event_id,
    )


def ingest_envelopes(
    envelopes: list[EventEnvelope],
    db: EventDatabase | None = None,
    working_directory: str | None = None,
) -> IngestReport:
    """Store envelopes, skipping duplicates identified by (source, event_id).

    Sequence numbers come from the envelope when provided; otherwise a
    per-session counter continuing after the highest sequence number already
    used (stored or earlier in the batch) is assigned. The live SessionManager
    state files are not touched: backfills derive ordering from storage.
    """
    db = db or EventDatabase()
    working_directory = working_directory or str(Path.cwd())
    report = IngestReport()
    next_sequence: dict[str, int] = {}

    for envelope in envelopes:
        if db.has_event(envelope.source, envelope.event_id):
            report.skipped_duplicates += 1
            continue

        if envelope.seq is not None:
            sequence_number = envelope.seq
            next_sequence[envelope.session_id] = max(
                envelope.seq + 1,
                next_sequence.get(envelope.session_id, 1 + db.get_max_sequence_number(envelope.session_id)),
            )
        else:
            next_sequence.setdefault(envelope.session_id, 1 + db.get_max_sequence_number(envelope.session_id))
            sequence_number = next_sequence[envelope.session_id]
            next_sequence[envelope.session_id] = sequence_number + 1

        try:
            event = envelope_to_event(envelope, working_directory, sequence_number)
            event.id = db.save_event(event)
        except sqlite3.IntegrityError:
            # Concurrent ingest of the same (source, event_id); the unique
            # partial index makes this safe to treat as a duplicate.
            report.skipped_duplicates += 1
            continue
        report.inserted += 1

    return report

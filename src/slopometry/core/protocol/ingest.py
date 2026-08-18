"""Ingestion of `EventEnvelope` batches into slopometry storage.

This is the shared path used by `slopometry ingest` for third-party agent
traces. Live harness adapters may bypass it because they perform additional
enrichment (git state, transcript analysis) tied to the running session.
"""

import os
import sqlite3

from pydantic import BaseModel

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.event_capture import capture_event
from slopometry.core.protocol.schema import EventEnvelope


class IngestReport(BaseModel):
    """Outcome of ingesting a batch of envelopes."""

    inserted: int = 0
    skipped_duplicates: int = 0


def ingest_envelopes(
    envelopes: list[EventEnvelope],
    db: EventDatabase | None = None,
    working_directory: str | None = None,
) -> IngestReport:
    """Store envelopes, skipping duplicates identified by (source, event_id).

    Sequence numbers come from the envelope when provided; otherwise a
    per-session counter continuing after the highest sequence number already
    used (stored or earlier in the batch) is assigned.
    """
    db = db or EventDatabase()
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
            capture_event(
                db,
                SessionManager(),
                working_directory or os.getcwd(),
                session_id=envelope.session_id,
                kind=envelope.kind,
                source=envelope.source,
                metadata=envelope.raw,
                sequence_number=sequence_number,
                timestamp=envelope.occurred_at,
                tool_name=envelope.tool_name,
                duration_ms=envelope.duration_ms,
                exit_code=envelope.exit_code,
                error_message=envelope.error_message,
                parent_session_id=envelope.parent_session_id,
                event_id=envelope.event_id,
                capture_session_context=False,
            )
        except sqlite3.IntegrityError:
            # Concurrent ingest of the same (source, event_id); the unique
            # partial index makes this safe to treat as a duplicate.
            report.skipped_duplicates += 1
            continue
        report.inserted += 1

    return report

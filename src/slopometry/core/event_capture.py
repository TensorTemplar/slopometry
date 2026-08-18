"""Shared event capture factory for all event ingestion paths.

Owns the assembly of a stored `HookEvent`: sequence numbering, git state,
project detection, tool classification, and persistence. Used by the Claude
and OpenCode live handlers and by the envelope ingest path, so no fourth
construction site for stored events should appear.
"""

from datetime import datetime
from pathlib import Path

from slopometry.core.database import EventDatabase, SessionManager
from slopometry.core.git_tracker import GitTracker
from slopometry.core.models.hook import HookEvent, get_tool_type
from slopometry.core.project_tracker import ProjectTracker
from slopometry.core.protocol.kinds import EventKind


def capture_event(
    db: EventDatabase,
    session_manager: SessionManager,
    working_directory: str,
    *,
    session_id: str,
    kind: EventKind,
    source: str,
    metadata: dict,
    sequence_number: int | None = None,
    timestamp: datetime | None = None,
    tool_name: str | None = None,
    duration_ms: int | None = None,
    exit_code: int | None = None,
    error_message: str | None = None,
    transcript_path: str | None = None,
    parent_session_id: str | None = None,
    event_id: str | None = None,
    capture_session_context: bool = True,
) -> HookEvent:
    """Build and persist one stored event for a session.

    When `capture_session_context` is True (live harness paths), git state is
    recorded on the session's first event and on stop events, and the project
    is detected from the working directory. Envelope ingestion passes False
    because the ingester's repository is unrelated to the traced session.
    """
    sequence_number = sequence_number or session_manager.get_next_sequence_number(session_id)

    git_state = None
    project = None
    if capture_session_context:
        if sequence_number == 1 or kind in (EventKind.STOP, EventKind.SUBAGENT_STOP):
            git_state = GitTracker().get_git_state()
        project = ProjectTracker(working_dir=Path(working_directory)).get_project()

    event = HookEvent(
        session_id=session_id,
        event_type=kind,
        timestamp=timestamp or datetime.now(),
        sequence_number=sequence_number,
        tool_name=tool_name,
        tool_type=get_tool_type(tool_name) if tool_name else None,
        metadata=metadata,
        duration_ms=duration_ms,
        exit_code=exit_code,
        error_message=error_message,
        git_state=git_state,
        working_directory=working_directory,
        project=project,
        transcript_path=transcript_path,
        source=source,
        parent_session_id=parent_session_id,
        event_id=event_id,
    )
    event.id = db.save_event(event)
    return event

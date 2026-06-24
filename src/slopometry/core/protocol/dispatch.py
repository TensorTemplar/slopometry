"""Generic event ingestion dispatcher.

Reads JSON from stdin, picks the right adapter by source, parses into an
AbstractHookEvent, attaches git/project context, assigns a sequence number,
and persists.

`emit_event` is the public entrypoint used by both the CLI and the harness-
specific handlers (`hook_handler.py`, `opencode_handler.py`) after they have
done their harness-specific glue.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from slopometry.core.git_tracker import GitTracker
from slopometry.core.lock import SlopometryLock
from slopometry.core.models.protocol.events import AbstractEventSource, AbstractEventType, AbstractHookEvent
from slopometry.core.project_tracker import ProjectTracker
from slopometry.core.protocol.adapters.base import ADAPTERS
from slopometry.core.protocol.session import SessionManager

logger = logging.getLogger(__name__)


def _capture_git_state(event_type: AbstractEventType, sequence_number: int):
    tracker = GitTracker()
    match (event_type, sequence_number):
        case (AbstractEventType.TOOL_CALL_STARTED, 1) | (AbstractEventType.TURN_COMPLETED, _):
            return tracker.get_git_state()
        case _:
            return None


def _capture_project(working_directory: str):
    return ProjectTracker(working_dir=Path(working_directory)).get_project()


def dispatch_event(
    source: AbstractEventSource,
    raw_payload: dict,
    *,
    working_directory: str | None = None,
    timestamp: datetime | None = None,
    event_type_override: AbstractEventType | None = None,
) -> AbstractHookEvent:
    """Parse a wire payload through the adapter for `source`, enrich, persist.

    Args:
        source: Which harness produced this payload.
        raw_payload: The JSON the harness sent (already parsed).
        working_directory: Override cwd; defaults to os.getcwd().
        timestamp: Override event timestamp; defaults to now().
        event_type_override: Skip adapter detection; force a specific event type.

    Returns:
        The persisted AbstractHookEvent (with assigned id and sequence_number).

    Raises:
        ValueError: If `source` has no registered adapter, or if the adapter
            rejects the payload.
    """
    adapter = ADAPTERS.get(source)
    if adapter is None:
        raise ValueError(f"No adapter registered for source {source.value!r}")

    cwd = working_directory or os.getcwd()
    event = adapter.parse(
        raw_payload,
        working_directory=cwd,
        timestamp=timestamp,
        event_type_override=event_type_override,
    )

    session_manager = SessionManager(source=source.value)
    event.sequence_number = session_manager.get_next_sequence_number(event.session_id)
    event.git_state = _capture_git_state(event.event_type, event.sequence_number)
    event.project = _capture_project(event.working_directory)

    lock = SlopometryLock(project_dir=cwd)
    with lock.acquire() as acquired:
        if not acquired:
            logger.debug("Could not acquire lock, skipping event persistence for %s", event.session_id)
            return event

        from slopometry.core.database import EventDatabase

        EventDatabase().save_event(event)

    return event


def emit_event_from_stdin(
    source: AbstractEventSource,
    event_type_override: AbstractEventType | None = None,
) -> int:
    """Read JSON from stdin, dispatch, return process exit code.

    Used by the `slopometry emit-event` CLI subcommand and by harness-specific
    entry points after they have read their stdin.
    """
    import sys

    try:
        stdin_input = sys.stdin.read().strip()
    except Exception:
        return 0
    if not stdin_input:
        return 0

    try:
        raw_payload = json.loads(stdin_input)
    except json.JSONDecodeError as e:
        from slopometry.core.settings import settings

        if settings.debug_mode:
            print(f"Slopometry: Failed to parse event JSON: {e}", file=sys.stderr)
        return 0

    dispatch_event(source, raw_payload, event_type_override=event_type_override)
    return 0

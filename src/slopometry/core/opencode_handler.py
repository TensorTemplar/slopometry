"""OpenCode hook handler — receives OpenCode plugin JSON, delegates to the
OpenCode adapter for parsing, persists via the abstract protocol, and runs
the Claude-Code stop-hook feedback pipeline (reused because OpenCode and
Claude Code produce equivalent working-tree statistics).

This module is the OpenCode-specific glue. Harness-agnostic types live in
`core.protocol.events`; the wire-format parser is in
`core.protocol.adapters.opencode`.
"""

import json
import logging
import sys

from slopometry.core.hook_handler import handle_stop_event
from slopometry.core.models.protocol.events import AbstractEventSource, AbstractEventType
from slopometry.core.protocol.dispatch import dispatch_event
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


def _read_stdin_with_timeout(timeout_seconds: float = 5.0) -> str:
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        return ""
    return sys.stdin.read().strip()


def _resolve_event_type(event_type: str) -> AbstractEventType:
    """Translate OpenCode's CLI event-type string into the canonical enum."""
    from slopometry.core.protocol.adapters.opencode import resolve_opencode_event_type

    return resolve_opencode_event_type(event_type)


def handle_opencode_hook(event_type: str) -> int:
    """Main entry point for OpenCode plugin event invocations.

    Called from CLI: `slopometry hook-opencode --event-type <type>`.

    Args:
        event_type: OpenCode's event-type discriminator string.

    Returns:
        Exit code (0 for success, 2 for blocking feedback).
    """
    try:
        stdin_input = _read_stdin_with_timeout()
    except Exception:
        return 0
    if not stdin_input:
        return 0

    try:
        raw_payload = json.loads(stdin_input)
    except json.JSONDecodeError as e:
        if settings.debug_mode:
            print(f"Slopometry: Failed to parse OpenCode event: {e}", file=sys.stderr)
        return 0

    try:
        abstract_type = _resolve_event_type(event_type)
        event = dispatch_event(
            AbstractEventSource.OPENCODE,
            raw_payload,
            event_type_override=abstract_type,
        )
    except Exception as e:
        if settings.debug_mode:
            print(f"Slopometry OpenCode hook error: {e}", file=sys.stderr)
        return 0

    if (
        settings.enable_complexity_analysis
        and event.event_type in (AbstractEventType.TURN_COMPLETED, AbstractEventType.SUBAGENT_COMPLETED)
    ):
        return handle_stop_event(event.session_id, event.working_directory)

    if settings.debug_mode:
        debug_info = {
            "slopometry_opencode_event": {
                "session_id": event.session_id,
                "event_type": event_type,
                "abstract_type": event.event_type.value,
                "sequence_number": event.sequence_number,
                "source": "opencode",
            }
        }
        print(f"Slopometry captured: {json.dumps(debug_info, indent=2)}", file=sys.stderr)

    return 0

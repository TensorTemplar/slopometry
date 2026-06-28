"""Per-session sequence numbering, abstracted from Claude-Code's ~/.claude layout.

Each AbstractEventSource has its own state directory so concurrent harnesses
don't collide. Legacy state at ~/.claude/slopometry/seq_*.txt is relocated on
first access.
"""

import logging
from pathlib import Path

from slopometry.core.models.protocol.events import AbstractEventSource

logger = logging.getLogger(__name__)

_LEGACY_STATE_DIR = Path.home() / ".claude" / "slopometry"
_DEFAULT_STATE_ROOT = Path.home() / ".slopometry" / "sessions"


class SessionManager:
    """Assigns monotonic sequence numbers per (source, session_id) pair.

    State lives at `<state_root>/<source>/seq_<session_id>.txt` by default.
    The legacy path `~/.claude/slopometry/seq_<session_id>.txt` is migrated
    to `~/.slopometry/sessions/claude_code/seq_<session_id>.txt` on first
    construction.
    """

    def __init__(
        self,
        source: str,
        state_root: Path | None = None,
    ) -> None:
        self.source = source
        self.state_root = state_root or _DEFAULT_STATE_ROOT
        self.state_dir = self.state_root / source
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_files()

    def _migrate_legacy_files(self) -> None:
        if self.source != AbstractEventSource.CLAUDE_CODE.value:
            return
        if not _LEGACY_STATE_DIR.exists():
            return
        for seq_file in _LEGACY_STATE_DIR.glob("seq_*.txt"):
            target = self.state_dir / seq_file.name
            if target.exists():
                continue
            try:
                seq_file.rename(target)
                logger.debug("Migrated session seq file %s -> %s", seq_file, target)
            except OSError as e:
                logger.debug("Could not migrate %s: %s", seq_file, e)

    def get_next_sequence_number(self, session_id: str) -> int:
        seq_file = self.state_dir / f"seq_{session_id}.txt"
        if seq_file.exists():
            try:
                current_seq = int(seq_file.read_text().strip())
                next_seq = current_seq + 1
            except (ValueError, FileNotFoundError) as e:
                logger.debug("Corrupt sequence file for %s, resetting: %s", session_id, e)
                next_seq = 1
        else:
            next_seq = 1
        seq_file.write_text(str(next_seq))
        return next_seq

"""Singleton Rich Console instance for consistent output across the application."""

import logging
import shutil
import subprocess
import sys

from rich.console import Console
from rich.pager import Pager

logger = logging.getLogger(__name__)


def _show_with_less(content: str) -> None:
    """Page content through less -R, falling back to direct stdout on failure.

    Uses -R so less correctly treats ANSI color codes as zero-width.
    Without it, less counts escape-code bytes as visual width, which in
    narrow terminals makes it miscalculate line positions and show (END)
    before all content is reachable.
    """
    terminal_height = shutil.get_terminal_size().lines
    content_lines = content.count("\n")
    if content_lines <= terminal_height - 1:
        sys.stdout.write(content)
        sys.stdout.flush()
        return

    try:
        proc = subprocess.Popen(
            ["less", "-R"],
            stdin=subprocess.PIPE,
            errors="backslashreplace",
        )
        pipe = proc.stdin
        assert pipe is not None
        try:
            with pipe:
                try:
                    pipe.write(content)
                except KeyboardInterrupt:
                    pass
        except OSError:
            # Broken pipe: user quit less before all content was written
            logger.debug("Broken pipe writing to less (user quit early)")
        while True:
            try:
                proc.wait()
                break
            except KeyboardInterrupt:
                pass
    except FileNotFoundError:
        logger.debug("less not found, writing directly to stdout")
        sys.stdout.write(content)
        sys.stdout.flush()


class _LessPager(Pager):
    """Adapter to satisfy Rich's Pager protocol."""

    def show(self, content: str) -> None:
        _show_with_less(content)


styled_pager = _LessPager()

# Single console instance used throughout the application
# This ensures pager context works correctly across modules
console = Console()

"""Compact event analyzer for extracting compact events from Claude Code transcripts."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from slopometry.core.models import CompactEvent

if TYPE_CHECKING:
    from slopometry.core.database import EventDatabase

logger = logging.getLogger(__name__)


class CompactBoundary(BaseModel, extra="allow"):
    """Parsed compact_boundary event from transcript."""

    type: str | None = None
    subtype: str | None = None
    content: str | None = None
    timestamp: str | None = None
    uuid: str | None = None
    compactMetadata: dict | None = None
    version: str | None = None
    gitBranch: str | None = None


class CompactSummary(BaseModel, extra="allow"):
    """Parsed isCompactSummary event from transcript."""

    type: str | None = None
    parentUuid: str | None = None
    isCompactSummary: bool | None = None
    message: dict | None = None
    timestamp: str | None = None


class CompactEventAnalyzer:
    """Analyzes Claude Code transcripts to extract compact events."""

    def analyze_transcript(self, transcript_path: Path) -> list[CompactEvent]:
        """Parse transcript JSONL and extract compact events.

        Compact events consist of:
        1. A boundary line with type="system", subtype="compact_boundary"
        2. A summary line with isCompactSummary=true linked via parentUuid

        Args:
            transcript_path: Path to the JSONL transcript file

        Returns:
            List of CompactEvent objects found in the transcript
        """
        compact_events: list[CompactEvent] = []
        pending_boundaries: dict[str, tuple[int, CompactBoundary]] = {}

        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    try:
                        raw_event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if self._is_compact_boundary(raw_event):
                        boundary = CompactBoundary.model_validate(raw_event)
                        if boundary.uuid:
                            pending_boundaries[boundary.uuid] = (line_number, boundary)

                    elif self._is_compact_summary(raw_event):
                        summary = CompactSummary.model_validate(raw_event)
                        parent_uuid = summary.parentUuid

                        if parent_uuid and parent_uuid in pending_boundaries:
                            line_num, boundary = pending_boundaries.pop(parent_uuid)
                            compact_event = self._create_compact_event(line_num, boundary, summary)
                            if compact_event:
                                compact_events.append(compact_event)

        except OSError as e:
            logger.warning(f"Failed to read transcript file {transcript_path}: {e}")

        return compact_events

    def _is_compact_boundary(self, raw_event: dict) -> bool:
        """Check if event is a compact_boundary system event."""
        return raw_event.get("type") == "system" and raw_event.get("subtype") == "compact_boundary"

    def _is_compact_summary(self, raw_event: dict) -> bool:
        """Check if event is a compact summary (isCompactSummary=true)."""
        return raw_event.get("isCompactSummary") is True

    def _create_compact_event(
        self, line_number: int, boundary: CompactBoundary, summary: CompactSummary
    ) -> CompactEvent | None:
        """Create a CompactEvent from boundary and summary data."""
        metadata = boundary.compactMetadata or {}
        trigger = metadata.get("trigger", "unknown")
        pre_tokens = metadata.get("preTokens", 0)

        summary_content = ""
        if summary.message:
            content = summary.message.get("content", "")
            if isinstance(content, str):
                summary_content = content

        timestamp_str = boundary.timestamp or summary.timestamp
        if not timestamp_str:
            logger.warning(f"Compact event at line {line_number} missing timestamp, skipping")
            return None

        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Compact event at line {line_number} has invalid timestamp '{timestamp_str}', skipping")
            return None

        return CompactEvent(
            line_number=line_number,
            trigger=trigger,
            pre_tokens=pre_tokens,
            summary_content=summary_content,
            timestamp=timestamp,
            uuid=boundary.uuid or "",
            version=boundary.version or "n/a",
            git_branch=boundary.gitBranch or "n/a",
        )


def discover_transcripts(working_directory: Path, db: "EventDatabase") -> list[Path]:
    """Find all transcripts relevant to the given project.

    Sources:
    1. Database: Query sessions with matching working_directory
    2. Claude Code default: ~/.claude/transcripts/*.jsonl

    Args:
        working_directory: Project directory to filter by
        db: EventDatabase instance for querying sessions

    Returns:
        List of unique transcript paths
    """

    transcripts: set[Path] = set()
    normalized_wd = working_directory.resolve()

    sessions = db.list_sessions_by_repository(working_directory)
    for session_id in sessions:
        stats = db.get_session_statistics(session_id)
        if stats and stats.transcript_path:
            path = Path(stats.transcript_path)
            if path.exists():
                transcripts.add(path)

    claude_transcripts_dir = Path.home() / ".claude" / "transcripts"
    if claude_transcripts_dir.exists():
        for transcript in claude_transcripts_dir.glob("**/*.jsonl"):
            if _transcript_matches_project(transcript, normalized_wd):
                transcripts.add(transcript)

    return list(transcripts)


def _transcript_matches_project(transcript_path: Path, working_directory: Path) -> bool:
    """Check if transcript's cwd matches the target working directory."""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                return False
            data = json.loads(first_line)
            cwd = data.get("cwd", "")
            if not cwd:
                return False
            return Path(cwd).resolve() == working_directory
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read transcript {transcript_path} for project matching: {e}")
        return False


def find_compact_instructions(transcript_path: Path, compact_line_number: int, lookback_lines: int = 50) -> str | None:
    """Search backwards for /compact command that triggered this compact.

    Args:
        transcript_path: Path to the transcript file
        compact_line_number: Line number of the compact_boundary event
        lookback_lines: How many lines to search backwards

    Returns:
        The user's compact instructions if found, None otherwise
    """
    try:
        with open(transcript_path, encoding="utf-8") as f:
            all_lines = f.readlines()

        start = max(0, compact_line_number - lookback_lines - 1)
        end = compact_line_number - 1
        lines_to_search = all_lines[start:end]

        for line in reversed(lines_to_search):
            try:
                data = json.loads(line)
                if data.get("type") != "user":
                    continue

                message = data.get("message", {})
                content = message.get("content", "")

                if isinstance(content, str) and "/compact" in content.lower():
                    return content
            except json.JSONDecodeError:
                continue

    except OSError as e:
        logger.warning(f"Failed to read transcript {transcript_path} for compact instructions: {e}")

    return None


def analyze_transcript_compacts(transcript_path: Path) -> list[CompactEvent]:
    """Convenience function to analyze compact events from a transcript.

    Args:
        transcript_path: Path to Claude Code transcript JSONL

    Returns:
        List of CompactEvent objects
    """
    analyzer = CompactEventAnalyzer()
    return analyzer.analyze_transcript(transcript_path)

"""Behavioral pattern analyzer for detecting ownership dodging and simple workarounds in transcripts."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slopometry.core.models.session import (
    BehavioralPatternCategory,
    BehavioralPatterns,
    PatternMatch,
)

logger = logging.getLogger(__name__)

OWNERSHIP_DODGING_PHRASES: list[str] = [
    "not caused by my changes",
    "not introduced by",
    "pre-existing",
    "existing issue",
    "existing bug",
    "unrelated to",
    "not related to",
    "was already",
    "separate issue",
    "different issue",
    "known issue",
    "known limitation",
]

SIMPLE_WORKAROUND_PHRASES: list[str] = [
    "simplest",
    "simple fix",
    "simple workaround",
    "quick fix",
    "quick workaround",
    "for now",
    "easiest",
    "minimal change",
    "as a workaround",
    "good enough",
    "good stopping point",
]

SNIPPET_BEFORE = 40
SNIPPET_AFTER = 80
MAX_SNIPPET_LEN = 120


def _build_pattern(phrases: list[str]) -> re.Pattern[str]:
    """Build a compiled regex that matches any of the given phrases with word boundaries."""
    escaped = [re.escape(p) for p in phrases]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


OWNERSHIP_DODGING_RE = _build_pattern(OWNERSHIP_DODGING_PHRASES)
SIMPLE_WORKAROUND_RE = _build_pattern(SIMPLE_WORKAROUND_PHRASES)


class ContentBlock(BaseModel, extra="allow"):
    """A single content block from an assistant message."""

    type: str = ""
    text: str = ""


class AssistantMessageContent(BaseModel, extra="allow"):
    """Message payload from a transcript assistant event."""

    content: list[ContentBlock] | str = Field(default="")


class TranscriptAssistantEvent(BaseModel, extra="allow"):
    """A transcript event that may be an assistant message with text content."""

    type: str | None = None
    message: AssistantMessageContent | None = None
    timestamp: str | None = None


def _extract_snippet(text: str, match_start: int, match_end: int) -> str:
    """Extract a context snippet around a match, truncated to MAX_SNIPPET_LEN."""
    start = max(0, match_start - SNIPPET_BEFORE)
    end = min(len(text), match_end + SNIPPET_AFTER)
    snippet = text[start:end].replace("\n", " ").strip()
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[:MAX_SNIPPET_LEN] + "..."
    if start > 0:
        snippet = "..." + snippet
    return snippet


def _extract_assistant_text(event: TranscriptAssistantEvent) -> str:
    """Extract visible text from an assistant message (skipping thinking and tool_use blocks)."""
    if not event.message:
        return ""

    content = event.message.content
    if isinstance(content, str):
        return content

    return "\n".join(block.text for block in content if block.type == "text" and block.text)


def _parse_timestamp(event: TranscriptAssistantEvent) -> datetime | None:
    """Parse timestamp from a transcript event."""
    if not event.timestamp:
        return None
    try:
        return datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _scan_text(
    text: str,
    pattern: re.Pattern[str],
    line_number: int,
    timestamp: datetime | None,
) -> list[PatternMatch]:
    """Scan text for all matches of a pattern, returning PatternMatch objects."""
    return [
        PatternMatch(
            pattern=m.group().lower(),
            line_number=line_number,
            context_snippet=_extract_snippet(text, m.start(), m.end()),
            timestamp=timestamp,
        )
        for m in pattern.finditer(text)
    ]


def _scan_events(
    events: list[tuple[int, TranscriptAssistantEvent]],
) -> tuple[list[PatternMatch], list[PatternMatch]]:
    """Scan parsed assistant events for both pattern categories."""
    ownership_matches: list[PatternMatch] = []
    workaround_matches: list[PatternMatch] = []

    for line_number, event in events:
        text = _extract_assistant_text(event)
        if not text:
            continue
        timestamp = _parse_timestamp(event)
        ownership_matches.extend(_scan_text(text, OWNERSHIP_DODGING_RE, line_number, timestamp))
        workaround_matches.extend(_scan_text(text, SIMPLE_WORKAROUND_RE, line_number, timestamp))

    return ownership_matches, workaround_matches


def _build_result(
    ownership_matches: list[PatternMatch],
    workaround_matches: list[PatternMatch],
    session_duration_minutes: float,
) -> BehavioralPatterns:
    """Build the BehavioralPatterns result from match lists."""
    return BehavioralPatterns(
        ownership_dodging=BehavioralPatternCategory(
            category_name="Ownership Dodging",
            matches=ownership_matches,
        ),
        simple_workaround=BehavioralPatternCategory(
            category_name="Simple Workaround",
            matches=workaround_matches,
        ),
        session_duration_minutes=session_duration_minutes,
    )


def analyze_behavioral_patterns(transcript_path: Path, session_duration_minutes: float) -> BehavioralPatterns:
    """Analyze a Claude Code transcript JSONL file for behavioral patterns.

    Scans assistant text blocks (excluding thinking and tool_use) for ownership
    dodging and simple workaround phrases. Returns match counts with per-minute rates.

    Args:
        transcript_path: Path to the JSONL transcript file
        session_duration_minutes: Session duration in minutes for rate calculation

    Returns:
        BehavioralPatterns with matches and rates
    """
    events: list[tuple[int, TranscriptAssistantEvent]] = []

    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if raw.get("type") != "assistant":
                    continue

                try:
                    event = TranscriptAssistantEvent.model_validate(raw)
                except Exception:
                    continue

                events.append((line_number, event))

    except OSError as e:
        logger.debug(f"Failed to read transcript for behavioral patterns: {e}")

    ownership_matches, workaround_matches = _scan_events(events)
    return _build_result(ownership_matches, workaround_matches, session_duration_minutes)


def analyze_opencode_behavioral_patterns(
    transcript: list[dict[str, Any]], session_duration_minutes: float
) -> BehavioralPatterns:
    """Analyze an OpenCode transcript for behavioral patterns.

    OpenCode transcripts are lists of message dicts with {role, parts, tokens}.

    Args:
        transcript: List of message dicts from the stop event metadata
        session_duration_minutes: Session duration in minutes for rate calculation

    Returns:
        BehavioralPatterns with matches and rates
    """
    ownership_matches: list[PatternMatch] = []
    workaround_matches: list[PatternMatch] = []

    for line_number, msg in enumerate(transcript, start=1):
        if msg.get("role") != "assistant":
            continue

        parts = msg.get("parts", [])
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        text = "\n".join(text_parts)

        if not text:
            continue

        ownership_matches.extend(_scan_text(text, OWNERSHIP_DODGING_RE, line_number, None))
        workaround_matches.extend(_scan_text(text, SIMPLE_WORKAROUND_RE, line_number, None))

    return _build_result(ownership_matches, workaround_matches, session_duration_minutes)

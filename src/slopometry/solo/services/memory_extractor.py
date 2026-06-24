"""Memory extraction from transcripts using LLM."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slopometry.core.models.memory import MemoryCandidate, MemoryType

logger = logging.getLogger(__name__)


class TranscriptTruncationConfig(BaseModel):
    """Per-tool-part summary truncation limits when emitting conversation text.

    These limits keep tool invocations from flooding the LLM prompt while
    preserving enough context for memory extraction to recognize what the
    tool did. Both limits are per-tool-part, not per-session.
    """

    model_config = ConfigDict(extra="forbid")

    tool_input_chars: int = Field(
        default=120,
        description="Max characters of a tool's input JSON to include in the reconstructed transcript",
    )
    tool_output_chars: int = Field(
        default=120,
        description="Max characters of a tool's output to include in the reconstructed transcript",
    )
    tool_result_chars: int = Field(
        default=200,
        description="Max characters of a Claude Code tool_result block to include in the reconstructed transcript",
    )

MEMORY_GUIDELINE_PROMPT = """You are analyzing a Claude Code session transcript to identify durable facts
that should be remembered across sessions.

MEMORY TYPES (ordered by retrieval value):

1. **user** — HIGHEST VALUE. Facts about the human's identity, role, expertise,
   and stable preferences. These directly shape how the agent should behave
   next session. Examples worth extracting:
   - "User works on slopometry, a Claude Code session tracker"
   - "User prefers strict type checking with pyright in basic mode"
   - "User maintains slopometry with uv tool install"
   - "User dislikes emojis in commit messages and code comments"
   - "User is a junior learning code review skills"
   Extract when the user STATES a preference, role, or stable fact about
   themselves. Don't extract one-off statements ("let me try X") or
   task-specific decisions.

2. **feedback** — Guidance on how to work, corrections, confirmed approaches.
   ALWAYS include the WHY ("learned from incident X", "user confirmed Y
   after we tried Z"). Without the why, the feedback is unsearchable later.
   Examples:
   - "Replace pytest.skip with assert when the test setup would have failed
     anyway — the skip masks a real problem (learned reviewing PR)"
   - "Use `git ls-files --cached --others` + `git hash-object`, NOT
     `write-tree`, for source digest (write-tree breaks existing filter
     tests)"

3. **project** — Work goals, constraints, topology NOT derivable from the
   current code or git. Examples:
   - "Project uses uv for dependency management, not pip"
   - "Database schema lives in `core/database.py` with raw SQL + migrations"
   - "Stop-hook feedback is gated by `settings.enable_complexity_feedback`,
     not always-on"
   Skip facts already visible in the codebase (file structure, imports,
   pyproject.toml contents, recent commits). Those are reconstructable.

4. **reference** — External resource pointers (URLs, dashboards, tickets,
   fork locations). Examples:
   - "Rust code analysis fork: github.com/Droidcraft/rust-code-analysis —
     install via `cargo install --git`"
   - "VictoriaMetrics uses `metric_relabel_configs` (not `relabel_configs`)
     for post-collection filtering"

MEMORY CRITERIA — the Litmus Test:
"If I started fresh next session, would NOT knowing this make me repeat a
mistake, re-derive something hard, or act against user preference?"
If YES → memory. If NO → skip.

HYGIENE:
- Convert relative dates ("19 days ago", "last week") to absolute dates
  (ISO format) so the memory stays meaningful as time passes
- One fact per memory — split compound observations
- Skip reconstructable facts: code structure, imports, recent git history,
  CLAUDE.md contents, package versions — these are visible in the repo
- Skip one-off task decisions ("we'll use Redis for this feature") that
  don't generalize

ANTI-STALENESS RULES (the failures we've seen):
- DO NOT extract a tool/dependency claim without a "current as of" qualifier
  if you cannot verify it. Bad: "Project uses radon". Good: "Project uses
  rust-code-analysis (radon was abandoned ~5 years ago per README history)".
- If the user mentions SWITCHING from one tool to another, prefer the
  CURRENT tool. Bad: extract both "we used X" and "we now use Y" — just
  extract "we now use Y, switched from X because Z".
- For facts that may have changed since the transcript (dependency
  versions, framework choices, team structure), include the temporal
  context: "as of 2026-06, X" — the freshness validator will surface
  these for review.
- Avoid imperative instructions shaped as memories ("use X for Y") unless
  the user explicitly endorsed the approach. Otherwise classify as
  `project` or `feedback` with the why.

Return a JSON array. Empty array is acceptable if no facts qualify:
[
  {
    "memory_type": "user",
    "content": "User maintains slopometry as a uv tool, not as a package install",
    "source_context": "stated when discussing install paths"
  },
  {
    "memory_type": "feedback",
    "content": "Prefer `enum.StrEnum` over `(str, Enum)` for new enums (learned from ruff UP042 cleanup)",
    "source_context": "code review during abstract hook protocol refactor"
  },
  {
    "memory_type": "project",
    "content": "slopometry uses rust-code-analysis (not radon) for Python complexity metrics, since 2026-01",
    "source_context": "documented in README 'BREAKING CHANGE' section"
  },
  {
    "memory_type": "reference",
    "content": "Internal LLM endpoint: https://llm2.droidcraft.org/minimax-m2-7/v1 (model: minimax-m2-7)",
    "source_context": "configured in settings.memory_llm_endpoint"
  }
]

Transcript to analyze:
"""


class LLMConnectionError(Exception):
    """Raised when LLM endpoint is unreachable or returns an error."""

    pass


class MemoryExtractor:
    """Extracts memory candidates from transcripts using LLM."""

    def __init__(self, llm_endpoint: str, llm_model: str, api_key: str = "dummy"):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.api_key = api_key

    def extract_memories_from_transcript(
        self,
        transcript_path: Path,
        truncation: TranscriptTruncationConfig | None = None,
    ) -> str:
        """Parse JSONL transcript and clean noise.

        Returns:
            Cleaned conversation text suitable for LLM analysis
        """
        truncation = truncation or TranscriptTruncationConfig()
        try:
            with open(transcript_path, encoding="utf-8") as f:
                lines = f.readlines()

            conversation_parts: list[str] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    msg_type = data.get("type")

                    if msg_type == "user":
                        message = data.get("message", {})
                        content = message.get("content", [])
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_result":
                                    result_text = block.get("content", "")
                                    if isinstance(result_text, str):
                                        text_parts.append(
                                            f"[tool result: {result_text[:truncation.tool_result_chars]}]"
                                        )
                        if text_parts:
                            conversation_parts.append(f"USER: {''.join(text_parts)}")

                    elif msg_type == "assistant":
                        message = data.get("message", {})
                        content = message.get("content", [])
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    text_parts.append(f"[TOOL: {tool_name}]")
                        if text_parts:
                            conversation_parts.append(f"ASSISTANT: {''.join(text_parts)}")

                    elif msg_type == "system":
                        subtype = data.get("subtype", "")
                        if subtype in ("stop_hook_summary", "ai-title"):
                            continue
                        message = data.get("message", {})
                        if message:
                            content = message.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))

                except (json.JSONDecodeError, KeyError):
                    continue

            return "\n".join(conversation_parts)

        except Exception as e:
            logger.error(f"Failed to parse transcript {transcript_path}: {e}")
            return ""

    def extract_memories_from_opencode_session(
        self,
        session_id: str,
        storage_root: Path,
        truncation: TranscriptTruncationConfig | None = None,
    ) -> str:
        """Reconstruct conversation text from OpenCode's session/message/part layout.

        OpenCode stores conversation state as separate JSON files under
        ``<storage_root>/message/<session_id>/<message_id>.json`` and
        ``<storage_root>/part/<message_id>/<part_id>.json`` rather than as a
        single JSONL transcript. This method walks those files in
        chronological order and emits the same ``USER:`` / ``ASSISTANT:``
        text format the Claude Code parser produces, so downstream LLM
        extraction works unchanged.

        Args:
            session_id: OpenCode session identifier (``ses_...``)
            storage_root: OpenCode storage root (``~/.local/share/opencode/storage``)
            truncation: Per-tool-part summary truncation limits

        Returns:
            Cleaned conversation text suitable for LLM analysis
        """
        truncation = truncation or TranscriptTruncationConfig()
        message_dir = storage_root / "message" / session_id
        if not message_dir.is_dir():
            logger.debug("No message directory for OpenCode session %s", session_id)
            return ""

        try:
            message_files = sorted(
                message_dir.glob("*.json"),
                key=lambda p: json.loads(p.read_text(encoding="utf-8")).get("time", {}).get("created", 0),
            )
        except (OSError, ValueError) as e:
            logger.error("Failed to enumerate OpenCode messages for %s: %s", session_id, e)
            return ""

        conversation_parts: list[str] = []
        for message_path in message_files:
            try:
                message = json.loads(message_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            role = message.get("role")
            if role not in ("user", "assistant"):
                continue

            message_id = message.get("id")
            if not message_id:
                continue

            part_dir = storage_root / "part" / message_id
            if not part_dir.is_dir():
                continue

            try:
                part_files = sorted(part_dir.glob("*.json"))
            except OSError:
                continue

            text_parts: list[str] = []
            for part_path in part_files:
                try:
                    part = json.loads(part_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

                part_type = part.get("type")
                if part_type == "text":
                    text_parts.append(part.get("text", ""))
                elif part_type == "tool":
                    tool_name = part.get("tool", "unknown")
                    state = part.get("state", {})
                    input_summary = json.dumps(state.get("input", {}))[: truncation.tool_input_chars]
                    output_summary = (state.get("output") or "")[: truncation.tool_output_chars]
                    text_parts.append(
                        f"[TOOL: {tool_name} input={input_summary!r} output={output_summary!r}]"
                    )

            if text_parts:
                prefix = "USER" if role == "user" else "ASSISTANT"
                conversation_parts.append(f"{prefix}: {''.join(text_parts)}")

        return "\n".join(conversation_parts)

    def generate_memory_candidates(self, transcript_snippet: str) -> list[MemoryCandidate]:
        """Generate memory candidates from transcript using LLM.

        Args:
            transcript_snippet: Cleaned conversation text

        Returns:
            List of MemoryCandidate objects

        Raises:
            LLMConnectionError: If LLM endpoint is unreachable or returns an error
            ValueError: If response cannot be parsed as memory candidates
        """
        if not transcript_snippet.strip():
            return []

        try:
            from openai import OpenAI
        except ImportError:
            raise LLMConnectionError("openai package required for memory extraction. Install with: pip install openai")

        try:
            client = OpenAI(base_url=self.llm_endpoint, api_key=self.api_key)
        except Exception as e:
            raise LLMConnectionError(f"Failed to create OpenAI client: {e}") from e

        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts memory candidates."},
                    {"role": "user", "content": MEMORY_GUIDELINE_PROMPT + transcript_snippet[:15000]},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as e:
            raise LLMConnectionError(f"Failed to connect to LLM endpoint {self.llm_endpoint}: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")

        json_str = content.strip()

        if json_str.startswith("<think>"):
            end_marker = "</think>"
            end_idx = json_str.find(end_marker)
            if end_idx != -1:
                json_str = json_str[end_idx + len(end_marker) :]
                while json_str.startswith("\n"):
                    json_str = json_str[1:]

        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        candidates: list[MemoryCandidate] = []
        for item in data:
            try:
                memory_type_str = item.get("memory_type", "")
                if memory_type_str not in ["user", "feedback", "project", "reference"]:
                    continue

                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType(memory_type_str),
                        content=item.get("content", ""),
                        source_context=item.get("source_context"),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping invalid memory candidate: {e}")
                continue

        return candidates

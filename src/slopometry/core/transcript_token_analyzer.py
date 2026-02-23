"""Transcript token analyzer for extracting and categorizing token usage."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slopometry.core.models.hook import ToolType
from slopometry.core.models.session import TokenUsage
from slopometry.core.plan_analyzer import PlanAnalyzer

logger = logging.getLogger(__name__)


class TranscriptMetadata(BaseModel):
    """Lightweight metadata extracted from the first few lines of a transcript."""

    agent_version: str | None = None
    model: str | None = None
    git_branch: str | None = None


def extract_transcript_metadata(transcript_path: Path) -> TranscriptMetadata:
    """Extract version, model, and git branch from a transcript file.

    Makes a single pass through the transcript, stopping early once all
    fields are found. Extracts:
    - version from first event that has it (Claude Code version)
    - message.model from first assistant event (LLM model)
    - gitBranch from first event that has it

    Args:
        transcript_path: Path to the JSONL transcript file

    Returns:
        TranscriptMetadata with extracted values (None for missing fields)
    """
    agent_version: str | None = None
    model: str | None = None
    git_branch: str | None = None
    skipped_lines = 0

    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping unparseable line in metadata extraction")
                    skipped_lines += 1
                    continue

                if agent_version is None and isinstance(event.get("version"), str):
                    agent_version = event["version"]

                if git_branch is None and isinstance(event.get("gitBranch"), str):
                    git_branch = event["gitBranch"]

                if model is None and event.get("type") == "assistant":
                    msg = event.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("model"), str):
                        model = msg["model"]

                if agent_version is not None and model is not None and git_branch is not None:
                    break
    except OSError as e:
        logger.warning(f"Failed to read transcript for metadata: {e}")

    if skipped_lines:
        logger.warning(f"Skipped {skipped_lines} unparseable line(s) in metadata extraction from {transcript_path}")

    return TranscriptMetadata(agent_version=agent_version, model=model, git_branch=git_branch)


class MessageUsage(BaseModel):
    """Token usage from an assistant message."""

    input_tokens: int = 0
    output_tokens: int = 0


class AssistantMessage(BaseModel):
    """Assistant message with usage and content."""

    usage: MessageUsage = Field(default_factory=MessageUsage)
    # content can be a list (Claude) or a string (some other models like MiniMax)
    content: list[dict[str, Any]] | str = Field(default="")


class ToolUseResult(BaseModel):
    """Result from a tool use (subagent tokens)."""

    totalTokens: int = 0


class TranscriptEvent(BaseModel, extra="allow"):
    """A single event from transcript.jsonl.

    Uses extra="allow" to tolerate additional fields from Claude Code.
    """

    type: str | None = None
    message: AssistantMessage | None = None
    # toolUseResult can be a dict, ToolUseResult, or string (error messages from some models)
    toolUseResult: ToolUseResult | dict[str, Any] | str | None = None


class TranscriptTokenAnalyzer:
    """Analyzes Claude Code transcripts to extract token usage by category."""

    def __init__(self) -> None:
        """Initialize the analyzer with tool classification sets."""
        self.search_tools = PlanAnalyzer.SEARCH_TOOLS
        self.implementation_tools = PlanAnalyzer.IMPLEMENTATION_TOOLS
        self._last_input_tokens: int = 0
        self._tool_use_categories: dict[str, str] = {}
        self._latest_raw_input_tokens: int = 0

    def analyze_transcript(self, transcript_path: Path) -> TokenUsage:
        """Analyze a transcript file and return categorized token usage.

        Args:
            transcript_path: Path to the JSONL transcript file

        Returns:
            TokenUsage with tokens categorized by exploration vs implementation
        """
        usage = TokenUsage()
        skipped_lines = 0

        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        raw_event = json.loads(line)
                        event = TranscriptEvent.model_validate(raw_event)
                    except Exception:
                        logger.debug("Skipping unparseable transcript event")
                        skipped_lines += 1
                        continue

                    self._process_event(event, usage)

        except OSError as e:
            logger.warning(f"Failed to read transcript file {transcript_path}: {e}")

        if skipped_lines:
            logger.warning(f"Skipped {skipped_lines} unparseable line(s) in {transcript_path}")

        usage.final_context_input_tokens = self._latest_raw_input_tokens
        usage.subagent_tokens = usage.explore_subagent_tokens + usage.non_explore_subagent_tokens

        return usage

    def _process_event(self, event: TranscriptEvent, usage: TokenUsage) -> None:
        """Process a single transcript event and update token usage.

        Args:
            event: Parsed transcript event
            usage: TokenUsage to update
        """
        if event.type == "assistant":
            self._process_assistant_message(event, usage)
        elif event.type == "user":
            self._process_tool_result(event, usage)

    def _process_assistant_message(self, event: TranscriptEvent, usage: TokenUsage) -> None:
        """Process an assistant message to extract and categorize tokens.

        Uses incremental input tokens (delta from previous turn) because each
        API response's input_tokens is cumulative — it includes all prior context.
        Summing raw input_tokens across turns would massively over-count.
        The delta is clamped to 0 to handle context compactions where the next
        turn's input may be smaller than the previous.

        Args:
            event: Assistant message event
            usage: TokenUsage to update
        """
        if not event.message:
            return

        msg_usage = event.message.usage
        if not msg_usage.input_tokens and not msg_usage.output_tokens:
            return

        raw_input_tokens = msg_usage.input_tokens
        output_tokens = msg_usage.output_tokens
        self._latest_raw_input_tokens = raw_input_tokens

        # Compute incremental input: only the new tokens added since last turn.
        # Each turn's input_tokens includes all prior context, so the delta
        # represents the actual new content (previous output + new user message).
        incremental_input = max(0, raw_input_tokens - self._last_input_tokens)
        self._last_input_tokens = raw_input_tokens

        usage.total_input_tokens += incremental_input
        usage.total_output_tokens += output_tokens

        content = event.message.content
        # Skip tool categorization if content is a string (non-standard format)
        if isinstance(content, str):
            # Fall back to implementation-only tokens for string content
            usage.implementation_input_tokens += incremental_input
            usage.implementation_output_tokens += output_tokens
            return

        tool_categories = self._get_tool_categories(content)

        if not tool_categories:
            usage.implementation_input_tokens += incremental_input
            usage.implementation_output_tokens += output_tokens
        elif "exploration" in tool_categories and "implementation" in tool_categories:
            exploration_ratio = tool_categories.count("exploration") / len(tool_categories)
            usage.exploration_input_tokens += int(incremental_input * exploration_ratio)
            usage.exploration_output_tokens += int(output_tokens * exploration_ratio)
            usage.implementation_input_tokens += int(incremental_input * (1 - exploration_ratio))
            usage.implementation_output_tokens += int(output_tokens * (1 - exploration_ratio))
        elif "exploration" in tool_categories:
            usage.exploration_input_tokens += incremental_input
            usage.exploration_output_tokens += output_tokens
        else:
            usage.implementation_input_tokens += incremental_input
            usage.implementation_output_tokens += output_tokens

    def _get_tool_categories(self, content: list[dict[str, Any]]) -> list[str]:
        """Extract tool categories from message content.

        Also stores tool_use_id → category in self._tool_use_categories so that
        subsequent user events (carrying toolUseResult) can route subagent tokens
        to the correct bucket.

        Args:
            content: Message content list (not string - caller handles that case)

        Returns:
            List of categories ("exploration" or "implementation") for each tool
        """
        categories = []

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") != "tool_use":
                continue

            tool_name = item.get("name", "")
            tool_input = item.get("input", {})
            tool_use_id = item.get("id", "")

            category = self._classify_tool(tool_name, tool_input)
            categories.append(category)

            if tool_use_id:
                self._tool_use_categories[tool_use_id] = category

        return categories

    def _classify_tool(self, tool_name: str, tool_input: dict) -> str:
        """Classify a tool as exploration or implementation.

        Args:
            tool_name: Name of the tool
            tool_input: Input parameters for the tool

        Returns:
            "exploration" or "implementation"
        """
        if tool_name == "Task":
            subagent_type = tool_input.get("subagent_type", "")
            if subagent_type and "explore" in subagent_type.lower():
                return "exploration"
            return "implementation"

        try:
            tool_type = ToolType(tool_name)
            if tool_type in self.search_tools:
                return "exploration"
            if tool_type in self.implementation_tools:
                return "implementation"
        except ValueError:
            logger.debug(f"Unknown tool type '{tool_name}', defaulting to implementation")

        return "implementation"

    def _process_tool_result(self, event: TranscriptEvent, usage: TokenUsage) -> None:
        """Process tool result to capture and route subagent tokens.

        Routes subagent tokens to explore_subagent_tokens or
        non_explore_subagent_tokens based on the originating tool_use category.
        Falls back to non_explore if the tool_use_id is unknown.

        Args:
            event: User message event (containing tool_result)
            usage: TokenUsage to update
        """
        total_tokens = 0
        # Handle different toolUseResult types: ToolUseResult, dict, or string
        tool_result = event.toolUseResult
        if isinstance(tool_result, ToolUseResult):
            total_tokens = tool_result.totalTokens
        elif isinstance(tool_result, dict):
            total_tokens = tool_result.get("totalTokens", 0) or 0
        # String toolUseResult (error messages) - ignore

        if not total_tokens:
            return

        # Extract tool_use_id from the user event's message content
        tool_use_id = self._extract_tool_use_id(event)
        category = self._tool_use_categories.get(tool_use_id, "implementation") if tool_use_id else "implementation"

        if category == "exploration":
            usage.explore_subagent_tokens += total_tokens
        else:
            usage.non_explore_subagent_tokens += total_tokens

    @staticmethod
    def _extract_tool_use_id(event: TranscriptEvent) -> str | None:
        """Extract tool_use_id from a user event's message content.

        User events in Claude Code transcripts contain tool_result blocks
        with a tool_use_id field referencing the originating assistant tool_use.

        Args:
            event: User message event

        Returns:
            The tool_use_id string, or None if not found
        """
        if not event.message:
            return None
        content = event.message.content
        if not isinstance(content, list):
            return None
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_use_id = item.get("tool_use_id")
                if tool_use_id:
                    return str(tool_use_id)
        return None


def analyze_transcript_tokens(transcript_path: Path) -> TokenUsage:
    """Convenience function to analyze token usage from a Claude Code transcript.

    Args:
        transcript_path: Path to Claude Code transcript JSONL

    Returns:
        TokenUsage with categorized metrics
    """
    analyzer = TranscriptTokenAnalyzer()
    return analyzer.analyze_transcript(transcript_path)


def analyze_opencode_transcript_tokens(transcript: list[dict]) -> TokenUsage:
    """Analyze token usage from an OpenCode session transcript.

    OpenCode transcripts are stored in stop event metadata (fetched via SDK).
    Each message has: role, tokens: {input, output, reasoning}, parts: [{type, tool}].

    Uses incremental input tokens (same cumulative issue as Claude Code) and
    classifies messages by tool usage in parts.

    Args:
        transcript: List of message dicts from the stop event metadata.

    Returns:
        TokenUsage with categorized metrics
    """
    usage = TokenUsage()
    last_input_tokens = 0
    search_tools = PlanAnalyzer.SEARCH_TOOLS
    implementation_tools = PlanAnalyzer.IMPLEMENTATION_TOOLS

    for msg in transcript:
        if msg.get("role") != "assistant":
            continue

        tokens = msg.get("tokens")
        if not tokens:
            continue

        raw_input = tokens.get("input", 0) or 0
        output = tokens.get("output", 0) or 0
        reasoning = tokens.get("reasoning", 0) or 0

        # Incremental input (same cumulative issue as Claude Code)
        incremental_input = max(0, raw_input - last_input_tokens)
        last_input_tokens = raw_input

        total_output = output + reasoning

        usage.total_input_tokens += incremental_input
        usage.total_output_tokens += total_output

        # Primary signal: agent field (e.g. "explore", "plan", "general")
        agent = msg.get("agent", "")
        if agent in ("explore", "search"):
            usage.exploration_input_tokens += incremental_input
            usage.exploration_output_tokens += total_output
            continue

        # Fallback: classify by tool usage in parts
        tool_categories = _classify_opencode_parts(msg.get("parts", []), search_tools, implementation_tools)

        if not tool_categories:
            usage.implementation_input_tokens += incremental_input
            usage.implementation_output_tokens += total_output
        elif "exploration" in tool_categories and "implementation" in tool_categories:
            exploration_ratio = tool_categories.count("exploration") / len(tool_categories)
            usage.exploration_input_tokens += int(incremental_input * exploration_ratio)
            usage.exploration_output_tokens += int(total_output * exploration_ratio)
            usage.implementation_input_tokens += int(incremental_input * (1 - exploration_ratio))
            usage.implementation_output_tokens += int(total_output * (1 - exploration_ratio))
        elif "exploration" in tool_categories:
            usage.exploration_input_tokens += incremental_input
            usage.exploration_output_tokens += total_output
        else:
            usage.implementation_input_tokens += incremental_input
            usage.implementation_output_tokens += total_output

    usage.final_context_input_tokens = last_input_tokens

    return usage


def _classify_opencode_parts(
    parts: list[dict], search_tools: set[ToolType], implementation_tools: set[ToolType]
) -> list[str]:
    """Classify tools in OpenCode message parts as exploration or implementation.

    Args:
        parts: Message parts list from OpenCode transcript.
        search_tools: Set of ToolTypes considered exploration.
        implementation_tools: Set of ToolTypes considered implementation.

    Returns:
        List of categories for each tool part found.
    """
    categories = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        tool_name = part.get("tool")
        if not tool_name:
            continue

        # Case-insensitive matching: OpenCode uses lowercase tool names
        # (e.g. "grep") while ToolType enum uses PascalCase (e.g. "Grep")
        tool_name_lower = tool_name.lower()
        matched = False
        for tool_type in search_tools:
            if tool_type.value.lower() == tool_name_lower:
                categories.append("exploration")
                matched = True
                break
        if not matched:
            for tool_type in implementation_tools:
                if tool_type.value.lower() == tool_name_lower:
                    categories.append("implementation")
                    matched = True
                    break
        if not matched:
            categories.append("implementation")
    return categories

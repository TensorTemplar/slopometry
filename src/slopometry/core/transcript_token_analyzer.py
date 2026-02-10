"""Transcript token analyzer for extracting and categorizing token usage."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slopometry.core.models import TokenUsage, ToolType
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
    content: list[dict[str, Any]] = Field(default_factory=list)


class ToolUseResult(BaseModel):
    """Result from a tool use (subagent tokens)."""

    totalTokens: int = 0  # camelCase matches transcript format


class TranscriptEvent(BaseModel, extra="allow"):
    """A single event from transcript.jsonl.

    Uses extra="allow" to tolerate additional fields from Claude Code.
    """

    type: str | None = None
    message: AssistantMessage | None = None
    toolUseResult: ToolUseResult | None = None  # camelCase matches transcript


class TranscriptTokenAnalyzer:
    """Analyzes Claude Code transcripts to extract token usage by category."""

    def __init__(self) -> None:
        """Initialize the analyzer with tool classification sets."""
        self.search_tools = PlanAnalyzer.SEARCH_TOOLS
        self.implementation_tools = PlanAnalyzer.IMPLEMENTATION_TOOLS

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

        Args:
            event: Assistant message event
            usage: TokenUsage to update
        """
        if not event.message:
            return

        msg_usage = event.message.usage
        if not msg_usage.input_tokens and not msg_usage.output_tokens:
            return

        input_tokens = msg_usage.input_tokens
        output_tokens = msg_usage.output_tokens

        usage.total_input_tokens += input_tokens
        usage.total_output_tokens += output_tokens

        content = event.message.content
        tool_categories = self._get_tool_categories(content)

        if not tool_categories:
            usage.implementation_input_tokens += input_tokens
            usage.implementation_output_tokens += output_tokens
        elif "exploration" in tool_categories and "implementation" in tool_categories:
            exploration_ratio = tool_categories.count("exploration") / len(tool_categories)
            usage.exploration_input_tokens += int(input_tokens * exploration_ratio)
            usage.exploration_output_tokens += int(output_tokens * exploration_ratio)
            usage.implementation_input_tokens += int(input_tokens * (1 - exploration_ratio))
            usage.implementation_output_tokens += int(output_tokens * (1 - exploration_ratio))
        elif "exploration" in tool_categories:
            usage.exploration_input_tokens += input_tokens
            usage.exploration_output_tokens += output_tokens
        else:
            usage.implementation_input_tokens += input_tokens
            usage.implementation_output_tokens += output_tokens

    def _get_tool_categories(self, content: list) -> list[str]:
        """Extract tool categories from message content.

        Args:
            content: Message content list

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

            category = self._classify_tool(tool_name, tool_input)
            categories.append(category)

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
        """Process tool result to capture subagent tokens.

        Subagent tokens are tracked separately in subagent_tokens field,
        not added to exploration/implementation totals since they're
        external to the main conversation flow.

        Args:
            event: User message event (containing tool_result)
            usage: TokenUsage to update
        """
        if event.toolUseResult and event.toolUseResult.totalTokens:
            usage.subagent_tokens += event.toolUseResult.totalTokens


def analyze_transcript_tokens(transcript_path: Path) -> TokenUsage:
    """Convenience function to analyze token usage from a transcript.

    Args:
        transcript_path: Path to Claude Code transcript JSONL

    Returns:
        TokenUsage with categorized metrics
    """
    analyzer = TranscriptTokenAnalyzer()
    return analyzer.analyze_transcript(transcript_path)

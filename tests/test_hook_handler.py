"""Tests for hook handler functionality."""

from slopometry.hook_handler import detect_event_type_from_parsed
from slopometry.models import (
    HookEventType,
    NotificationInput,
    PostToolUseInput,
    PreToolUseInput,
    StopInput,
    SubagentStopInput,
)


class TestEventTypeDetection:
    """Test the pattern match logic for detecting event types."""

    def test_pre_tool_use_input_detection(self):
        """Test that PreToolUseInput maps to PRE_TOOL_USE event type."""
        input_data = PreToolUseInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.PRE_TOOL_USE

    def test_post_tool_use_input_detection(self):
        """Test that PostToolUseInput maps to POST_TOOL_USE event type."""
        input_data = PostToolUseInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"success": True},
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.POST_TOOL_USE

    def test_notification_input_detection(self):
        """Test that NotificationInput maps to NOTIFICATION event type."""
        input_data = NotificationInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            message="Test notification",
            title="Test Title",
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.NOTIFICATION

    def test_stop_input_detection(self):
        """Test that StopInput maps to STOP event type."""
        input_data = StopInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            stop_hook_active=True,
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.STOP

    def test_subagent_stop_input_detection(self):
        """Test that SubagentStopInput maps to SUBAGENT_STOP event type."""
        input_data = SubagentStopInput(
            session_id="test-session",
            transcript_path="/tmp/test.jsonl",
            stop_hook_active=True,
        )

        result = detect_event_type_from_parsed(input_data)

        assert result == HookEventType.SUBAGENT_STOP

    def test_all_input_types_are_handled(self):
        """Test that all defined input types have corresponding pattern matches.

        This test ensures we don't forget to update the pattern match when adding new input types.
        """
        input_types = [
            PreToolUseInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                tool_name="Test",
            ),
            PostToolUseInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                tool_name="Test",
                tool_response="success",
            ),
            NotificationInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
                message="test",
            ),
            StopInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
            ),
            SubagentStopInput(
                session_id="test",
                transcript_path="/tmp/test.jsonl",
            ),
        ]

        expected_types = [
            HookEventType.PRE_TOOL_USE,
            HookEventType.POST_TOOL_USE,
            HookEventType.NOTIFICATION,
            HookEventType.STOP,
            HookEventType.SUBAGENT_STOP,
        ]

        for input_data, expected_type in zip(input_types, expected_types):
            result = detect_event_type_from_parsed(input_data)
            assert result == expected_type, f"Input {type(input_data).__name__} should map to {expected_type}"

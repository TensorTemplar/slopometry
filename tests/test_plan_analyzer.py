from slopometry.core.models import ToolType
from slopometry.core.plan_analyzer import PlanAnalyzer


def test_increment_event_count__task_explore_increments_search_metrics():
    """Verify that a TASK tool with 'Explore' subagent type is counted as a search event."""
    analyzer = PlanAnalyzer()

    # Initial state
    assert analyzer.search_events_since_last_todo == 0
    assert analyzer.implementation_events_since_last_todo == 0

    # Act
    tool_input = {"subagent_type": "Explore", "prompt": "foo"}
    analyzer.increment_event_count(ToolType.TASK, tool_input)

    # Assert
    assert analyzer.search_events_since_last_todo == 1
    assert analyzer.implementation_events_since_last_todo == 0
    assert analyzer.events_since_last_todo == 1


def test_increment_event_count__task_plan_increments_implementation_metrics():
    """Verify that a TASK tool with 'Plan' subagent type is counted as an implementation event."""
    analyzer = PlanAnalyzer()

    # Initial state
    assert analyzer.search_events_since_last_todo == 0
    assert analyzer.implementation_events_since_last_todo == 0

    # Act
    tool_input = {"subagent_type": "Plan", "prompt": "foo"}
    analyzer.increment_event_count(ToolType.TASK, tool_input)

    # Assert
    assert analyzer.search_events_since_last_todo == 0
    assert analyzer.implementation_events_since_last_todo == 1
    assert analyzer.events_since_last_todo == 1


def test_increment_event_count__task_unknown_defaults_to_implementation():
    """Verify that a TASK tool with unknown or missing subagent type defaults to implementation."""
    analyzer = PlanAnalyzer()

    # Act 1: Unknown type
    tool_input = {"subagent_type": "SomethingElse"}
    analyzer.increment_event_count(ToolType.TASK, tool_input)
    assert analyzer.implementation_events_since_last_todo == 1

    # Act 2: No input
    analyzer.increment_event_count(ToolType.TASK, None)
    assert analyzer.implementation_events_since_last_todo == 2

    assert analyzer.search_events_since_last_todo == 0


def test_increment_event_count__standard_search_tools_still_count_as_search():
    """Verify that standard search tools (like READ) are still counted as search."""
    analyzer = PlanAnalyzer()

    analyzer.increment_event_count(ToolType.READ)

    assert analyzer.search_events_since_last_todo == 1
    assert analyzer.implementation_events_since_last_todo == 0


def test_increment_event_count__standard_implementation_tools_still_count_as_implementation():
    """Verify that standard implementation tools (like WRITE) are still counted as implementation."""
    analyzer = PlanAnalyzer()

    analyzer.increment_event_count(ToolType.WRITE)

    assert analyzer.search_events_since_last_todo == 0
    assert analyzer.implementation_events_since_last_todo == 1

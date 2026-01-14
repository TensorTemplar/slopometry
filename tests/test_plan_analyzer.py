from slopometry.core.models import ToolType
from slopometry.core.plan_analyzer import PlanAnalyzer


def test_increment_event_count__task_explore_increments_search_metrics() -> None:
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


def test_increment_event_count__task_plan_increments_implementation_metrics() -> None:
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


def test_increment_event_count__task_unknown_defaults_to_implementation() -> None:
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


def test_increment_event_count__standard_search_tools_still_count_as_search() -> None:
    """Verify that standard search tools (like READ) are still counted as search."""
    analyzer = PlanAnalyzer()

    analyzer.increment_event_count(ToolType.READ)

    assert analyzer.search_events_since_last_todo == 1
    assert analyzer.implementation_events_since_last_todo == 0


def test_increment_event_count__standard_implementation_tools_still_count_as_implementation() -> None:
    """Verify that standard implementation tools (like WRITE) are still counted as implementation."""
    analyzer = PlanAnalyzer()

    analyzer.increment_event_count(ToolType.WRITE)

    assert analyzer.search_events_since_last_todo == 0
    assert analyzer.implementation_events_since_last_todo == 1


def test_analyze_write_event__detects_plan_file() -> None:
    """Write to ~/.claude/plans/*.md should be tracked as a plan file."""
    analyzer = PlanAnalyzer()

    tool_input = {"file_path": "/home/user/.claude/plans/zany-strolling-eich.md"}
    analyzer.analyze_write_event(tool_input)

    evolution = analyzer.get_plan_evolution()
    assert evolution.plan_files_created == 1
    assert "/home/user/.claude/plans/zany-strolling-eich.md" in evolution.plan_file_paths


def test_analyze_write_event__ignores_non_plan_file() -> None:
    """Write to regular files should not be tracked as plan files."""
    analyzer = PlanAnalyzer()

    tool_input = {"file_path": "/home/user/project/src/main.py"}
    analyzer.analyze_write_event(tool_input)

    evolution = analyzer.get_plan_evolution()
    assert evolution.plan_files_created == 0
    assert len(evolution.plan_file_paths) == 0


def test_analyze_write_event__handles_windows_paths() -> None:
    """Write to .claude\\plans\\ on Windows should be tracked."""
    analyzer = PlanAnalyzer()

    tool_input = {"file_path": "C:\\Users\\test\\.claude\\plans\\my-plan.md"}
    analyzer.analyze_write_event(tool_input)

    evolution = analyzer.get_plan_evolution()
    assert evolution.plan_files_created == 1


def test_analyze_write_event__deduplicates_same_file() -> None:
    """Multiple writes to the same plan file should count as one."""
    analyzer = PlanAnalyzer()

    tool_input = {"file_path": "/home/user/.claude/plans/test-plan.md"}
    analyzer.analyze_write_event(tool_input)
    analyzer.analyze_write_event(tool_input)  # Same file again

    evolution = analyzer.get_plan_evolution()
    assert evolution.plan_files_created == 1
    assert len(evolution.plan_file_paths) == 1


def test_analyze_write_event__handles_empty_file_path() -> None:
    """Missing or empty file_path should not cause errors."""
    analyzer = PlanAnalyzer()

    # Empty string
    analyzer.analyze_write_event({"file_path": ""})
    # Missing key
    analyzer.analyze_write_event({})

    evolution = analyzer.get_plan_evolution()
    assert evolution.plan_files_created == 0


def test_get_plan_evolution__includes_final_todos() -> None:
    """Verify that final_todos contains the todo items from the last TodoWrite."""
    from datetime import datetime

    analyzer = PlanAnalyzer()

    # Simulate a TodoWrite event with multiple todos
    tool_input = {
        "todos": [
            {"content": "First task", "status": "completed", "activeForm": "Completing first task"},
            {"content": "Second task", "status": "in_progress", "activeForm": "Working on second task"},
            {"content": "Third task", "status": "pending", "activeForm": "Pending third task"},
        ]
    }
    analyzer.analyze_todo_write_event(tool_input, datetime.now())

    evolution = analyzer.get_plan_evolution()

    # Verify final_todos is populated
    assert len(evolution.final_todos) == 3

    # Verify each todo has correct content and status
    contents = {todo.content for todo in evolution.final_todos}
    assert "First task" in contents
    assert "Second task" in contents
    assert "Third task" in contents

    # Verify statuses are preserved
    status_by_content = {todo.content: todo.status for todo in evolution.final_todos}
    assert status_by_content["First task"] == "completed"
    assert status_by_content["Second task"] == "in_progress"
    assert status_by_content["Third task"] == "pending"


def test_get_plan_evolution__final_todos_empty_when_no_todowrite() -> None:
    """Verify final_todos is empty when no TodoWrite events occurred."""
    analyzer = PlanAnalyzer()

    # Only add a plan file, no TodoWrite events
    analyzer.analyze_write_event({"file_path": "/home/user/.claude/plans/test.md"})

    evolution = analyzer.get_plan_evolution()

    assert evolution.final_todos == []
    assert evolution.plan_files_created == 1

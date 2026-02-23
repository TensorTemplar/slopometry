from datetime import datetime

from slopometry.core.models.hook import ToolType
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


# --- TaskCreate / TaskUpdate tests (Claude Code new tool names) ---


def test_analyze_task_create_event__creates_todo_and_plan_step() -> None:
    """TaskCreate should create a TodoItem and generate a PlanStep."""
    analyzer = PlanAnalyzer()

    tool_input = {"subject": "Fix login bug", "description": "Fix the auth flow", "activeForm": "Fixing login bug"}
    tool_response = {"taskId": "1"}

    analyzer.analyze_task_create_event(tool_input, tool_response, datetime.now())

    evolution = analyzer.get_plan_evolution()
    assert evolution.total_plan_steps == 1
    assert len(evolution.final_todos) == 1
    assert evolution.final_todos[0].content == "Fix login bug"
    assert evolution.final_todos[0].status == "pending"
    assert evolution.final_todos[0].activeForm == "Fixing login bug"


def test_analyze_task_create_event__multiple_tasks() -> None:
    """Multiple TaskCreate events should accumulate todos."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_create_event(
        {"subject": "Task A", "activeForm": "Working on A"},
        {"taskId": "1"},
        datetime.now(),
    )
    analyzer.analyze_task_create_event(
        {"subject": "Task B", "activeForm": "Working on B"},
        {"taskId": "2"},
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 2
    contents = {t.content for t in evolution.final_todos}
    assert contents == {"Task A", "Task B"}


def test_analyze_task_create_event__ignores_empty_subject() -> None:
    """TaskCreate with empty subject should be ignored."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_create_event({"subject": ""}, {}, datetime.now())

    evolution = analyzer.get_plan_evolution()
    assert evolution.total_plan_steps == 0
    assert len(evolution.final_todos) == 0


def test_analyze_task_update_event__updates_status() -> None:
    """TaskUpdate should change task status and generate a PlanStep with status change."""
    analyzer = PlanAnalyzer()

    # Create a task first
    analyzer.analyze_task_create_event(
        {"subject": "Implement feature", "activeForm": "Implementing"},
        {"taskId": "1"},
        datetime.now(),
    )

    # Update status to in_progress
    analyzer.analyze_task_update_event(
        {"taskId": "1", "status": "in_progress"},
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 1
    assert evolution.final_todos[0].status == "in_progress"

    # The second plan step should show a status change
    assert len(evolution.plan_steps) >= 2
    last_step = evolution.plan_steps[-1]
    assert "Implement feature" in last_step.todos_status_changed
    assert last_step.todos_status_changed["Implement feature"] == ("pending", "in_progress")


def test_analyze_task_update_event__completes_task() -> None:
    """TaskUpdate to completed status should be reflected in plan evolution."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_create_event(
        {"subject": "Write tests", "activeForm": "Writing tests"},
        {"taskId": "1"},
        datetime.now(),
    )
    analyzer.analyze_task_update_event(
        {"taskId": "1", "status": "completed"},
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert evolution.total_todos_completed == 1
    assert evolution.final_todos[0].status == "completed"


def test_analyze_task_update_event__deletes_task() -> None:
    """TaskUpdate with status=deleted should remove the task."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_create_event(
        {"subject": "Temporary task", "activeForm": "Working"},
        {"taskId": "1"},
        datetime.now(),
    )
    analyzer.analyze_task_update_event(
        {"taskId": "1", "status": "deleted"},
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 0
    # The deletion should appear as a todo_removed in the plan step
    last_step = evolution.plan_steps[-1]
    assert "Temporary task" in last_step.todos_removed


def test_analyze_task_update_event__ignores_unknown_task_id() -> None:
    """TaskUpdate with unknown taskId should be ignored."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_update_event(
        {"taskId": "nonexistent", "status": "completed"},
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert evolution.total_plan_steps == 0


def test_analyze_task_create_event__uses_subject_as_fallback_key() -> None:
    """When tool_response has no taskId, subject should be used as key."""
    analyzer = PlanAnalyzer()

    analyzer.analyze_task_create_event(
        {"subject": "Fallback task", "activeForm": "Working"},
        {},  # No taskId in response
        datetime.now(),
    )

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 1
    assert evolution.final_todos[0].content == "Fallback task"


# --- OpenCode todo format tests ---


def test_analyze_todo_write_event__opencode_format_with_priority() -> None:
    """OpenCode todos have priority instead of activeForm — should parse correctly."""
    analyzer = PlanAnalyzer()

    tool_input = {
        "todos": [
            {"content": "Fix bug", "status": "pending", "priority": "high"},
            {"content": "Add tests", "status": "in_progress", "priority": "medium"},
        ]
    }
    analyzer.analyze_todo_write_event(tool_input, datetime.now())

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 2

    todo_by_content = {t.content: t for t in evolution.final_todos}
    assert todo_by_content["Fix bug"].priority == "high"
    assert todo_by_content["Fix bug"].activeForm == ""  # Not provided by OpenCode
    assert todo_by_content["Add tests"].status == "in_progress"


def test_analyze_todo_write_event__minimal_todo_only_content() -> None:
    """Todos with only content field should use defaults for status and other fields."""
    analyzer = PlanAnalyzer()

    tool_input = {
        "todos": [
            {"content": "Minimal todo"},
        ]
    }
    analyzer.analyze_todo_write_event(tool_input, datetime.now())

    evolution = analyzer.get_plan_evolution()
    assert len(evolution.final_todos) == 1
    assert evolution.final_todos[0].content == "Minimal todo"
    assert evolution.final_todos[0].status == "pending"
    assert evolution.final_todos[0].activeForm == ""
    assert evolution.final_todos[0].priority == ""


# --- Tool classification tests ---


def test_task_create_classified_as_implementation_tool() -> None:
    """TASK_CREATE should be in IMPLEMENTATION_TOOLS."""
    assert ToolType.TASK_CREATE in PlanAnalyzer.IMPLEMENTATION_TOOLS


def test_task_update_classified_as_implementation_tool() -> None:
    """TASK_UPDATE should be in IMPLEMENTATION_TOOLS."""
    assert ToolType.TASK_UPDATE in PlanAnalyzer.IMPLEMENTATION_TOOLS


def test_task_list_classified_as_search_tool() -> None:
    """TASK_LIST should be in SEARCH_TOOLS."""
    assert ToolType.TASK_LIST in PlanAnalyzer.SEARCH_TOOLS


def test_task_get_classified_as_search_tool() -> None:
    """TASK_GET should be in SEARCH_TOOLS."""
    assert ToolType.TASK_GET in PlanAnalyzer.SEARCH_TOOLS

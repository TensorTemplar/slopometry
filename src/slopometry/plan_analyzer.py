"""Plan evolution analysis for TodoWrite events."""

from datetime import datetime
from typing import Any

from slopometry.models import PlanEvolution, PlanStep, TodoItem, ToolType


class PlanAnalyzer:
    """Analyzes TodoWrite events to track plan evolution."""

    SEARCH_TOOLS = {
        ToolType.GREP,
        ToolType.GLOB,
        ToolType.WEB_SEARCH,
        ToolType.WEB_FETCH,
        ToolType.READ,
        ToolType.LS,
        ToolType.TASK,
        ToolType.TODO_READ,
        ToolType.NOTEBOOK_READ,
        ToolType.MCP_IDE_GET_DIAGNOSTICS,
        ToolType.MCP_IDE_GET_WORKSPACE_INFO,
        ToolType.MCP_IDE_GET_FILE_CONTENTS,
        ToolType.MCP_IDE_SEARCH_FILES,
        ToolType.MCP_FILESYSTEM_READ,
        ToolType.MCP_FILESYSTEM_LIST,
        ToolType.MCP_DATABASE_QUERY,
        ToolType.MCP_DATABASE_SCHEMA,
        ToolType.MCP_WEB_SCRAPE,
        ToolType.MCP_WEB_SEARCH,
        ToolType.MCP_GITHUB_GET_REPO,
        ToolType.MCP_GITHUB_LIST_ISSUES,
        ToolType.MCP_SLACK_LIST_CHANNELS,
    }

    IMPLEMENTATION_TOOLS = {
        ToolType.WRITE,
        ToolType.EDIT,
        ToolType.MULTI_EDIT,
        ToolType.BASH,
        ToolType.TODO_WRITE,
        ToolType.NOTEBOOK_EDIT,
        ToolType.EXIT_PLAN_MODE,
        ToolType.MCP_IDE_EXECUTE_CODE,
        ToolType.MCP_IDE_CREATE_FILE,
        ToolType.MCP_IDE_DELETE_FILE,
        ToolType.MCP_IDE_RENAME_FILE,
        ToolType.MCP_FILESYSTEM_WRITE,
        ToolType.MCP_GITHUB_CREATE_ISSUE,
        ToolType.MCP_SLACK_SEND_MESSAGE,
    }

    def __init__(self):
        """Initialize the plan analyzer."""
        self.previous_todos: dict[str, TodoItem] = {}
        self.plan_steps: list[PlanStep] = []
        self.step_number = 0
        self.events_since_last_todo = 0
        self.search_events_since_last_todo = 0
        self.implementation_events_since_last_todo = 0

    def analyze_todo_write_event(self, tool_input: dict[str, Any], timestamp: datetime) -> None:
        """Analyze a TodoWrite event and track plan evolution.

        Args:
            tool_input: The tool input containing todos
            timestamp: When the event occurred
        """
        todos_data = tool_input.get("todos", [])
        if not todos_data:
            return

        current_todos = {}
        for todo_data in todos_data:
            try:
                todo = TodoItem(**todo_data)
                current_todos[todo.id] = todo
            except Exception:
                continue

        plan_step = self._calculate_plan_step(current_todos, timestamp)
        if plan_step:
            self.plan_steps.append(plan_step)

        self.previous_todos = current_todos
        self.events_since_last_todo = 0
        self.search_events_since_last_todo = 0
        self.implementation_events_since_last_todo = 0

    def increment_event_count(self, tool_type: ToolType | None = None) -> None:
        """Increment the count of events since last TodoWrite."""
        self.events_since_last_todo += 1

        if tool_type:
            if tool_type in self.SEARCH_TOOLS:
                self.search_events_since_last_todo += 1
            elif tool_type in self.IMPLEMENTATION_TOOLS:
                self.implementation_events_since_last_todo += 1

    def get_plan_evolution(self) -> PlanEvolution:
        """Generate final plan evolution summary.

        Returns:
            PlanEvolution with aggregated statistics
        """
        if not self.plan_steps:
            return PlanEvolution()

        all_todo_contents = set()
        completed_todo_contents = set()

        for step in self.plan_steps:
            all_todo_contents.update(step.todos_added)
            for content, (_, new_status) in step.todos_status_changed.items():
                all_todo_contents.add(content)
                if new_status == "completed":
                    completed_todo_contents.add(content)

        for todo in self.previous_todos.values():
            all_todo_contents.add(todo.content)
            if todo.status == "completed":
                completed_todo_contents.add(todo.content)

        total_todos_created = len(all_todo_contents)
        total_todos_completed = len(completed_todo_contents)

        total_events = sum(step.events_in_step for step in self.plan_steps)
        average_events_per_step = total_events / len(self.plan_steps) if self.plan_steps else 0.0

        planning_efficiency = total_todos_completed / total_todos_created if total_todos_created > 0 else 0.0

        final_todo_count = len(self.previous_todos)

        total_search_events = sum(step.search_events for step in self.plan_steps)
        total_implementation_events = sum(step.implementation_events for step in self.plan_steps)
        overall_search_to_implementation_ratio = (
            total_search_events / total_implementation_events if total_implementation_events > 0 else 0.0
        )

        return PlanEvolution(
            total_plan_steps=len(self.plan_steps),
            total_todos_created=total_todos_created,
            total_todos_completed=total_todos_completed,
            average_events_per_step=average_events_per_step,
            plan_steps=self.plan_steps,
            final_todo_count=final_todo_count,
            planning_efficiency=planning_efficiency,
            total_search_events=total_search_events,
            total_implementation_events=total_implementation_events,
            overall_search_to_implementation_ratio=overall_search_to_implementation_ratio,
        )

    def _calculate_plan_step(self, current_todos: dict[str, TodoItem], timestamp: datetime) -> PlanStep | None:
        """Calculate the differences between previous and current todos.

        Args:
            current_todos: Current todo items by ID
            timestamp: When this step occurred

        Returns:
            PlanStep describing the changes, or None if first step
        """
        self.step_number += 1

        search_to_implementation_ratio = (
            self.search_events_since_last_todo / self.implementation_events_since_last_todo
            if self.implementation_events_since_last_todo > 0
            else 0.0
        )

        if not self.previous_todos:
            return PlanStep(
                step_number=self.step_number,
                events_in_step=self.events_since_last_todo,
                todos_added=[todo.content for todo in current_todos.values()],
                timestamp=timestamp,
                search_events=self.search_events_since_last_todo,
                implementation_events=self.implementation_events_since_last_todo,
                search_to_implementation_ratio=search_to_implementation_ratio,
            )

        previous_ids = set(self.previous_todos.keys())
        current_ids = set(current_todos.keys())

        new_todo_ids = current_ids - previous_ids
        todos_added = [current_todos[todo_id].content for todo_id in new_todo_ids]

        removed_todo_ids = previous_ids - current_ids
        todos_removed = [self.previous_todos[todo_id].content for todo_id in removed_todo_ids]

        common_ids = previous_ids & current_ids
        todos_status_changed = {}
        todos_content_changed = {}

        for todo_id in common_ids:
            prev_todo = self.previous_todos[todo_id]
            curr_todo = current_todos[todo_id]

            if prev_todo.status != curr_todo.status:
                todos_status_changed[curr_todo.content] = (prev_todo.status, curr_todo.status)

            if prev_todo.content != curr_todo.content:
                todos_content_changed[prev_todo.content] = (prev_todo.content, curr_todo.content)

        return PlanStep(
            step_number=self.step_number,
            events_in_step=self.events_since_last_todo,
            todos_added=todos_added,
            todos_removed=todos_removed,
            todos_status_changed=todos_status_changed,
            todos_content_changed=todos_content_changed,
            timestamp=timestamp,
            search_events=self.search_events_since_last_todo,
            implementation_events=self.implementation_events_since_last_todo,
            search_to_implementation_ratio=search_to_implementation_ratio,
        )

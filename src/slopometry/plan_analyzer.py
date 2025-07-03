"""Plan evolution analysis for TodoWrite events."""

import json
from datetime import datetime
from typing import Any

from .models import PlanEvolution, PlanStep, TodoItem


class PlanAnalyzer:
    """Analyzes TodoWrite events to track plan evolution."""
    
    def __init__(self):
        """Initialize the plan analyzer."""
        self.previous_todos: dict[str, TodoItem] = {}
        self.plan_steps: list[PlanStep] = []
        self.step_number = 0
        self.events_since_last_todo = 0
    
    def analyze_todo_write_event(self, tool_input: dict[str, Any], timestamp: datetime) -> None:
        """Analyze a TodoWrite event and track plan evolution.
        
        Args:
            tool_input: The tool input containing todos
            timestamp: When the event occurred
        """
        # Extract todos from tool input
        todos_data = tool_input.get("todos", [])
        if not todos_data:
            return
        
        # Parse current todos
        current_todos = {}
        for todo_data in todos_data:
            try:
                todo = TodoItem(**todo_data)
                current_todos[todo.id] = todo
            except Exception:
                # Skip malformed todo items
                continue
        
        # Calculate diff from previous todos
        plan_step = self._calculate_plan_step(current_todos, timestamp)
        if plan_step:
            self.plan_steps.append(plan_step)
        
        # Update state
        self.previous_todos = current_todos
        self.events_since_last_todo = 0
    
    def increment_event_count(self) -> None:
        """Increment the count of events since last TodoWrite."""
        self.events_since_last_todo += 1
    
    def get_plan_evolution(self) -> PlanEvolution:
        """Generate final plan evolution summary.
        
        Returns:
            PlanEvolution with aggregated statistics
        """
        if not self.plan_steps:
            return PlanEvolution()
        
        # Track unique todos and their final status
        all_todo_contents = set()
        completed_todo_contents = set()
        
        # Collect all todos ever mentioned across all steps
        for step in self.plan_steps:
            all_todo_contents.update(step.todos_added)
            # Track todos that reached completed status in this step
            for content, (_, new_status) in step.todos_status_changed.items():
                all_todo_contents.add(content)
                if new_status == "completed":
                    completed_todo_contents.add(content)
        
        # Also count todos that are currently completed in final state
        for todo in self.previous_todos.values():
            all_todo_contents.add(todo.content)
            if todo.status == "completed":
                completed_todo_contents.add(todo.content)
        
        total_todos_created = len(all_todo_contents)
        total_todos_completed = len(completed_todo_contents)
        
        total_events = sum(step.events_in_step for step in self.plan_steps)
        average_events_per_step = total_events / len(self.plan_steps) if self.plan_steps else 0.0
        
        planning_efficiency = (
            total_todos_completed / total_todos_created 
            if total_todos_created > 0 else 0.0
        )
        
        final_todo_count = len(self.previous_todos)
        
        return PlanEvolution(
            total_plan_steps=len(self.plan_steps),
            total_todos_created=total_todos_created,
            total_todos_completed=total_todos_completed,
            average_events_per_step=average_events_per_step,
            plan_steps=self.plan_steps,
            final_todo_count=final_todo_count,
            planning_efficiency=planning_efficiency
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
        
        # For the first TodoWrite, just record it without diff
        if not self.previous_todos:
            return PlanStep(
                step_number=self.step_number,
                events_in_step=self.events_since_last_todo,
                todos_added=[todo.content for todo in current_todos.values()],
                timestamp=timestamp
            )
        
        # Calculate differences
        previous_ids = set(self.previous_todos.keys())
        current_ids = set(current_todos.keys())
        
        # New todos (by ID)
        new_todo_ids = current_ids - previous_ids
        todos_added = [current_todos[todo_id].content for todo_id in new_todo_ids]
        
        # Removed todos (by ID)
        removed_todo_ids = previous_ids - current_ids
        todos_removed = [self.previous_todos[todo_id].content for todo_id in removed_todo_ids]
        
        # Changed todos (same ID, different content or status)
        common_ids = previous_ids & current_ids
        todos_status_changed = {}
        todos_content_changed = {}
        
        for todo_id in common_ids:
            prev_todo = self.previous_todos[todo_id]
            curr_todo = current_todos[todo_id]
            
            # Status change
            if prev_todo.status != curr_todo.status:
                todos_status_changed[curr_todo.content] = (prev_todo.status, curr_todo.status)
            
            # Content change (same ID, different content)
            if prev_todo.content != curr_todo.content:
                todos_content_changed[prev_todo.content] = (prev_todo.content, curr_todo.content)
        
        return PlanStep(
            step_number=self.step_number,
            events_in_step=self.events_since_last_todo,
            todos_added=todos_added,
            todos_removed=todos_removed,
            todos_status_changed=todos_status_changed,
            todos_content_changed=todos_content_changed,
            timestamp=timestamp
        )
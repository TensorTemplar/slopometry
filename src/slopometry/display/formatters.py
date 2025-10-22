"""Rich formatting utilities for displaying slopometry data."""

from rich.console import Console
from rich.table import Table

from slopometry.core.models import SessionStatistics

console = Console()


def display_session_summary(stats: SessionStatistics, session_id: str) -> None:
    """Display comprehensive session statistics with Rich formatting."""
    console.print(f"\n[bold]Session Statistics: {session_id}[/bold]")
    console.print(f"Start: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if stats.project:
        console.print(f"Project: [magenta]{stats.project.name}[/] ({stats.project.source.value})")

    if stats.end_time:
        duration = (stats.end_time - stats.start_time).total_seconds()
        console.print(f"Duration: {duration:.1f} seconds")

    if stats.working_directory:
        console.print(f"Working Directory: {stats.working_directory}")

    console.print(f"Total events: {stats.total_events}")

    # Events by type table
    if stats.events_by_type:
        _display_events_by_type_table(stats.events_by_type)

    # Tool usage table
    if stats.tool_usage:
        _display_tool_usage_table(stats.tool_usage)

    # Tool duration info
    if stats.average_tool_duration_ms:
        console.print(f"\nAverage tool duration: {stats.average_tool_duration_ms:.0f}ms")

    # Error count
    if stats.error_count > 0:
        console.print(f"[red]Errors: {stats.error_count}[/red]")

    # Git metrics
    if stats.initial_git_state and stats.initial_git_state.is_git_repo:
        _display_git_metrics(stats)

    # Complexity metrics
    if stats.complexity_metrics and stats.complexity_metrics.total_files_analyzed > 0:
        _display_complexity_metrics(stats)

    # Complexity delta
    if stats.complexity_delta:
        _display_complexity_delta(stats)

    # Plan evolution
    if stats.plan_evolution and stats.plan_evolution.total_plan_steps > 0:
        _display_plan_evolution(stats)


def _display_events_by_type_table(events_by_type: dict) -> None:
    """Display events by type table."""
    table = Table(title="Events by Type")
    table.add_column("Event Type", style="cyan")
    table.add_column("Count", justify="right")

    for event_type, count in sorted(events_by_type.items()):
        table.add_row(event_type.value, str(count))

    console.print(table)


def _display_tool_usage_table(tool_usage: dict) -> None:
    """Display tool usage table."""
    table = Table(title="Tool Usage")
    table.add_column("Tool", style="green")
    table.add_column("Count", justify="right")

    for tool_type, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        table.add_row(tool_type.value, str(count))

    console.print(table)


def _display_git_metrics(stats: SessionStatistics) -> None:
    """Display git metrics section."""
    console.print("\n[bold]Git Metrics[/bold]")
    console.print(f"Commits made: [green]{stats.commits_made}[/green]")

    if stats.initial_git_state.current_branch:
        console.print(f"Branch: {stats.initial_git_state.current_branch}")

    if stats.initial_git_state.has_uncommitted_changes:
        console.print("[yellow]Had uncommitted changes at start[/yellow]")

    if stats.final_git_state and stats.final_git_state.has_uncommitted_changes:
        console.print("[yellow]Has uncommitted changes at end[/yellow]")


def _display_complexity_metrics(stats: SessionStatistics) -> None:
    """Display complexity metrics section."""
    console.print("\n[bold]Complexity Metrics[/bold]")

    # Overview table
    overview_table = Table(title="Complexity Overview")
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", justify="right")

    metrics = stats.complexity_metrics
    overview_table.add_row("Files analyzed", str(metrics.total_files_analyzed))
    overview_table.add_row("[bold]Cyclomatic Complexity[/bold]", "")
    overview_table.add_row("  Total complexity", str(metrics.total_complexity))
    overview_table.add_row("  Average complexity", f"{metrics.average_complexity:.1f}")
    overview_table.add_row("  Max complexity", str(metrics.max_complexity))
    overview_table.add_row("  Min complexity", str(metrics.min_complexity))
    overview_table.add_row("[bold]Halstead Metrics[/bold]", "")
    overview_table.add_row("  Total volume", f"{metrics.total_volume:.1f}")
    overview_table.add_row("  Average volume", f"{metrics.average_volume:.1f}")
    overview_table.add_row("  Total difficulty", f"{metrics.total_difficulty:.1f}")
    overview_table.add_row("  Average difficulty", f"{metrics.average_difficulty:.1f}")
    overview_table.add_row("  Total effort", f"{metrics.total_effort:.1f}")
    overview_table.add_row("[bold]Maintainability Index[/bold]", "")
    overview_table.add_row("  Total MI", f"{metrics.total_mi:.1f}")
    overview_table.add_row("  Average MI", f"{metrics.average_mi:.1f} (higher is better)")

    console.print(overview_table)

    if metrics.files_by_complexity:
        files_table = Table(title="Files by Complexity")
        files_table.add_column("File", style="cyan")
        files_table.add_column("Complexity", justify="right")

        sorted_files = sorted(metrics.files_by_complexity.items(), key=lambda x: x[1], reverse=True)[:10]

        for file_path, complexity in sorted_files:
            files_table.add_row(file_path, str(complexity))

        console.print(files_table)

    if metrics.files_with_parse_errors:
        errors_table = Table(title="[yellow]Files with Parse Errors[/yellow]")
        errors_table.add_column("File", style="yellow")
        errors_table.add_column("Error", style="dim")

        for file_path, error_msg in metrics.files_with_parse_errors.items():
            errors_table.add_row(file_path, error_msg)

        console.print(errors_table)


def _display_complexity_delta(stats: SessionStatistics) -> None:
    """Display complexity delta section."""
    delta = stats.complexity_delta
    console.print("\n[bold]Complexity Delta (vs Session Start)[/bold]")

    changes_table = Table(title="Overall Changes")
    changes_table.add_column("Metric", style="cyan")
    changes_table.add_column("Change", justify="right")

    cc_color = (
        "green" if delta.total_complexity_change < 0 else "red" if delta.total_complexity_change > 0 else "yellow"
    )
    changes_table.add_row("[bold]Cyclomatic Complexity[/bold]", "")
    changes_table.add_row("  Total complexity", f"[{cc_color}]{delta.total_complexity_change:+d}[/{cc_color}]")
    changes_table.add_row("  Average complexity", f"[{cc_color}]{delta.avg_complexity_change:+.1f}[/{cc_color}]")

    changes_table.add_row("[bold]Halstead Metrics[/bold]", "")
    vol_color = "green" if delta.total_volume_change < 0 else "red" if delta.total_volume_change > 0 else "yellow"
    changes_table.add_row("  Total volume", f"[{vol_color}]{delta.total_volume_change:+.1f}[/{vol_color}]")
    changes_table.add_row("  Average volume", f"[{vol_color}]{delta.avg_volume_change:+.1f}[/{vol_color}]")

    diff_color = (
        "green" if delta.total_difficulty_change < 0 else "red" if delta.total_difficulty_change > 0 else "yellow"
    )
    changes_table.add_row("  Total difficulty", f"[{diff_color}]{delta.total_difficulty_change:+.1f}[/{diff_color}]")
    changes_table.add_row("  Average difficulty", f"[{diff_color}]{delta.avg_difficulty_change:+.1f}[/{diff_color}]")

    effort_color = "green" if delta.total_effort_change < 0 else "red" if delta.total_effort_change > 0 else "yellow"
    changes_table.add_row("  Total effort", f"[{effort_color}]{delta.total_effort_change:+.1f}[/{effort_color}]")

    # Maintainability Index changes (higher is better, so green for positive)
    changes_table.add_row("[bold]Maintainability Index[/bold]", "")
    mi_color = "red" if delta.total_mi_change < 0 else "green" if delta.total_mi_change > 0 else "yellow"
    changes_table.add_row("  Total MI", f"[{mi_color}]{delta.total_mi_change:+.1f}[/{mi_color}]")
    changes_table.add_row("  Average MI", f"[{mi_color}]{delta.avg_mi_change:+.1f}[/{mi_color}]")

    changes_table.add_row("[bold]File Changes[/bold]", "")
    file_color = "green" if delta.net_files_change < 0 else "red" if delta.net_files_change > 0 else "yellow"
    changes_table.add_row("  Net files change", f"[{file_color}]{delta.net_files_change:+d}[/{file_color}]")
    changes_table.add_row("  Files added", str(len(delta.files_added)))
    changes_table.add_row("  Files removed", str(len(delta.files_removed)))

    console.print(changes_table)

    if delta.files_added:
        files_added_table = Table(title="Files Added")
        files_added_table.add_column("File", style="green")
        for file_path in delta.files_added[:10]:
            files_added_table.add_row(file_path)
        if len(delta.files_added) > 10:
            files_added_table.add_row(f"... and {len(delta.files_added) - 10} more")
        console.print(files_added_table)

    if delta.files_removed:
        files_removed_table = Table(title="Files Removed")
        files_removed_table.add_column("File", style="red")
        for file_path in delta.files_removed[:10]:
            files_removed_table.add_row(file_path)
        if len(delta.files_removed) > 10:
            files_removed_table.add_row(f"... and {len(delta.files_removed) - 10} more")
        console.print(files_removed_table)

    if delta.files_changed:
        sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        if sorted_changes:
            file_changes_table = Table(title="File Complexity Changes")
            file_changes_table.add_column("File", style="cyan")
            file_changes_table.add_column("Previous", justify="right")
            file_changes_table.add_column("Current", justify="right")
            file_changes_table.add_column("Change", justify="right")

            for file_path, change in sorted_changes:
                current_complexity = stats.complexity_metrics.files_by_complexity.get(file_path, 0)
                previous_complexity = current_complexity - change

                change_color = "green" if change < 0 else "red"
                file_changes_table.add_row(
                    file_path,
                    str(previous_complexity),
                    str(current_complexity),
                    f"[{change_color}]{change:+d}[/{change_color}]",
                )

            console.print(file_changes_table)


def _display_plan_evolution(stats: SessionStatistics) -> None:
    """Display plan evolution section."""
    evolution = stats.plan_evolution
    console.print("\n[bold]Plan Evolution[/bold]")
    console.print(f"Total planning steps: {evolution.total_plan_steps}")
    console.print(f"Todos created: {evolution.total_todos_created}")
    console.print(f"Todos completed: {evolution.total_todos_completed}")
    console.print(f"Planning efficiency: {evolution.planning_efficiency:.1%}")
    console.print(f"Average events per step: {evolution.average_events_per_step:.1f}")
    console.print(f"Search events: {evolution.total_search_events}")
    console.print(f"Implementation events: {evolution.total_implementation_events}")
    console.print(f"Search/Implementation ratio: {evolution.overall_search_to_implementation_ratio:.2f}")

    if evolution.plan_steps:
        table = Table(title="Planning Steps")
        table.add_column("Step", style="cyan", width=4)
        table.add_column("Events", justify="right", width=6)
        table.add_column("S/I Ratio", justify="right", width=8)
        table.add_column("Changes", style="yellow")

        for step in evolution.plan_steps[:5]:
            changes = []
            if step.todos_added:
                changes.append(f"+{len(step.todos_added)} added")
            if step.todos_removed:
                changes.append(f"-{len(step.todos_removed)} removed")
            if step.todos_status_changed:
                changes.append(f"{len(step.todos_status_changed)} status changed")
            if step.todos_content_changed:
                changes.append(f"{len(step.todos_content_changed)} content changed")

            change_summary = ", ".join(changes) if changes else "Initial plan"
            ratio_display = f"{step.search_to_implementation_ratio:.2f}" if step.implementation_events > 0 else "N/A"
            table.add_row(str(step.step_number), str(step.events_in_step), ratio_display, change_summary)

        if len(evolution.plan_steps) > 5:
            table.add_row("...", "...", "...", f"... and {len(evolution.plan_steps) - 5} more steps")

        console.print(table)


def create_sessions_table(sessions_data: list[dict]) -> Table:
    """Create a Rich table for displaying session list."""
    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Project", style="magenta")
    table.add_column("Start Time", style="green")
    table.add_column("Events", justify="right")
    table.add_column("Tools Used", justify="right")

    for session_data in sessions_data:
        project_display = (
            f"{session_data['project_name']} ({session_data['project_source']})"
            if session_data["project_name"]
            else "N/A"
        )
        table.add_row(
            session_data["session_id"],
            project_display,
            session_data["start_time"],
            str(session_data["total_events"]),
            str(session_data["tools_used"]),
        )

    return table


def create_experiment_table(experiments_data: list[dict]) -> Table:
    """Create a Rich table for displaying experiment runs."""
    table = Table(title="Experiment Runs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Repository", style="magenta")
    table.add_column("Commits", style="blue")
    table.add_column("Start Time", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Status", style="bold")

    for exp_data in experiments_data:
        status_style = (
            "green" if exp_data["status"] == "completed" else "red" if exp_data["status"] == "failed" else "yellow"
        )

        table.add_row(
            exp_data["id"],
            exp_data["repository_name"],
            exp_data["commits_display"],
            exp_data["start_time"],
            exp_data["duration"],
            f"[{status_style}]{exp_data['status']}[/]",
        )

    return table


def create_user_story_entries_table(entries_data: list, count: int) -> Table:
    """Create a Rich table for displaying user story entries."""
    table = Table(title=f"Recent User Story Entries (showing {count})")
    table.add_column("Entry ID", style="blue", no_wrap=True, width=10)
    table.add_column("Date", style="cyan")
    table.add_column("Commits", style="green")
    table.add_column("Rating", style="yellow")
    table.add_column("Model", style="blue")
    table.add_column("Repository", style="magenta")

    for entry_data in entries_data:
        table.add_row(
            entry_data.entry_id,
            entry_data.date,
            entry_data.commits,
            entry_data.rating,
            entry_data.model,
            entry_data.repository,
        )

    return table


def create_nfp_objectives_table(objectives_data: list[dict]) -> Table:
    """Create a Rich table for displaying NFP objectives."""
    table = Table(title="NFP Objectives")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Commits", style="blue")
    table.add_column("Stories", style="green", justify="right")
    table.add_column("Complexity", style="yellow", justify="right")
    table.add_column("Created", style="dim")

    for obj_data in objectives_data:
        table.add_row(
            obj_data["id"],
            obj_data["title"],
            obj_data["commits"],
            str(obj_data["story_count"]),
            str(obj_data["complexity"]),
            obj_data["created_date"],
        )

    return table


def create_features_table(features_data: list[dict]) -> Table:
    """Create a Rich table for displaying detected features."""
    table = Table(title="Detected Features", show_lines=True)
    table.add_column("Feature ID", style="blue", no_wrap=True, width=10)
    table.add_column("Feature", style="cyan", no_wrap=False)
    table.add_column("Base → Head", style="green")
    table.add_column("Best Story", style="magenta", no_wrap=True, width=10)
    table.add_column("Merge Message", style="yellow", no_wrap=False)

    for feature_data in features_data:
        table.add_row(
            feature_data["feature_id"],
            feature_data["feature_message"],
            feature_data["commits_display"],
            feature_data["best_entry_id"],
            feature_data["merge_message"],
        )

    return table


def create_progress_history_table(progress_data: list[dict]) -> Table:
    """Create a Rich table for displaying experiment progress history."""
    table = Table(title="Progress History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("CLI Score", style="green", justify="right")
    table.add_column("Complexity", style="blue", justify="right")
    table.add_column("Halstead", style="yellow", justify="right")
    table.add_column("Maintainability", style="magenta", justify="right")

    for progress_row in progress_data:
        table.add_row(
            progress_row["timestamp"],
            progress_row["cli_score"],
            progress_row["complexity_score"],
            progress_row["halstead_score"],
            progress_row["maintainability_score"],
        )

    return table

"""Rich formatting utilities for displaying slopometry data."""

from rich.console import Console
from rich.table import Table

from slopometry.core.models import (
    ContextCoverage,
    CurrentChangesAnalysis,
    ImpactAssessment,
    ImpactCategory,
    PlanEvolution,
    RepoBaseline,
    SessionStatistics,
    StagedChangesAnalysis,
)

console = Console()


def display_session_summary(
    stats: SessionStatistics,
    session_id: str,
    baseline: RepoBaseline | None = None,
    assessment: ImpactAssessment | None = None,
) -> None:
    """Display comprehensive session statistics with Rich formatting.

    Args:
        stats: Session statistics to display
        session_id: The session identifier
        baseline: Optional repository baseline for comparison
        assessment: Optional impact assessment computed from baseline
    """
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

    if stats.events_by_type:
        _display_events_by_type_table(stats.events_by_type)

    if stats.tool_usage:
        _display_tool_usage_table(stats.tool_usage)

    if stats.average_tool_duration_ms:
        console.print(f"\nAverage tool duration: {stats.average_tool_duration_ms:.0f}ms")

    if stats.error_count > 0:
        console.print(f"[red]Errors: {stats.error_count}[/red]")

    if stats.initial_git_state and stats.initial_git_state.is_git_repo:
        _display_git_metrics(stats)

    if stats.complexity_metrics and stats.complexity_metrics.total_files_analyzed > 0:
        _display_complexity_metrics(stats)

    if stats.complexity_delta:
        _display_complexity_delta(stats, baseline, assessment)

    if stats.plan_evolution and stats.plan_evolution.total_plan_steps > 0:
        _display_work_summary(stats.plan_evolution)

    if stats.context_coverage and stats.context_coverage.files_edited:
        _display_context_coverage(stats.context_coverage)


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

    overview_table = Table(title="Complexity Overview")
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", justify="right")

    metrics = stats.complexity_metrics
    if not metrics:
        return

    overview_table.add_row("Files analyzed", str(metrics.total_files_analyzed))
    overview_table.add_row("[bold]Cyclomatic Complexity[/bold]", "")
    overview_table.add_row("  Total complexity", str(metrics.total_complexity))
    overview_table.add_row("  Average complexity", f"{metrics.average_complexity:.1f}")
    overview_table.add_row("  Max complexity", str(metrics.max_complexity))
    overview_table.add_row("  Min complexity", str(metrics.min_complexity))
    overview_table.add_row("[bold]Halstead Metrics[/bold]", "")
    overview_table.add_row("  Average volume", f"{metrics.average_volume:.1f}")
    overview_table.add_row("  Average effort", f"{metrics.average_effort:.2f}")
    overview_table.add_row("[bold]Maintainability Index[/bold]", "")
    overview_table.add_row("  Total MI", f"{metrics.total_mi:.1f}")
    overview_table.add_row("  File average MI", f"{metrics.average_mi:.1f} (higher is better)")

    overview_table.add_row("[bold]Token Usage[/bold]", "")
    overview_table.add_row("  Total Tokens", _format_token_count(metrics.total_tokens))
    overview_table.add_row("  Average Tokens", f"{metrics.average_tokens:.1f}")
    overview_table.add_row("  Max Tokens", str(metrics.max_tokens))

    overview_table.add_row("[bold]Python Quality[/bold]", "")
    overview_table.add_row("  Type Hint Coverage", f"{metrics.type_hint_coverage:.1f}%")
    overview_table.add_row("  Docstring Coverage", f"{metrics.docstring_coverage:.1f}%")

    any_color = "green" if metrics.any_type_percentage < 5 else "yellow" if metrics.any_type_percentage < 15 else "red"
    overview_table.add_row("  Any Type Usage", f"[{any_color}]{metrics.any_type_percentage:.1f}%[/{any_color}]")

    str_color = "green" if metrics.str_type_percentage < 20 else "yellow" if metrics.str_type_percentage < 40 else "red"
    overview_table.add_row("  str Type Usage", f"[{str_color}]{metrics.str_type_percentage:.1f}%[/{str_color}]")

    if metrics.test_coverage_percent is not None:
        cov_color = (
            "green"
            if metrics.test_coverage_percent >= 80
            else "yellow"
            if metrics.test_coverage_percent >= 50
            else "red"
        )
        source_info = f" [dim](from {metrics.test_coverage_source})[/dim]" if metrics.test_coverage_source else ""
        overview_table.add_row(
            "  Test Coverage", f"[{cov_color}]{metrics.test_coverage_percent:.1f}%[/{cov_color}]{source_info}"
        )
    else:
        overview_table.add_row("  Test Coverage", "[dim]N/A (run pytest first)[/dim]")

    overview_table.add_row(
        "  Deprecations", f"[yellow]{metrics.deprecation_count}[/yellow]" if metrics.deprecation_count > 0 else "0"
    )

    overview_table.add_row("[bold]Code Smells[/bold]", "")

    smell_color = "red" if metrics.orphan_comment_count > 0 else "green"
    overview_table.add_row("  Orphan Comments", f"[{smell_color}]{metrics.orphan_comment_count}[/{smell_color}]")

    smell_color = "red" if metrics.untracked_todo_count > 0 else "green"
    overview_table.add_row("  Untracked TODOs", f"[{smell_color}]{metrics.untracked_todo_count}[/{smell_color}]")

    smell_color = "red" if metrics.inline_import_count > 0 else "green"
    overview_table.add_row("  Inline Imports", f"[{smell_color}]{metrics.inline_import_count}[/{smell_color}]")

    smell_color = "red" if metrics.dict_get_with_default_count > 0 else "green"
    overview_table.add_row(
        "  .get() w/ Defaults", f"[{smell_color}]{metrics.dict_get_with_default_count}[/{smell_color}]"
    )

    smell_color = "red" if metrics.hasattr_getattr_count > 0 else "green"
    overview_table.add_row("  hasattr/getattr", f"[{smell_color}]{metrics.hasattr_getattr_count}[/{smell_color}]")

    smell_color = "red" if metrics.nonempty_init_count > 0 else "green"
    overview_table.add_row("  Non-empty __init__", f"[{smell_color}]{metrics.nonempty_init_count}[/{smell_color}]")

    smell_color = "red" if metrics.test_skip_count > 0 else "green"
    overview_table.add_row("  Test Skips", f"[{smell_color}]{metrics.test_skip_count}[/{smell_color}]")

    smell_color = "red" if metrics.swallowed_exception_count > 0 else "green"
    overview_table.add_row(
        "  Swallowed Exceptions", f"[{smell_color}]{metrics.swallowed_exception_count}[/{smell_color}]"
    )

    smell_color = "red" if metrics.type_ignore_count > 0 else "green"
    overview_table.add_row("  Type Ignores", f"[{smell_color}]{metrics.type_ignore_count}[/{smell_color}]")

    smell_color = "red" if metrics.dynamic_execution_count > 0 else "green"
    overview_table.add_row("  Dynamic Execution", f"[{smell_color}]{metrics.dynamic_execution_count}[/{smell_color}]")

    console.print(overview_table)

    # Display detailed code smells
    _display_code_smells_detailed(metrics)

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


def _display_complexity_delta(
    stats: SessionStatistics,
    baseline: RepoBaseline | None = None,
    assessment: ImpactAssessment | None = None,
) -> None:
    """Display complexity delta section with optional baseline comparison."""
    delta = stats.complexity_delta
    if not delta:
        return

    has_baseline = baseline is not None and assessment is not None

    title = "Complexity Delta (vs Session Start)"
    if has_baseline:
        title += f" - Baseline: {baseline.total_commits_analyzed} commits"
    console.print(f"\n[bold]{title}[/bold]")

    changes_table = Table(title="Overall Changes")
    changes_table.add_column("Metric", style="cyan")
    changes_table.add_column("Change", justify="right")
    if has_baseline:
        changes_table.add_column("vs Baseline", justify="right")

    # Average Cyclomatic Complexity - lower is better
    cc_color = "green" if delta.avg_complexity_change < 0 else "red" if delta.avg_complexity_change > 0 else "yellow"
    cc_baseline = _format_baseline_cell(assessment.cc_z_score, invert=True) if has_baseline else None
    changes_table.add_row(
        "Average Cyclomatic Complexity",
        f"[{cc_color}]{delta.avg_complexity_change:+.2f}[/{cc_color}]",
        cc_baseline if has_baseline else None,
    )

    # Average Effort - lower is better (complexity density)
    effort_color = "green" if delta.avg_effort_change < 0 else "red" if delta.avg_effort_change > 0 else "yellow"
    effort_baseline = _format_baseline_cell(assessment.effort_z_score, invert=True) if has_baseline else None
    changes_table.add_row(
        "Average Effort",
        f"[{effort_color}]{delta.avg_effort_change:+.2f}[/{effort_color}]",
        effort_baseline if has_baseline else None,
    )

    # Maintainability Index (per-file average) - higher is better
    mi_color = "red" if delta.avg_mi_change < 0 else "green" if delta.avg_mi_change > 0 else "yellow"
    mi_baseline = _format_baseline_cell(assessment.mi_z_score, invert=False) if has_baseline else None
    changes_table.add_row(
        "Maintainability (file avg)",
        f"[{mi_color}]{delta.avg_mi_change:+.2f}[/{mi_color}]",
        mi_baseline if has_baseline else None,
    )

    # Token Deltas
    token_color = "red" if delta.total_tokens_change > 0 else "green" if delta.total_tokens_change < 0 else "yellow"
    changes_table.add_row(
        "Total Tokens",
        f"[{token_color}]{delta.total_tokens_change:+d}[/{token_color}]",
        "" if has_baseline else None,
    )

    file_color = "green" if delta.net_files_change < 0 else "red" if delta.net_files_change > 0 else "yellow"
    changes_table.add_row(
        "Files changed",
        f"[{file_color}]{delta.net_files_change:+d}[/{file_color}] ({len(delta.files_added)} added, {len(delta.files_removed)} removed)",
        "" if has_baseline else None,
    )

    console.print(changes_table)

    if has_baseline and assessment:
        impact_color = _get_impact_color(assessment.impact_category)
        console.print(
            f"\n[bold]Overall Impact:[/bold] [{impact_color}]{assessment.impact_category.value.upper()}[/{impact_color}] "
            f"(score: {assessment.impact_score:+.2f})"
        )

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


def _format_token_count(tokens: int) -> str:
    """Format token count for display (e.g., 150000 -> '~150K').

    Args:
        tokens: Number of tokens

    Returns:
        Formatted string like '~150K' or '~1.5M'
    """
    if tokens >= 1_000_000:
        return f"~{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"~{tokens / 1_000:.0f}K"
    else:
        return f"~{tokens}"


def _display_work_summary(evolution: PlanEvolution) -> None:
    """Display compact work summary with task completion and work style."""
    console.print(
        f"\nTasks: {evolution.total_todos_completed}/{evolution.total_todos_created} completed ({evolution.planning_efficiency:.0%})"
    )
    impl_percentage = 100 - evolution.exploration_percentage

    # Build work style line with optional token info
    if evolution.token_usage and evolution.token_usage.total_tokens > 0:
        exploration_tokens = _format_token_count(evolution.token_usage.exploration_tokens)
        implementation_tokens = _format_token_count(evolution.token_usage.implementation_tokens)
        console.print(
            f"Work style: {evolution.exploration_percentage:.0f}% exploration ({exploration_tokens} tokens), "
            f"{impl_percentage:.0f}% implementation ({implementation_tokens} tokens)"
        )
    else:
        console.print(
            f"Work style: {evolution.exploration_percentage:.0f}% exploration, {impl_percentage:.0f}% implementation"
        )


def _display_context_coverage(coverage: ContextCoverage) -> None:
    """Display context coverage section showing what files were read before editing."""
    console.print("\n[bold]Context Coverage[/bold]")
    console.print(f"Files edited: {len(coverage.files_edited)}")
    console.print(f"Files read: {len(coverage.files_read)}")

    read_before_ratio = coverage.files_read_before_edit_ratio
    ratio_color = "green" if read_before_ratio >= 0.9 else "yellow" if read_before_ratio >= 0.7 else "red"
    console.print(f"Read before edit: [{ratio_color}]{read_before_ratio:.0%}[/{ratio_color}]")

    if coverage.file_coverage:
        table = Table(title="Coverage by Edited File")
        table.add_column("File", style="cyan", no_wrap=True, max_width=40)
        table.add_column("Read?", justify="center", width=5)
        table.add_column("Imports", justify="right", width=10)
        table.add_column("Dependents", justify="right", width=10)
        table.add_column("Tests", justify="right", width=8)

        for fc in coverage.file_coverage[:10]:
            read_mark = "[green]✓[/green]" if fc.was_read_before_edit else "[red]✗[/red]"

            imports_display = _format_coverage_ratio(len(fc.imports_read), len(fc.imports))
            dependents_display = _format_coverage_ratio(len(fc.dependents_read), len(fc.dependents))
            tests_display = _format_coverage_ratio(len(fc.test_files_read), len(fc.test_files))

            file_display = fc.file_path
            if len(file_display) > 40:
                file_display = "..." + file_display[-37:]

            table.add_row(file_display, read_mark, imports_display, dependents_display, tests_display)

        if len(coverage.file_coverage) > 10:
            table.add_row("...", "", "", "", f"({len(coverage.file_coverage) - 10} more)")

        console.print(table)

    if coverage.blind_spots:
        console.print(f"\n[yellow]Potential blind spots ({len(coverage.blind_spots)} files):[/yellow]")
        for blind_spot in coverage.blind_spots:
            console.print(f"  • {blind_spot}")


def _format_coverage_ratio(read: int, total: int) -> str:
    """Format a coverage ratio with color coding."""
    if total == 0:
        return "[dim]n/a[/dim]"
    ratio = read / total
    color = "green" if ratio >= 0.8 else "yellow" if ratio >= 0.5 else "red"
    return f"[{color}]{read}/{total}[/{color}]"


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


def display_staged_impact_analysis(analysis: StagedChangesAnalysis) -> None:
    """Display staged changes impact analysis with Rich formatting."""
    console.print("\n[bold]Staged Changes Impact Analysis[/bold]")
    console.print(f"Repository: {analysis.repository_path}")

    console.print(f"\n[bold]Staged Files ({len(analysis.staged_files)}):[/bold]")
    for f in analysis.staged_files[:5]:
        console.print(f"  [dim]- {f}[/dim]")
    if len(analysis.staged_files) > 5:
        console.print(f"  [dim]... and {len(analysis.staged_files) - 5} more[/dim]")

    baseline = analysis.baseline
    console.print(f"\n[bold]Repository Baseline ({baseline.total_commits_analyzed} commits):[/bold]")

    baseline_table = Table(show_header=True, header_style="bold")
    baseline_table.add_column("Metric", style="cyan")
    baseline_table.add_column("Mean Δ", justify="right")
    baseline_table.add_column("Std Dev", justify="right")
    baseline_table.add_column("Trend", justify="right")

    baseline_table.add_row(
        "Cyclomatic Complexity",
        f"{baseline.cc_delta_stats.mean:+.1f}",
        f"±{baseline.cc_delta_stats.std_dev:.1f}",
        _format_trend(baseline.cc_delta_stats.trend_coefficient, lower_is_better=True),
    )
    baseline_table.add_row(
        "Halstead Effort",
        f"{baseline.effort_delta_stats.mean:+.2f}",
        f"±{baseline.effort_delta_stats.std_dev:.2f}",
        _format_trend(baseline.effort_delta_stats.trend_coefficient, lower_is_better=True),
    )
    baseline_table.add_row(
        "Maintainability Index",
        f"{baseline.mi_delta_stats.mean:+.2f}",
        f"±{baseline.mi_delta_stats.std_dev:.2f}",
        _format_trend(baseline.mi_delta_stats.trend_coefficient, lower_is_better=False),
    )

    console.print(baseline_table)

    assessment = analysis.assessment
    console.print("\n[bold]Staged Changes Impact:[/bold]")

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Metric", style="cyan")
    impact_table.add_column("Change", justify="right")
    impact_table.add_column("Z-Score", justify="right")
    impact_table.add_column("Assessment", style="dim")

    cc_color = "green" if assessment.cc_z_score < 0 else "red" if assessment.cc_z_score > 0 else "yellow"
    impact_table.add_row(
        "Cyclomatic Complexity",
        f"[{cc_color}]{assessment.cc_delta:+.2f}[/{cc_color}]",
        f"{assessment.cc_z_score:+.2f}",
        _interpret_z_score(-assessment.cc_z_score),
    )

    effort_color = "green" if assessment.effort_z_score < 0 else "red" if assessment.effort_z_score > 0 else "yellow"
    impact_table.add_row(
        "Average Effort",
        f"[{effort_color}]{assessment.effort_delta:+.2f}[/{effort_color}]",
        f"{assessment.effort_z_score:+.2f}",
        _interpret_z_score(-assessment.effort_z_score),
    )

    mi_color = "green" if assessment.mi_z_score > 0 else "red" if assessment.mi_z_score < 0 else "yellow"
    impact_table.add_row(
        "Maintainability Index",
        f"[{mi_color}]{assessment.mi_delta:+.2f}[/{mi_color}]",
        f"{assessment.mi_z_score:+.2f}",
        _interpret_z_score(assessment.mi_z_score),
    )

    console.print(impact_table)

    category_styles = {
        ImpactCategory.SIGNIFICANT_IMPROVEMENT: (
            "bold green",
            "Your changes are significantly better than typical commits!",
        ),
        ImpactCategory.MINOR_IMPROVEMENT: ("green", "Your changes are slightly better than typical commits."),
        ImpactCategory.NEUTRAL: ("yellow", "Your changes are about average for this repository."),
        ImpactCategory.MINOR_DEGRADATION: ("red", "Your changes add slightly more complexity than typical commits."),
        ImpactCategory.SIGNIFICANT_DEGRADATION: (
            "bold red",
            "Your changes add significantly more complexity than typical commits.",
        ),
    }

    style, message = category_styles.get(
        assessment.impact_category,
        ("white", "Impact assessment complete."),
    )

    category_display = assessment.impact_category.value.replace("_", " ").upper()
    console.print(f"\n[bold]Overall Impact:[/bold] [{style}]{category_display}[/] ({assessment.impact_score:+.2f})")
    console.print(f"[dim]{message}[/dim]")


def _format_trend(trend_coefficient: float, lower_is_better: bool) -> str:
    """Format trend coefficient with arrow and color."""
    if abs(trend_coefficient) < 0.01:
        return "[yellow]→ stable[/yellow]"

    if lower_is_better:
        if trend_coefficient < 0:
            return f"[green]↘ {trend_coefficient:.3f}[/green]"
        else:
            return f"[red]↗ +{trend_coefficient:.3f}[/red]"
    else:
        if trend_coefficient > 0:
            return f"[green]↗ +{trend_coefficient:.3f}[/green]"
        else:
            return f"[red]↘ {trend_coefficient:.3f}[/red]"


def _interpret_z_score(normalized_z: float) -> str:
    """Interpret z-score for display (positive = good after normalization)."""
    if normalized_z > 1.5:
        return "significantly better than avg"
    elif normalized_z > 0.5:
        return "better than average"
    elif normalized_z > -0.5:
        return "about average"
    elif normalized_z > -1.5:
        return "worse than average"
    else:
        return "significantly worse than avg"


def _format_baseline_cell(z_score: float, invert: bool = False) -> str:
    """Format a baseline comparison cell with Z-score and color.

    Args:
        z_score: The raw Z-score value
        invert: If True, negative Z-score is good (for CC/Effort where lower is better)
    """
    # Normalize the z-score for display (positive = good)
    normalized_z = -z_score if invert else z_score

    if normalized_z > 1.0:
        color = "green"
        indicator = "↓↓" if invert else "↑↑"
    elif normalized_z > 0.5:
        color = "green"
        indicator = "↓" if invert else "↑"
    elif normalized_z > -0.5:
        color = "yellow"
        indicator = "→"
    elif normalized_z > -1.0:
        color = "red"
        indicator = "↑" if invert else "↓"
    else:
        color = "red"
        indicator = "↑↑" if invert else "↓↓"

    return f"[{color}]{indicator} z={z_score:+.1f}[/{color}]"


def _get_impact_color(category: ImpactCategory) -> str:
    """Get color for impact category."""
    match category:
        case ImpactCategory.SIGNIFICANT_IMPROVEMENT:
            return "green bold"
        case ImpactCategory.MINOR_IMPROVEMENT:
            return "green"
        case ImpactCategory.NEUTRAL:
            return "yellow"
        case ImpactCategory.MINOR_DEGRADATION:
            return "red"
        case ImpactCategory.SIGNIFICANT_DEGRADATION:
            return "red bold"


def display_current_impact_analysis(analysis: CurrentChangesAnalysis) -> None:
    """Display uncommitted changes impact analysis with Rich formatting."""
    console.print("\n[bold]Uncommitted Changes Impact Analysis[/bold]")

    console.print(f"Repository: {analysis.repository_path}")

    # Dropped list of changed files as requested by user to reduce noise

    display_baseline_comparison(
        baseline=analysis.baseline,
        assessment=analysis.assessment,
        title="Uncommitted Changes Impact",
    )

    console.print("\n[bold]Token Impact:[/bold]")
    token_table = Table(show_header=True)
    token_table.add_column("Metric", style="cyan")
    token_table.add_column("Value", justify="right")

    # Calculate delta for just the uncommitted changes if needed, but delta is in the baseline comparison table usually?
    # No, baseline comparison table shows complexity/effort/MI deltas.
    # We want to show raw token counts here as well.

    token_table.add_row("Tokens in Edited Files", _format_token_count(analysis.changed_files_tokens))
    token_table.add_row("Tokens in Blind Spots", _format_token_count(analysis.blind_spot_tokens))
    token_table.add_row(
        "Complete Picture Context Size", f"[bold]{_format_token_count(analysis.complete_picture_context_size)}[/bold]"
    )
    console.print(token_table)

    console.print("\n[bold]Current Code Quality:[/bold]")
    quality_table = Table(show_header=True)
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value", justify="right")

    metrics = analysis.current_metrics
    quality_table.add_row("Type Hint Coverage", f"{metrics.type_hint_coverage:.1f}%")
    quality_table.add_row("Docstring Coverage", f"{metrics.docstring_coverage:.1f}%")

    any_color = "green" if metrics.any_type_percentage < 5 else "yellow" if metrics.any_type_percentage < 15 else "red"
    quality_table.add_row("Any Type Usage", f"[{any_color}]{metrics.any_type_percentage:.1f}%[/{any_color}]")

    str_color = "green" if metrics.str_type_percentage < 20 else "yellow" if metrics.str_type_percentage < 40 else "red"
    quality_table.add_row("str Type Usage", f"[{str_color}]{metrics.str_type_percentage:.1f}%[/{str_color}]")

    if metrics.test_coverage_percent is not None:
        cov_color = (
            "green"
            if metrics.test_coverage_percent >= 80
            else "yellow"
            if metrics.test_coverage_percent >= 50
            else "red"
        )
        source_info = f" [dim](from {metrics.test_coverage_source})[/dim]" if metrics.test_coverage_source else ""
        quality_table.add_row(
            "Test Coverage", f"[{cov_color}]{metrics.test_coverage_percent:.1f}%[/{cov_color}]{source_info}"
        )
    else:
        quality_table.add_row("Test Coverage", "[dim]N/A (run pytest first)[/dim]")

    dep_style = "red" if metrics.deprecation_count > 0 else "green"
    quality_table.add_row("Deprecations", f"[{dep_style}]{metrics.deprecation_count}[/{dep_style}]")

    console.print(quality_table)

    # Coverage for Edited Files
    if analysis.filtered_coverage:
        console.print("\n[bold]Code Coverage for Edited Files:[/bold]")
        cov_table = Table(show_header=True)
        cov_table.add_column("File", style="cyan")
        cov_table.add_column("Coverage", justify="right")

        for fname in sorted(analysis.filtered_coverage.keys()):
            pct = analysis.filtered_coverage[fname]
            color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            cov_table.add_row(fname, f"[{color}]{pct:.1f}%[/{color}]")
        console.print(cov_table)

    if analysis.blind_spots:
        console.print(f"\n[yellow]Potential blind spots ({len(analysis.blind_spots)} files):[/yellow]")
        # Show all blind spots as requested
        for blind_spot in analysis.blind_spots:
            console.print(f"  • {blind_spot}")

    # Display detailed code smells for changed files only
    filter_set = set(analysis.changed_files) if analysis.changed_files else None
    _display_code_smells_detailed(metrics, filter_files=filter_set)


def _display_code_smells_detailed(metrics, filter_files: set[str] | None = None) -> None:
    """Display a detailed table of code smells with complete file lists.

    Args:
        metrics: The metrics object containing code smell data.
        filter_files: Optional set of file paths (relative). If provided, only
                     code smells in these files will be displayed.
    """

    # helper to filter and check if we have smells
    def get_filtered_files(files: list[str]) -> list[str]:
        if not filter_files:
            return files
        return [f for f in files if f in filter_files]

    # Check if we have any smells to display after filtering
    has_smells = False

    # We need to compute filtered lists first to know if we should show the table
    orphan_files = get_filtered_files(metrics.orphan_comment_files)
    todo_files = get_filtered_files(metrics.untracked_todo_files)
    import_files = get_filtered_files(metrics.inline_import_files)
    get_files = get_filtered_files(metrics.dict_get_with_default_files)
    getattr_files = get_filtered_files(metrics.hasattr_getattr_files)
    init_files = get_filtered_files(metrics.nonempty_init_files)
    test_skip_files = get_filtered_files(metrics.test_skip_files)
    swallowed_exception_files = get_filtered_files(metrics.swallowed_exception_files)
    type_ignore_files = get_filtered_files(metrics.type_ignore_files)
    dynamic_execution_files = get_filtered_files(metrics.dynamic_execution_files)

    if (
        orphan_files
        or todo_files
        or import_files
        or get_files
        or getattr_files
        or init_files
        or test_skip_files
        or swallowed_exception_files
        or type_ignore_files
        or dynamic_execution_files
    ):
        has_smells = True

    if not has_smells:
        return

    console.print("\n[bold]Code Smells Details:[/bold]")
    if filter_files:
        console.print("[dim]Showing smells for changed files only[/dim]")

    table = Table(show_header=True, show_lines=True)
    table.add_column("Smell Type", style="cyan", width=20)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Affected Files", style="dim")

    def add_smell_row(label: str, files: list[str]) -> None:
        count = len(files)
        if count == 0:
            return

        color = "red"
        count_str = f"[{color}]{count}[/{color}]"

        # Sort files for consistent display
        files_display = "\n".join(sorted(files))

        table.add_row(label, count_str, files_display)

    add_smell_row("Orphan Comments", orphan_files)
    add_smell_row("Untracked TODOs", todo_files)
    add_smell_row("Inline Imports", import_files)
    add_smell_row(".get() w/ Defaults", get_files)
    add_smell_row("hasattr/getattr", getattr_files)
    add_smell_row("Non-empty __init__", init_files)
    add_smell_row("Test Skips", test_skip_files)
    add_smell_row("Swallowed Exceptions", swallowed_exception_files)
    add_smell_row("Type Ignores", type_ignore_files)
    add_smell_row("Dynamic Execution", dynamic_execution_files)

    console.print(table)


def display_baseline_comparison(
    baseline: RepoBaseline,
    assessment: ImpactAssessment,
    title: str = "Impact Assessment",
) -> None:
    """Display baseline comparison with impact assessment.

    This is a shared formatter used by current-impact, analyze-commits,
    solo latest, and stop hook feedback.
    """

    console.print(f"\n[bold]Repository Baseline ({baseline.total_commits_analyzed} commits):[/bold]")

    baseline_table = Table(show_header=True, header_style="bold")
    baseline_table.add_column("Metric", style="cyan")
    baseline_table.add_column("Mean Δ", justify="right")
    baseline_table.add_column("Std Dev", justify="right")
    baseline_table.add_column("Trend", justify="right")

    baseline_table.add_row(
        "Cyclomatic Complexity",
        f"{baseline.cc_delta_stats.mean:+.1f}",
        f"±{baseline.cc_delta_stats.std_dev:.1f}",
        _format_trend(baseline.cc_delta_stats.trend_coefficient, lower_is_better=True),
    )
    baseline_table.add_row(
        "Halstead Effort",
        f"{baseline.effort_delta_stats.mean:+.2f}",
        f"±{baseline.effort_delta_stats.std_dev:.2f}",
        _format_trend(baseline.effort_delta_stats.trend_coefficient, lower_is_better=True),
    )
    baseline_table.add_row(
        "Maintainability Index",
        f"{baseline.mi_delta_stats.mean:+.2f}",
        f"±{baseline.mi_delta_stats.std_dev:.2f}",
        _format_trend(baseline.mi_delta_stats.trend_coefficient, lower_is_better=False),
    )

    console.print(baseline_table)

    console.print(f"\n[bold]{title}:[/bold]")

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Metric", style="cyan")
    impact_table.add_column("Change", justify="right")
    impact_table.add_column("Z-Score", justify="right")
    impact_table.add_column("Assessment", style="dim")

    cc_color = "green" if assessment.cc_z_score < 0 else "red" if assessment.cc_z_score > 0 else "yellow"
    impact_table.add_row(
        "Cyclomatic Complexity",
        f"[{cc_color}]{assessment.cc_delta:+.2f}[/{cc_color}]",
        f"{assessment.cc_z_score:+.2f}",
        _interpret_z_score(-assessment.cc_z_score),
    )

    effort_color = "green" if assessment.effort_z_score < 0 else "red" if assessment.effort_z_score > 0 else "yellow"
    impact_table.add_row(
        "Average Effort",
        f"[{effort_color}]{assessment.effort_delta:+.2f}[/{effort_color}]",
        f"{assessment.effort_z_score:+.2f}",
        _interpret_z_score(-assessment.effort_z_score),
    )

    mi_color = "green" if assessment.mi_z_score > 0 else "red" if assessment.mi_z_score < 0 else "yellow"
    impact_table.add_row(
        "Maintainability Index",
        f"[{mi_color}]{assessment.mi_delta:+.2f}[/{mi_color}]",
        f"{assessment.mi_z_score:+.2f}",
        _interpret_z_score(assessment.mi_z_score),
    )

    console.print(impact_table)

    category_styles = {
        ImpactCategory.SIGNIFICANT_IMPROVEMENT: ("bold green", "Significantly better than typical commits!"),
        ImpactCategory.MINOR_IMPROVEMENT: ("green", "Slightly better than typical commits."),
        ImpactCategory.NEUTRAL: ("yellow", "About average for this repository."),
        ImpactCategory.MINOR_DEGRADATION: ("red", "Slightly more complexity than typical commits."),
        ImpactCategory.SIGNIFICANT_DEGRADATION: ("bold red", "Significantly more complexity than typical commits."),
    }

    style, message = category_styles.get(
        assessment.impact_category,
        ("white", "Impact assessment complete."),
    )

    category_display = assessment.impact_category.value.replace("_", " ").upper()
    console.print(f"\n[bold]Overall Impact:[/bold] [{style}]{category_display}[/] ({assessment.impact_score:+.2f})")
    console.print(f"[dim]{message}[/dim]")


def display_baseline_comparison_compact(
    baseline: RepoBaseline,
    assessment: ImpactAssessment,
) -> str:
    """Return a compact string for baseline comparison (for hook feedback)."""
    lines = []
    lines.append(f"Repository Baseline ({baseline.total_commits_analyzed} commits):")

    cc_sign = "↓" if assessment.cc_z_score < 0 else "↑" if assessment.cc_z_score > 0 else "→"
    cc_quality = "good" if assessment.cc_z_score < 0 else "above avg" if assessment.cc_z_score > 0 else "avg"
    lines.append(f"  CC: {assessment.cc_delta:+.2f} (Z: {assessment.cc_z_score:+.2f} {cc_sign} {cc_quality})")

    effort_sign = "↓" if assessment.effort_z_score < 0 else "↑" if assessment.effort_z_score > 0 else "→"
    effort_quality = (
        "good" if assessment.effort_z_score < 0 else "above avg" if assessment.effort_z_score > 0 else "avg"
    )
    lines.append(
        f"  Effort: {assessment.effort_delta:+.2f} (Z: {assessment.effort_z_score:+.2f} {effort_sign} {effort_quality})"
    )

    mi_sign = "↑" if assessment.mi_z_score > 0 else "↓" if assessment.mi_z_score < 0 else "→"
    mi_quality = "good" if assessment.mi_z_score > 0 else "below avg" if assessment.mi_z_score < 0 else "avg"
    lines.append(f"  MI: {assessment.mi_delta:+.2f} (Z: {assessment.mi_z_score:+.2f} {mi_sign} {mi_quality})")

    category_display = assessment.impact_category.value.replace("_", " ").upper()
    lines.append(f"Session Impact: {category_display} ({assessment.impact_score:+.2f})")

    return "\n".join(lines)

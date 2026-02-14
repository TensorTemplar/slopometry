"""Rich formatting utilities for displaying slopometry data."""

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from slopometry.core.models.baseline import (
    BaselineStrategy,
    ImplementationComparison,
    SmellAdvantage,
    ZScoreInterpretation,
)
from slopometry.core.models.display import ExperimentDisplayData, LeaderboardEntry, NFPObjectiveDisplayData
from slopometry.core.models.experiment import ProgressDisplayData
from slopometry.core.models.hook import HookEventType, ToolType
from slopometry.core.models.session import CompactEvent, TokenUsage
from slopometry.core.models.smell import SMELL_REGISTRY, SmellCategory, get_smell_label, get_smells_by_category
from slopometry.core.models.user_story import UserStoryDisplayData
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


def truncate_path(path: str, max_width: int = 55) -> str:
    """Truncate a file path keeping prefix and basename, ellipsizing the middle.

    Displays paths like: src/.../utils/helpers.py
    - Keeps the first component (e.g., 'src/', 'test/')
    - Keeps the basename and immediate parent if possible
    - Truncates the middle with '...'

    Args:
        path: The file path to truncate
        max_width: Maximum width for the displayed path

    Returns:
        Truncated path string
    """
    if len(path) <= max_width:
        return path

    p = Path(path)
    parts = p.parts

    if len(parts) <= 2:
        return path[: max_width - 3] + "..."

    prefix = parts[0]
    if prefix == "/":
        prefix = "/" + parts[1] if len(parts) > 1 else "/"
        tail_parts = parts[-2:] if len(parts) > 3 else parts[-1:]
    else:
        tail_parts = parts[-2:] if len(parts) > 2 else parts[-1:]

    tail = str(Path(*tail_parts))

    ellipsis = "/.../"
    available = max_width - len(prefix) - len(ellipsis) - len(tail)

    if available < 0:
        basename = p.name
        remaining = max_width - len(prefix) - len(ellipsis) - len(basename)
        if remaining >= 0:
            return f"{prefix}{ellipsis}{basename}"
        return path[: max_width - 3] + "..."

    return f"{prefix}{ellipsis}{tail}"


from slopometry.core.models.baseline import (
    CrossProjectComparison,
    CurrentChangesAnalysis,
    GalenMetrics,
    ImpactAssessment,
    ImpactCategory,
    QPEScore,
    RepoBaseline,
    StagedChangesAnalysis,
)
from slopometry.core.models.complexity import ExtendedComplexityMetrics
from slopometry.core.models.hook import AnalysisSource
from slopometry.core.models.session import ContextCoverage, PlanEvolution, SessionStatistics
from slopometry.display.console import console


def _calculate_galen_metrics_from_baseline(
    baseline: RepoBaseline | None, current_tokens: int | None
) -> GalenMetrics | None:
    """Calculate Galen metrics from commit history token growth.

    Uses the baseline's commit date range and oldest commit token count
    to calculate productivity over the analyzed commit period.
    """
    if baseline is None:
        return None

    if not baseline.oldest_commit_date or not baseline.newest_commit_date:
        return None

    if baseline.oldest_commit_tokens is None or current_tokens is None:
        return None

    time_delta = baseline.newest_commit_date - baseline.oldest_commit_date
    period_days = time_delta.total_seconds() / 86400

    if period_days <= 0:
        return None

    tokens_changed = current_tokens - baseline.oldest_commit_tokens

    return GalenMetrics.calculate(tokens_changed=tokens_changed, period_days=period_days)


def _display_microsoft_ngmi_alert(galen_metrics: GalenMetrics) -> None:
    """Display the Microsoft NGMI alert header when below 1 Galen productivity.

    Shows a prominent alert with the Galen Rate and whether the developer is
    on track to hit 1 Galen (1M tokens/month) by end of month.
    """
    rate = galen_metrics.galen_rate
    rate_color = _color_for_galen_rate(rate)

    now = datetime.now()
    days_remaining = 30 - now.day

    tokens_per_day = galen_metrics.tokens_per_day
    projected_monthly = tokens_per_day * 30

    console.print()
    console.print("[bold]" + "=" * 60 + "[/bold]")

    if rate >= 1.0:
        console.print(f"[green bold]GALEN RATE: {rate:.2f} Galens - You're on track![/green bold]")
        console.print(f"[green]Projected: ~{projected_monthly / 1_000_000:.2f}M tokens/month[/green]")
    else:
        tokens_needed = galen_metrics.tokens_per_day_to_reach_one_galen or 0
        console.print(f"[{rate_color} bold]GALEN RATE: {rate:.2f} Galens[/{rate_color} bold]")
        console.print(f"[yellow]Need +{tokens_needed:,.0f} tokens/day to hit 1 Galen[/yellow]")

        if days_remaining > 0:
            tokens_still_needed = 1_000_000 - (projected_monthly * (now.day / 30))
            if tokens_still_needed > tokens_per_day * days_remaining:
                console.print("[red bold]You're NGMI![/red bold]")
            else:
                console.print(f"[yellow]{days_remaining} days left this month - pick up the pace![/yellow]")
        else:
            console.print("[red bold]Month is over - you're NGMI![/red bold]")

    console.print("[bold]" + "=" * 60 + "[/bold]")


def display_session_summary(
    stats: SessionStatistics,
    session_id: str,
    baseline: RepoBaseline | None = None,
    assessment: ImpactAssessment | None = None,
    show_smell_files: bool = False,
    show_file_details: bool = False,
) -> None:
    """Display comprehensive session statistics with Rich formatting.

    Args:
        stats: Session statistics to display
        session_id: The session identifier
        baseline: Optional repository baseline for comparison
        assessment: Optional impact assessment computed from baseline
        show_smell_files: Show files affected by each code smell
        show_file_details: Show full file lists in delta sections
    """
    # Calculate Galen metrics from commit history (not session duration)
    current_tokens = stats.complexity_metrics.total_tokens if stats.complexity_metrics else None
    baseline_galen_metrics = _calculate_galen_metrics_from_baseline(baseline, current_tokens)

    if settings.enable_working_at_microsoft and baseline_galen_metrics:
        _display_microsoft_ngmi_alert(baseline_galen_metrics)

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

    if stats.plan_evolution and (
        stats.plan_evolution.total_plan_steps > 0
        or stats.plan_evolution.plan_files_created > 0
        or stats.plan_evolution.final_todos
    ):
        _display_plan_info(stats.plan_evolution)

    if stats.events_by_type:
        _display_events_by_type_table(stats.events_by_type)

    if stats.tool_usage:
        _display_tool_usage_table(stats.tool_usage)

    if stats.compact_events:
        _display_compact_events(stats.compact_events)

    token_usage = stats.plan_evolution.token_usage if stats.plan_evolution else None
    if token_usage or stats.compact_events:
        _display_token_impact(token_usage, stats.compact_events)

    if stats.average_tool_duration_ms:
        console.print(f"\nAverage tool duration: {stats.average_tool_duration_ms:.0f}ms")

    if stats.error_count > 0:
        console.print(f"[red]Errors: {stats.error_count}[/red]")

    if stats.initial_git_state and stats.initial_git_state.is_git_repo:
        _display_git_metrics(stats)

    if stats.complexity_metrics and stats.complexity_metrics.total_files_analyzed > 0:
        _display_complexity_metrics(stats, galen_metrics=baseline_galen_metrics, show_smell_files=show_smell_files)

    if stats.complexity_delta:
        _display_complexity_delta(stats, baseline, assessment, show_file_details=show_file_details)

    if stats.context_coverage and stats.context_coverage.files_edited:
        _display_context_coverage(stats.context_coverage, show_file_details=show_file_details)


def _display_events_by_type_table(events_by_type: dict[HookEventType, int]) -> None:
    """Display events by type table."""
    table = Table(title="Events by Type")
    table.add_column("Event Type", style="cyan")
    table.add_column("Count", justify="right")

    for event_type, count in sorted(events_by_type.items()):
        table.add_row(event_type.value, str(count))

    console.print(table)


def _display_tool_usage_table(tool_usage: dict[ToolType, int]) -> None:
    """Display tool usage table."""
    table = Table(title="Tool Usage")
    table.add_column("Tool", style="green")
    table.add_column("Count", justify="right")

    for tool_type, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        table.add_row(tool_type.value, str(count))

    console.print(table)


def _display_compact_events(compact_events: list[CompactEvent]) -> None:
    """Display compact events table.

    Args:
        compact_events: List of compact events from the session
    """
    if not compact_events:
        return

    console.print(f"\n[bold]Compacts ({len(compact_events)})[/bold]")
    table = Table()
    table.add_column("Date/Time", style="cyan")
    table.add_column("Trigger", style="yellow")
    table.add_column("Pre-Tokens", justify="right")
    table.add_column("Version", style="dim")
    table.add_column("Branch", style="magenta")
    table.add_column("Line", justify="right", style="dim")

    for compact in compact_events:
        table.add_row(
            compact.timestamp.strftime("%m-%d %H:%M"),
            compact.trigger,
            _format_token_count(compact.pre_tokens),
            compact.version,
            compact.git_branch,
            str(compact.line_number),
        )
    console.print(table)


def _display_token_impact(token_usage: TokenUsage | None, compact_events: list[CompactEvent]) -> None:
    """Display token impact section with exploration/implementation breakdown.

    Args:
        token_usage: Token usage metrics from plan evolution
        compact_events: List of compact events for calculating 'without compact'
    """
    if not token_usage:
        return

    console.print("\n[bold]Token Impact:[/bold]")
    token_table = Table(show_header=True)
    token_table.add_column("Metric", style="cyan")
    token_table.add_column("Value", justify="right")

    token_table.add_row("Changeset Tokens", _format_token_count(token_usage.total_tokens))
    token_table.add_row("Exploration Tokens", _format_token_count(token_usage.exploration_tokens))
    token_table.add_row("Implementation Tokens", _format_token_count(token_usage.implementation_tokens))

    if token_usage.subagent_tokens > 0:
        token_table.add_row("Subagent Tokens", _format_token_count(token_usage.subagent_tokens))

    if compact_events:
        tokens_without_compact = sum(c.pre_tokens for c in compact_events)
        token_table.add_row(
            "[yellow]Tokens Without Compact[/yellow]",
            f"[yellow]{_format_token_count(tokens_without_compact)}[/yellow]",
        )

    console.print(token_table)


def _display_git_metrics(stats: SessionStatistics) -> None:
    """Display git metrics section."""
    console.print("\n[bold]Git Metrics[/bold]")
    console.print(f"Commits made: [green]{stats.commits_made}[/green]")

    if stats.initial_git_state and stats.initial_git_state.current_branch:
        console.print(f"Branch: {stats.initial_git_state.current_branch}")

    if stats.initial_git_state and stats.initial_git_state.has_uncommitted_changes:
        console.print("[yellow]Had uncommitted changes at start[/yellow]")

    if stats.final_git_state and stats.final_git_state.has_uncommitted_changes:
        console.print("[yellow]Has uncommitted changes at end[/yellow]")


def _display_complexity_metrics(
    stats: SessionStatistics,
    galen_metrics: GalenMetrics | None = None,
    show_smell_files: bool = False,
) -> None:
    """Display complexity metrics section."""
    metrics = stats.complexity_metrics
    if not metrics:
        return

    console.print("\n[bold]Complexity Metrics[/bold]")
    _display_complexity_overview(metrics)

    if galen_metrics:
        _display_galen_rate(galen_metrics)

    if show_smell_files:
        _display_code_smells_detailed(metrics)
    else:
        smell_files = metrics.get_smell_files()
        has_smell_files = any(smell_files.values())
        if has_smell_files:
            console.print("\n[dim]Run with --smell-details to see affected files[/dim]")

    if metrics.files_by_effort:
        _display_files_by_effort(metrics)

    if show_smell_files:
        _display_smell_offenders(metrics)

    if metrics.files_with_parse_errors:
        _display_parse_errors(metrics)


def _display_complexity_overview(metrics: ExtendedComplexityMetrics) -> None:
    """Display the main complexity metrics overview table."""
    overview_table = Table()
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", justify="right")

    overview_table.add_row("Files analyzed", str(metrics.total_files_analyzed))
    overview_table.add_row("[bold]Cyclomatic Complexity[/bold]", "")
    overview_table.add_row("  Average complexity", f"{metrics.average_complexity:.1f}")
    overview_table.add_row("  Max complexity", str(metrics.max_complexity))
    overview_table.add_row("  Min complexity", str(metrics.min_complexity))
    overview_table.add_row("[bold]Halstead Metrics[/bold]", "")
    overview_table.add_row("  Average volume", f"{metrics.average_volume:.1f}")
    overview_table.add_row("  Average effort", f"{metrics.average_effort:.2f}")
    if metrics.files_by_effort:
        max_effort_file, max_effort = max(metrics.files_by_effort.items(), key=lambda x: x[1])
        overview_table.add_row("  Max effort", f"{max_effort:,.0f} ({truncate_path(max_effort_file, max_width=30)})")
    overview_table.add_row("[bold]Maintainability Index[/bold]", "")
    overview_table.add_row("  Total MI", f"{metrics.total_mi:.1f}")
    overview_table.add_row("  File average MI", f"{metrics.average_mi:.1f} (higher is better)")
    if metrics.files_by_mi:
        min_mi_file, min_mi = min(metrics.files_by_mi.items(), key=lambda x: x[1])
        overview_table.add_row("  Min MI", f"{min_mi:.1f} ({truncate_path(min_mi_file, max_width=30)})")

    overview_table.add_row("[bold]Token Usage[/bold]", "")
    overview_table.add_row("  Total Tokens", _format_token_count(metrics.total_tokens))
    overview_table.add_row("  Average Tokens", f"{metrics.average_tokens:.1f}")
    overview_table.add_row("  Max Tokens", str(metrics.max_tokens))

    overview_table.add_row("[bold]Python Quality[/bold]", "")
    overview_table.add_row("  Type Hint Coverage", f"{metrics.type_hint_coverage:.1f}%")
    overview_table.add_row("  Docstring Coverage", f"{metrics.docstring_coverage:.1f}%")

    any_color = _color_for_inverted_threshold(metrics.any_type_percentage, 5, 15)
    overview_table.add_row("  Any Type Usage", f"[{any_color}]{metrics.any_type_percentage:.1f}%[/{any_color}]")

    str_color = _color_for_inverted_threshold(metrics.str_type_percentage, 20, 40)
    overview_table.add_row("  str Type Usage", f"[{str_color}]{metrics.str_type_percentage:.1f}%[/{str_color}]")

    if metrics.test_coverage_percent is not None:
        cov_color = _color_for_coverage(metrics.test_coverage_percent)
        source_info = f" [dim](from {metrics.test_coverage_source})[/dim]" if metrics.test_coverage_source else ""
        overview_table.add_row(
            "  Test Coverage", f"[{cov_color}]{metrics.test_coverage_percent:.1f}%[/{cov_color}]{source_info}"
        )
    else:
        overview_table.add_row("  Test Coverage", "[dim]N/A (run pytest first)[/dim]")

    overview_table.add_row(
        "  Deprecations (excl. runtime)",
        f"[yellow]{metrics.deprecation_count}[/yellow]" if metrics.deprecation_count > 0 else "0",
    )

    overview_table.add_row("[bold]Code Smells[/bold]", "")
    smell_dict = {s.name: s.count for s in metrics.get_smells()}

    for category in [SmellCategory.GENERAL, SmellCategory.PYTHON]:
        category_label = "General" if category == SmellCategory.GENERAL else "Python"
        overview_table.add_row(f"  [dim]{category_label}[/dim]", "")
        for defn in get_smells_by_category(category):
            count = smell_dict.get(defn.internal_name, 0)
            smell_color = "red" if count > 0 else "green"
            overview_table.add_row(f"    {defn.label}", f"[{smell_color}]{count}[/{smell_color}]")

    console.print(overview_table)


def _display_files_by_effort(metrics: ExtendedComplexityMetrics) -> None:
    """Display files sorted by Halstead effort."""
    files_table = Table(title="Files by Halstead Effort")
    files_table.add_column("File", style="cyan", no_wrap=True, max_width=55)
    files_table.add_column("Effort", justify="right", width=12)

    sorted_files = sorted(metrics.files_by_effort.items(), key=lambda x: x[1], reverse=True)[:10]

    for file_path, effort in sorted_files:
        files_table.add_row(truncate_path(file_path, max_width=55), f"{effort:,.0f}")

    console.print(files_table)


def _display_smell_offenders(metrics: ExtendedComplexityMetrics) -> None:
    """Display files with the most code smells."""
    smell_files = metrics.get_smell_files()
    file_smell_counts: Counter[str] = Counter()
    for files in smell_files.values():
        for f in files:
            file_smell_counts[f] += 1

    if file_smell_counts:
        offenders_table = Table(title="Worst Smell Offenders")
        offenders_table.add_column("File", style="cyan", no_wrap=True, max_width=55)
        offenders_table.add_column("Smell Types", justify="right", width=12)

        sorted_offenders = sorted(file_smell_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, count in sorted_offenders:
            offenders_table.add_row(truncate_path(file_path, max_width=55), str(count))

        console.print(offenders_table)


def _display_parse_errors(metrics: ExtendedComplexityMetrics) -> None:
    """Display files that could not be parsed."""
    errors_table = Table(title="[yellow]Files with Parse Errors[/yellow]")
    errors_table.add_column("File", style="yellow", no_wrap=True, max_width=55)
    errors_table.add_column("Error", style="dim")

    for file_path, error_msg in metrics.files_with_parse_errors.items():
        errors_table.add_row(truncate_path(file_path, max_width=55), error_msg)

    console.print(errors_table)


def _display_complexity_delta(
    stats: SessionStatistics,
    baseline: RepoBaseline | None = None,
    assessment: ImpactAssessment | None = None,
    show_file_details: bool = False,
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

    changes_table = Table()
    changes_table.add_column("Metric", style="cyan")
    changes_table.add_column("Change", justify="right")
    if has_baseline:
        changes_table.add_column("vs Baseline", justify="right")

    cc_color = _color_for_positive_negative(delta.avg_complexity_change)
    cc_baseline = _format_baseline_cell(assessment.cc_z_score, invert=True) if has_baseline else None
    changes_table.add_row(
        "Average Cyclomatic Complexity",
        f"[{cc_color}]{delta.avg_complexity_change:+.2f}[/{cc_color}]",
        cc_baseline if has_baseline else None,
    )

    effort_color = _color_for_positive_negative(delta.avg_effort_change)
    effort_baseline = _format_baseline_cell(assessment.effort_z_score, invert=True) if has_baseline else None
    changes_table.add_row(
        "Average Effort",
        f"[{effort_color}]{delta.avg_effort_change:+.2f}[/{effort_color}]",
        effort_baseline if has_baseline else None,
    )

    mi_color = _color_for_positive_negative(delta.avg_mi_change, invert=True)
    mi_baseline = _format_baseline_cell(assessment.mi_z_score, invert=False) if has_baseline else None
    changes_table.add_row(
        "Maintainability (file avg)",
        f"[{mi_color}]{delta.avg_mi_change:+.2f}[/{mi_color}]",
        mi_baseline if has_baseline else None,
    )

    token_color = _color_for_positive_negative(delta.total_tokens_change, invert=True)
    changes_table.add_row(
        "Total Tokens",
        f"[{token_color}]{delta.total_tokens_change:+d}[/{token_color}]",
        "" if has_baseline else None,
    )

    file_color = _color_for_positive_negative(delta.net_files_change)
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

    if show_file_details:
        if delta.files_added:
            files_added_table = Table(title="Files Added")
            files_added_table.add_column("File", style="green", no_wrap=True, max_width=55)
            for file_path in delta.files_added[:10]:
                files_added_table.add_row(truncate_path(file_path, max_width=55))
            if len(delta.files_added) > 10:
                files_added_table.add_row(f"... and {len(delta.files_added) - 10} more")
            console.print(files_added_table)

        if delta.files_removed:
            files_removed_table = Table(title="Files Removed")
            files_removed_table.add_column("File", style="red", no_wrap=True, max_width=55)
            for file_path in delta.files_removed[:10]:
                files_removed_table.add_row(truncate_path(file_path, max_width=55))
            if len(delta.files_removed) > 10:
                files_removed_table.add_row(f"... and {len(delta.files_removed) - 10} more")
            console.print(files_removed_table)

        if delta.files_effort_changed:
            sorted_changes = sorted(delta.files_effort_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

            if sorted_changes:
                file_changes_table = Table(title="File Effort Changes")
                file_changes_table.add_column("File", style="cyan", no_wrap=True, max_width=55)
                file_changes_table.add_column("Previous", justify="right", width=12)
                file_changes_table.add_column("Current", justify="right", width=12)
                file_changes_table.add_column("Change", justify="right", width=12)

                files_by_effort = stats.complexity_metrics.files_by_effort if stats.complexity_metrics else {}
                for file_path, change in sorted_changes:
                    if file_path not in files_by_effort:
                        continue
                    current_effort = files_by_effort[file_path]
                    previous_effort = current_effort - change

                    change_color = _color_for_positive_negative(change)
                    file_changes_table.add_row(
                        truncate_path(file_path, max_width=55),
                        f"{previous_effort:,.0f}",
                        f"{current_effort:,.0f}",
                        f"[{change_color}]{change:+,.0f}[/{change_color}]",
                    )

                console.print(file_changes_table)
    else:
        if delta.files_effort_changed:
            sorted_changes = sorted(delta.files_effort_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            if sorted_changes:
                file_changes_table = Table(title="Top File Effort Changes")
                file_changes_table.add_column("File", style="cyan", no_wrap=True, max_width=55)
                file_changes_table.add_column("Change", justify="right", width=12)

                for file_path, change in sorted_changes:
                    change_color = _color_for_positive_negative(change)
                    file_changes_table.add_row(
                        truncate_path(file_path, max_width=55), f"[{change_color}]{change:+,.0f}[/{change_color}]"
                    )

                console.print(file_changes_table)

        has_file_data = delta.files_added or delta.files_removed or len(delta.files_effort_changed) > 3
        if has_file_data:
            console.print("\n[dim]Run with --file-details to see all file changes[/dim]")


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


def _color_for_threshold(value: float, green_threshold: float, yellow_threshold: float) -> str:
    """Return color for values where higher is better.

    Args:
        value: The value to color
        green_threshold: threshold for green (inclusive)
        yellow_threshold: threshold for yellow (inclusive)

    Returns:
        "green", "yellow", or "red"
    """
    if value >= green_threshold:
        return "green"
    elif value >= yellow_threshold:
        return "yellow"
    else:
        return "red"


def _color_for_inverted_threshold(value: float, green_threshold: float, yellow_threshold: float) -> str:
    """Return color for values where lower is better.

    Args:
        value: The value to color
        green_threshold: threshold for green (exclusive - lower is better)
        yellow_threshold: threshold for yellow (exclusive)

    Returns:
        "green", "yellow", or "red"
    """
    if value < green_threshold:
        return "green"
    elif value < yellow_threshold:
        return "yellow"
    else:
        return "red"


def _color_for_positive_negative(value: float, invert: bool = False) -> str:
    """Return color for delta values where negative is good (or positive if invert=True).

    Args:
        value: The delta value
        invert: If True, positive is good (e.g., for Maintainability Index)

    Returns:
        "green" if good change, "red" if bad change, "yellow" if zero
    """
    if invert:
        return "green" if value > 0 else "red" if value < 0 else "yellow"
    else:
        return "green" if value < 0 else "red" if value > 0 else "yellow"


def _color_for_galen_rate(rate: float) -> str:
    """Return color for Galen productivity rate."""
    return _color_for_threshold(rate, 1.0, 0.5)


def _color_for_qpe(qpe: float) -> str:
    """Return color for QPE score."""
    return _color_for_threshold(qpe, 0.6, 0.4)


def _color_for_coverage(pct: float) -> str:
    """Return color for coverage percentage."""
    return _color_for_threshold(pct, 80, 50)


def _color_for_context_coverage(ratio: float) -> str:
    """Return color for context coverage ratio."""
    return _color_for_threshold(ratio, 0.9, 0.7)


def _color_for_read_before_edit(ratio: float) -> str:
    """Return color for files read before edit ratio."""
    return _color_for_threshold(ratio, 0.8, 0.5)


def _color_for_advantage(adv: float) -> str:
    """Return color for GRPO advantage."""
    if adv > 0.01:
        return "green"
    elif adv < -0.01:
        return "red"
    else:
        return "yellow"


def _display_galen_rate(galen_metrics: GalenMetrics, title: str = "Galen Rate") -> None:
    """Display Galen Rate metrics with color coding.

    1 Galen = 1 million code tokens per developer per month.
    Based on Microsoft's C++ to Rust migration productivity benchmark.

    Color scheme:
      - >= 1.0 Galens: green (on track or ahead)
      - 0.5 - 1.0 Galens: yellow (making progress)
      - < 0.5 Galens: red (falling behind)
    """
    console.print(f"\n[bold]{title}:[/bold]")
    galen_table = Table(show_header=True)
    galen_table.add_column("Metric", style="cyan")
    galen_table.add_column("Value", justify="right")

    sign = "+" if galen_metrics.tokens_changed >= 0 else ""
    galen_table.add_row("Token Delta", f"{sign}{galen_metrics.tokens_changed:,}")

    if galen_metrics.period_days >= 1:
        galen_table.add_row("Analysis Period", f"{galen_metrics.period_days:.1f} days")
    else:
        hours = galen_metrics.period_days * 24
        galen_table.add_row("Analysis Period", f"{hours:.1f} hours")

    rate = galen_metrics.galen_rate
    rate_color = _color_for_galen_rate(rate)
    galen_table.add_row("Galen Rate", f"[{rate_color}]{rate:.2f} Galens[/{rate_color}]")

    if galen_metrics.tokens_per_day_to_reach_one_galen is not None:
        needed = galen_metrics.tokens_per_day_to_reach_one_galen
        galen_table.add_row("Tokens/day to 1 Galen", f"[yellow]+{needed:,.0f}/day needed[/yellow]")

    console.print(galen_table)


def _display_plan_info(evolution: PlanEvolution) -> None:
    """Display plan and todo information section.

    Shows:
    - TodoWrite usage counts and completion rate
    - Plan file paths with existence check and clickable links
    - Final todo items with status indicators

    Args:
        evolution: The plan evolution data containing todos and plan files
    """
    console.print("\n[bold]Plans & Todos[/bold]")

    if evolution.total_todos_created > 0:
        efficiency_color = (
            "green"
            if evolution.planning_efficiency >= 0.8
            else "yellow"
            if evolution.planning_efficiency >= 0.5
            else "red"
        )
        console.print(
            f"Tasks: {evolution.total_todos_completed}/{evolution.total_todos_created} completed "
            f"([{efficiency_color}]{evolution.planning_efficiency:.0%}[/{efficiency_color}])"
        )

    if evolution.plan_files_created > 0:
        console.print(f"\n[bold]Plan Files ({evolution.plan_files_created}):[/bold]")
        for plan_path in evolution.plan_file_paths:
            expanded_path = Path(plan_path).expanduser()
            truncated = truncate_path(plan_path, max_width=60)
            status = _get_file_status(expanded_path)
            if status == "exists":
                console.print(f"  [link=file://{expanded_path}]{truncated}[/link] [green](exists)[/green]")
            elif status == "empty":
                console.print(f"  [link=file://{expanded_path}]{truncated}[/link] [dim](empty)[/dim]")
            else:
                console.print(f"  {truncated} [dim](deleted)[/dim]")

    if evolution.final_todos:
        console.print(f"\n[bold]Final Todos ({len(evolution.final_todos)}):[/bold]")
        for todo in evolution.final_todos:
            status_indicator = _get_todo_status_indicator(todo.status)
            console.print(f"  {status_indicator} {todo.content}")


def _get_todo_status_indicator(status: str) -> str:
    """Get the status indicator for a todo item.

    Args:
        status: The todo status ('completed', 'in_progress', or 'pending')

    Returns:
        Formatted status indicator string with color
    """
    if status == "completed":
        return "[green]✓[/green]"
    elif status == "in_progress":
        return "[yellow]→[/yellow]"
    else:  # pending
        return "[dim]○[/dim]"


def _get_file_status(file_path: Path) -> str:
    """Check if a file exists and has content.

    Args:
        file_path: Path to the file to check

    Returns:
        'exists' if file has content, 'empty' if file is empty/whitespace-only, 'deleted' if missing
    """
    if not file_path.exists():
        return "deleted"
    if file_path.stat().st_size == 0:
        return "empty"
    # For small files, check if content is just whitespace or empty JSON
    if file_path.stat().st_size < 100:
        content = file_path.read_text().strip()
        if not content or content in ("", "{}", "[]", "null"):
            return "empty"
    return "exists"


def _display_work_summary(evolution: PlanEvolution) -> None:
    """Display compact work summary with task completion and work style."""
    console.print(
        f"\nTasks: {evolution.total_todos_completed}/{evolution.total_todos_created} completed ({evolution.planning_efficiency:.0%})"
    )
    impl_percentage = 100 - evolution.exploration_percentage

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


def _display_context_coverage(coverage: ContextCoverage, show_file_details: bool = False) -> None:
    """Display context coverage section showing what files were read before editing."""
    console.print("\n[bold]Context Coverage[/bold]")
    console.print(f"Files edited: {len(coverage.files_edited)}")
    console.print(f"Files read: {len(coverage.files_read)}")

    read_before_ratio = coverage.files_read_before_edit_ratio
    ratio_color = _color_for_context_coverage(read_before_ratio)
    console.print(f"Read before edit: [{ratio_color}]{read_before_ratio:.0%}[/{ratio_color}]")

    if coverage.file_coverage:
        table = Table(title="Coverage by Edited File")
        table.add_column("File", style="cyan", no_wrap=True, max_width=55)
        table.add_column("Read?", justify="center", width=6)
        table.add_column("Imports", justify="right", width=10)
        table.add_column("Dependents", justify="right", width=10)
        table.add_column("Tests", justify="right", width=8)

        for fc in coverage.file_coverage[:10]:
            read_mark = "[green]✓[/green]" if fc.was_read_before_edit else "[red]✗[/red]"

            imports_display = _format_coverage_ratio(len(fc.imports_read), len(fc.imports))
            dependents_display = _format_coverage_ratio(len(fc.dependents_read), len(fc.dependents))
            tests_display = _format_coverage_ratio(len(fc.test_files_read), len(fc.test_files))

            file_display = truncate_path(fc.file_path, max_width=55)

            table.add_row(file_display, read_mark, imports_display, dependents_display, tests_display)

        if len(coverage.file_coverage) > 10:
            table.add_row("...", "", "", "", f"({len(coverage.file_coverage) - 10} more)")

        console.print(table)

    if coverage.blind_spots:
        if show_file_details:
            console.print(f"\n[yellow]Potential blind spots ({len(coverage.blind_spots)} files):[/yellow]")
            for blind_spot in coverage.blind_spots:
                console.print(f"  • {truncate_path(blind_spot, max_width=70)}")
        else:
            console.print(
                f"\n[dim]Potential blind spots: {len(coverage.blind_spots)} files (use --file-details to list)[/dim]"
            )


def _format_coverage_ratio(read: int, total: int) -> str:
    """Format a coverage ratio with color coding."""
    if total == 0:
        return "[dim]n/a[/dim]"
    ratio = read / total
    color = _color_for_read_before_edit(ratio)
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


def create_experiment_table(experiments_data: list[ExperimentDisplayData]) -> Table:
    """Create a Rich table for displaying experiment runs."""
    table = Table(title="Experiment Runs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Repository", style="magenta")
    table.add_column("Commits", style="blue")
    table.add_column("Start Time", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Status", style="bold")

    for exp_data in experiments_data:
        status_style = "green" if exp_data.status == "completed" else "red" if exp_data.status == "failed" else "yellow"

        table.add_row(
            exp_data.id,
            exp_data.repository_name,
            exp_data.commits_display,
            exp_data.start_time,
            exp_data.duration,
            f"[{status_style}]{exp_data.status}[/]",
        )

    return table


def create_user_story_entries_table(entries_data: list[UserStoryDisplayData], count: int) -> Table:
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


def create_nfp_objectives_table(objectives_data: list[NFPObjectiveDisplayData]) -> Table:
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
            obj_data.id,
            obj_data.title,
            obj_data.commits,
            str(obj_data.story_count),
            str(obj_data.complexity),
            obj_data.created_date,
        )

    return table


def create_features_table(features_data: list[dict[str, Any]]) -> Table:
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


def create_progress_history_table(progress_data: list[ProgressDisplayData]) -> Table:
    """Create a Rich table for displaying experiment progress history."""
    table = Table(title="Progress History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("CLI Score", style="green", justify="right")
    table.add_column("Complexity", style="blue", justify="right")
    table.add_column("Halstead", style="yellow", justify="right")
    table.add_column("Maintainability", style="magenta", justify="right")

    for progress_row in progress_data:
        table.add_row(
            progress_row.timestamp,
            progress_row.cli_score,
            progress_row.complexity_score,
            progress_row.halstead_score,
            progress_row.maintainability_score,
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

    cc_color = _color_for_positive_negative(assessment.cc_z_score)
    impact_table.add_row(
        "Cyclomatic Complexity",
        f"[{cc_color}]{assessment.cc_delta:+.2f}[/{cc_color}]",
        f"{assessment.cc_z_score:+.2f}",
        _interpret_z_score(-assessment.cc_z_score),
    )

    effort_color = _color_for_positive_negative(assessment.effort_z_score)
    impact_table.add_row(
        "Average Effort",
        f"[{effort_color}]{assessment.effort_delta:+.2f}[/{effort_color}]",
        f"{assessment.effort_z_score:+.2f}",
        _interpret_z_score(-assessment.effort_z_score),
    )

    mi_color = _color_for_positive_negative(assessment.mi_z_score, invert=True)
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
    """Interpret z-score for display (positive = good after normalization).

    Uses verbose mode from ZScoreInterpretation for more nuanced output.
    """
    return ZScoreInterpretation.from_z_score(normalized_z, verbose=True).value


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
        case _:
            return "white"


def display_current_impact_analysis(
    analysis: CurrentChangesAnalysis,
    compact_events: list[CompactEvent] | None = None,
    show_file_details: bool = False,
) -> None:
    """Display changes impact analysis with Rich formatting.

    Args:
        analysis: The current changes analysis to display
        compact_events: Optional list of compact events from session transcript
        show_file_details: Whether to show detailed file lists (blind spots)
    """
    if analysis.source == AnalysisSource.PREVIOUS_COMMIT:
        title = f"Previous Commit Impact Analysis ({analysis.base_commit_sha}..{analysis.analyzed_commit_sha})"
        impact_title = "Previous Commit Impact"
    else:
        title = "Uncommitted Changes Impact Analysis"
        impact_title = "Uncommitted Changes Impact"

    console.print(f"\n[bold]{title}[/bold]")

    console.print(f"Repository: {analysis.repository_path}")

    display_baseline_comparison(
        baseline=analysis.baseline,
        assessment=analysis.assessment,
        title=impact_title,
        smell_advantages=analysis.smell_advantages or None,
    )

    console.print("\n[bold]Token Impact:[/bold]")
    token_table = Table(show_header=True)
    token_table.add_column("Metric", style="cyan")
    token_table.add_column("Value", justify="right")

    token_table.add_row("Tokens in Edited Files", _format_token_count(analysis.changed_files_tokens))
    token_table.add_row("Tokens in Blind Spots", _format_token_count(analysis.blind_spot_tokens))
    token_table.add_row(
        "Complete Picture Context Size", f"[bold]{_format_token_count(analysis.complete_picture_context_size)}[/bold]"
    )

    if compact_events:
        tokens_without_compact = sum(c.pre_tokens for c in compact_events)
        token_table.add_row(
            "[yellow]Tokens Without Compact[/yellow]",
            f"[yellow]{_format_token_count(tokens_without_compact)}[/yellow]",
        )

    console.print(token_table)

    if analysis.galen_metrics:
        _display_galen_rate(analysis.galen_metrics)

    console.print("\n[bold]Current Code Quality:[/bold]")
    quality_table = Table(show_header=True)
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value", justify="right")

    metrics = analysis.current_metrics
    quality_table.add_row("Type Hint Coverage", f"{metrics.type_hint_coverage:.1f}%")
    quality_table.add_row("Docstring Coverage", f"{metrics.docstring_coverage:.1f}%")

    any_color = _color_for_inverted_threshold(metrics.any_type_percentage, 5, 15)
    quality_table.add_row("Any Type Usage", f"[{any_color}]{metrics.any_type_percentage:.1f}%[/{any_color}]")

    str_color = _color_for_inverted_threshold(metrics.str_type_percentage, 20, 40)
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
    quality_table.add_row("Deprecations (excl. runtime)", f"[{dep_style}]{metrics.deprecation_count}[/{dep_style}]")

    console.print(quality_table)

    if analysis.filtered_coverage:
        console.print("\n[bold]Code Coverage for Edited Files:[/bold]")
        cov_table = Table(show_header=True)
        cov_table.add_column("File", style="cyan", no_wrap=True, max_width=55)
        cov_table.add_column("Coverage", justify="right", width=10)

        for fname in sorted(analysis.filtered_coverage.keys()):
            pct = analysis.filtered_coverage[fname]
            color = _color_for_coverage(pct)
            cov_table.add_row(truncate_path(fname, max_width=55), f"[{color}]{pct:.1f}%[/{color}]")
        console.print(cov_table)

    if analysis.blind_spots:
        if show_file_details:
            console.print(f"\n[yellow]Potential blind spots ({len(analysis.blind_spots)} files):[/yellow]")
            for blind_spot in analysis.blind_spots:
                console.print(f"  • {truncate_path(blind_spot, max_width=70)}")
        else:
            console.print(
                f"\n[dim]Potential blind spots: {len(analysis.blind_spots)} files (use --file-details to list)[/dim]"
            )

    filter_set = set(analysis.changed_files) if analysis.changed_files else None
    _display_code_smells_detailed(metrics, filter_files=filter_set)


def _display_code_smells_detailed(metrics: ExtendedComplexityMetrics, filter_files: set[str] | None = None) -> None:
    """Display a detailed table of code smells with complete file lists.

    Args:
        metrics: The metrics object containing code smell data.
        filter_files: Optional set of file paths (relative). If provided, only
                     code smells in these files will be displayed.
    """

    def get_filtered_files(files: list[str]) -> list[str]:
        if not filter_files:
            return files
        return [f for f in files if f in filter_files]

    smell_files = metrics.get_smell_files()
    smell_data_by_category: dict[SmellCategory, list[tuple[str, list[str]]]] = {
        SmellCategory.GENERAL: [],
        SmellCategory.PYTHON: [],
    }
    for defn in SMELL_REGISTRY.values():
        files = get_filtered_files(smell_files[defn.internal_name])
        if files:
            smell_data_by_category[defn.category].append((defn.label, files))

    has_smells = any(smell_data_by_category[cat] for cat in smell_data_by_category)
    if not has_smells:
        return

    console.print("\n[bold]Code Smells Details:[/bold]")
    if filter_files:
        console.print("[dim]Showing smells for changed files only[/dim]")

    table = Table(show_header=True, show_lines=True)
    table.add_column("Smell Type", style="cyan", width=22)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Affected Files", style="dim", max_width=55)

    for category in [SmellCategory.GENERAL, SmellCategory.PYTHON]:
        smell_data = smell_data_by_category[category]
        if not smell_data:
            continue
        category_label = "General" if category == SmellCategory.GENERAL else "Python"
        table.add_row(f"[bold dim]{category_label}[/bold dim]", "", "")
        for label, files in smell_data:
            count_str = f"[red]{len(files)}[/red]"
            files_display = "\n".join(truncate_path(f, max_width=55) for f in sorted(files))
            table.add_row(f"  {label}", count_str, files_display)

    console.print(table)


def display_baseline_comparison(
    baseline: RepoBaseline,
    assessment: ImpactAssessment,
    title: str = "Impact Assessment",
    smell_advantages: list["SmellAdvantage"] | None = None,
) -> None:
    """Display baseline comparison with impact assessment.

    This is a shared formatter used by current-impact, analyze-commits,
    solo latest, and stop hook feedback.

    Args:
        baseline: Repository baseline for context
        assessment: Impact assessment with z-scores
        title: Section title
        smell_advantages: Optional per-smell advantage breakdown from QPE comparison
    """

    strategy_info = ""
    if baseline.strategy:
        strategy_label = baseline.strategy.resolved.value.replace("_", "-")
        if baseline.strategy.requested == BaselineStrategy.AUTO:
            strategy_info = f" | strategy: {strategy_label} (auto, merge ratio: {baseline.strategy.merge_ratio:.0%})"
        else:
            strategy_info = f" | strategy: {strategy_label}"

    console.print(f"\n[bold]Repository Baseline ({baseline.total_commits_analyzed} commits{strategy_info}):[/bold]")

    baseline_table = Table(show_header=True, header_style="bold")
    baseline_table.add_column("Metric", style="cyan")
    baseline_table.add_column("Mean Δ", justify="right")
    baseline_table.add_column("Std Dev", justify="right")
    baseline_table.add_column("Trend", justify="right")

    if baseline.qpe_stats:
        baseline_table.add_row(
            "QPE (GRPO)",
            f"{baseline.qpe_stats.mean:+.4f}",
            f"±{baseline.qpe_stats.std_dev:.4f}",
            _format_trend(baseline.qpe_stats.trend_coefficient, lower_is_better=False),
        )

    if baseline.token_delta_stats:
        baseline_table.add_row(
            "Tokens",
            _format_token_count(int(baseline.token_delta_stats.mean)),
            f"±{_format_token_count(int(baseline.token_delta_stats.std_dev))}",
            _format_trend(baseline.token_delta_stats.trend_coefficient, lower_is_better=False),
        )

    console.print(baseline_table)

    # Show current QPE absolute value for context
    if baseline.current_qpe:
        current_qpe_val = baseline.current_qpe.qpe
        qpe_abs_color = _color_for_qpe(current_qpe_val)
        console.print(f"  Current QPE: [{qpe_abs_color}]{current_qpe_val:.4f}[/{qpe_abs_color}]")

    console.print(f"\n[bold]{title}:[/bold]")

    impact_table = Table(show_header=True, header_style="bold")
    impact_table.add_column("Metric", style="cyan")
    impact_table.add_column("Change", justify="right")
    impact_table.add_column("Z-Score", justify="right")
    impact_table.add_column("Assessment", style="dim")

    # Apply deadband: if QPE delta is negligible, don't show misleading z-score
    qpe_delta_negligible = abs(assessment.qpe_delta) < 0.001
    if qpe_delta_negligible:
        impact_table.add_row(
            "QPE (GRPO)",
            "[dim]negligible[/dim]",
            f"[dim]{assessment.qpe_z_score:+.2f}[/dim]",
            "[dim]change too small to assess[/dim]",
        )
    else:
        qpe_color = _color_for_positive_negative(assessment.qpe_z_score)
        impact_table.add_row(
            "QPE (GRPO)",
            f"[{qpe_color}]{assessment.qpe_delta:+.4f}[/{qpe_color}]",
            f"{assessment.qpe_z_score:+.2f}",
            _interpret_z_score(assessment.qpe_z_score),
        )

    # Tokens row - neutral interpretation (size isn't inherently good/bad)
    token_color = "green" if assessment.token_z_score > 0 else "red" if assessment.token_z_score < 0 else "yellow"
    impact_table.add_row(
        "Tokens",
        f"[{token_color}]{_format_token_count(assessment.token_delta)}[/{token_color}]",
        f"[{token_color}]{assessment.token_z_score:+.2f}[/{token_color}]",
        _interpret_z_score(assessment.token_z_score),
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

    # Show per-smell advantage breakdown when available
    if smell_advantages:
        non_zero = [sa for sa in smell_advantages if sa.weighted_delta != 0.0]
        if non_zero:
            smell_table = Table(title="Smell Changes (baseline vs current)", show_header=True)
            smell_table.add_column("Smell", style="cyan")
            smell_table.add_column("Before", justify="right")
            smell_table.add_column("After", justify="right")
            smell_table.add_column("Weight", justify="right")
            smell_table.add_column("Weighted Δ", justify="right")

            for sa in non_zero:
                delta_color = _color_for_positive_negative(sa.weighted_delta)
                smell_table.add_row(
                    get_smell_label(sa.smell_name),
                    str(sa.baseline_count),
                    str(sa.candidate_count),
                    f"{sa.weight:.2f}",
                    f"[{delta_color}]{sa.weighted_delta:+.3f}[/{delta_color}]",
                )

            console.print(smell_table)


def display_baseline_comparison_compact(
    baseline: RepoBaseline,
    assessment: ImpactAssessment,
) -> str:
    """Return a compact string for baseline comparison (for hook feedback)."""
    lines = []
    lines.append(f"Repository Baseline ({baseline.total_commits_analyzed} commits):")

    if abs(assessment.qpe_delta) < 0.001:
        lines.append("  QPE (GRPO): negligible change")
    else:
        qpe_sign = "↑" if assessment.qpe_z_score > 0 else "↓" if assessment.qpe_z_score < 0 else "→"
        qpe_quality = "good" if assessment.qpe_z_score > 0 else "below avg" if assessment.qpe_z_score < 0 else "avg"
        lines.append(
            f"  QPE (GRPO): {assessment.qpe_delta:+.4f} (Z: {assessment.qpe_z_score:+.2f} {qpe_sign} {qpe_quality})"
        )

    category_display = assessment.impact_category.value.replace("_", " ").upper()
    lines.append(f"Session Impact: {category_display} ({assessment.impact_score:+.2f})")

    return "\n".join(lines)


def display_qpe_score(
    qpe_score: "QPEScore",
    metrics: "ExtendedComplexityMetrics",
) -> None:
    """Display Quality-Per-Effort score with component breakdown.

    Args:
        qpe_score: Computed QPE score with components
        metrics: Extended complexity metrics for context
    """

    console.print("\n[bold]Quality Score[/bold]")

    qpe_color = _color_for_qpe(qpe_score.qpe)
    console.print(f"  [bold]QPE:[/bold] [{qpe_color}]{qpe_score.qpe:.4f}[/{qpe_color}]")

    component_table = Table(title="QPE Components", show_header=True)
    component_table.add_column("Component", style="cyan")
    component_table.add_column("Value", justify="right")
    component_table.add_column("Description", style="dim")

    component_table.add_row(
        "MI (normalized)",
        f"{qpe_score.mi_normalized:.3f}",
        f"Maintainability Index / 100 (raw: {metrics.average_mi:.1f})",
    )

    smell_color = _color_for_inverted_threshold(qpe_score.smell_penalty, 0.1, 0.3)
    component_table.add_row(
        "Smell Penalty",
        f"[{smell_color}]{qpe_score.smell_penalty:.3f}[/{smell_color}]",
        "Sigmoid-saturated deduction (0-0.9)",
    )

    component_table.add_row(
        "Adjusted Quality",
        f"{qpe_score.adjusted_quality:.3f}",
        "MI × (1 - smell_penalty) + bonuses",
    )

    console.print(component_table)

    smell_counts_dict = qpe_score.smell_counts.model_dump()
    if any(count > 0 for count in smell_counts_dict.values()):
        smell_table = Table(title="Code Smell Breakdown", show_header=True)
        smell_table.add_column("Smell", style="cyan")
        smell_table.add_column("Count", justify="right")

        for category in [SmellCategory.GENERAL, SmellCategory.PYTHON]:
            category_smells = [
                (name, smell_counts_dict.get(name, 0))
                for name, defn in SMELL_REGISTRY.items()
                if defn.category == category
            ]
            category_smells = [(n, c) for n, c in category_smells if c > 0]
            if not category_smells:
                continue
            category_label = "General" if category == SmellCategory.GENERAL else "Python"
            smell_table.add_row(f"[bold dim]{category_label}[/bold dim]", "")
            for smell_name, count in sorted(category_smells, key=lambda x: -x[1]):
                smell_table.add_row(f"  {get_smell_label(smell_name)}", str(count))

        console.print(smell_table)

    console.print(
        "\n[dim]Higher QPE (GRPO) = better quality per effort | Higher Quality = better absolute quality[/dim]"
    )


def display_cross_project_comparison(comparison: "CrossProjectComparison") -> None:
    """Display cross-project comparison results ranked by QPE.

    Args:
        comparison: Cross-project comparison results
    """
    console.print(f"\n[bold]Cross-Project Comparison ({comparison.total_projects} projects)[/bold]")
    console.print(f"[dim]Compared at: {comparison.compared_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")

    table = Table(show_header=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("QPE", justify="right")
    table.add_column("MI", justify="right")
    table.add_column("Smell Penalty", justify="right")

    for rank, result in enumerate(comparison.rankings, 1):
        rank_style = "green" if rank == 1 else "yellow" if rank == 2 else ""
        qpe_color = _color_for_qpe(result.qpe_score.qpe)
        smell_color = _color_for_inverted_threshold(result.qpe_score.smell_penalty, 0.1, 0.3)

        table.add_row(
            f"[{rank_style}]#{rank}[/{rank_style}]" if rank_style else f"#{rank}",
            result.project_name,
            f"[{qpe_color}]{result.qpe_score.qpe:.4f}[/{qpe_color}]",
            f"{result.metrics.average_mi:.1f}",
            f"[{smell_color}]{result.qpe_score.smell_penalty:.3f}[/{smell_color}]",
        )

    console.print(table)
    console.print("\n[dim]Higher QPE = better quality[/dim]")


def display_leaderboard(entries: list[LeaderboardEntry]) -> None:
    """Display the Quality leaderboard for cross-project comparison.

    Args:
        entries: List of LeaderboardEntry objects, already sorted by quality score
    """
    console.print("\n[bold]Quality Leaderboard[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("Quality", justify="right")
    table.add_column("Smell", justify="right")
    table.add_column("MI", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Effort", justify="right")
    table.add_column("Commit", justify="center")
    table.add_column("Commit Date", justify="center")

    for rank, entry in enumerate(entries, 1):
        rank_style = "green" if rank == 1 else "yellow" if rank == 2 else "blue" if rank == 3 else ""
        qual_color = _color_for_qpe(entry.qpe_score)
        smell_color = _color_for_inverted_threshold(entry.smell_penalty, 0.1, 0.3)

        try:
            metrics = ExtendedComplexityMetrics.model_validate_json(entry.metrics_json)
            total_tokens = metrics.total_tokens
        except Exception as e:
            logger.warning(f"Failed to parse metrics_json for {entry.project_name}: {e}")
            total_tokens = 0

        tokens_str = f"{total_tokens // 1000}K" if total_tokens >= 1000 else str(total_tokens)
        effort_str = f"{entry.total_effort / 1000:.0f}K" if entry.total_effort >= 1000 else f"{entry.total_effort:.0f}"

        table.add_row(
            f"[{rank_style}]#{rank}[/{rank_style}]" if rank_style else f"#{rank}",
            entry.project_name,
            f"[{qual_color}]{entry.qpe_score:.4f}[/{qual_color}]",
            f"[{smell_color}]{entry.smell_penalty:.3f}[/{smell_color}]",
            f"{entry.mi_normalized:.3f}",
            f"[dim]{tokens_str}[/dim]",
            f"[dim]{effort_str}[/dim]",
            f"[dim]{entry.commit_sha_short}[/dim]",
            entry.measured_at.strftime("%Y-%m-%d"),
        )

    console.print(table)
    console.print("\n[dim]Higher Quality = better absolute code quality. Use --append to add projects.[/dim]")


def display_implementation_comparison(comparison: ImplementationComparison) -> None:
    """Display implementation comparison with QPE breakdown and per-smell advantage.

    Args:
        comparison: ImplementationComparison result from compare_subtrees()
    """
    console.print(f"\n[bold]Implementation Comparison ({comparison.ref})[/bold]")
    console.print(f"  A: {comparison.prefix_a}")
    console.print(f"  B: {comparison.prefix_b}")

    # QPE summary table
    qpe_table = Table(title="QPE Comparison", show_header=True)
    qpe_table.add_column("Metric", style="cyan")
    qpe_table.add_column("A", justify="right")
    qpe_table.add_column("B", justify="right")

    qpe_a_color = _color_for_qpe(comparison.qpe_a.qpe)
    qpe_b_color = _color_for_qpe(comparison.qpe_b.qpe)

    qpe_table.add_row(
        "QPE",
        f"[{qpe_a_color}]{comparison.qpe_a.qpe:.4f}[/{qpe_a_color}]",
        f"[{qpe_b_color}]{comparison.qpe_b.qpe:.4f}[/{qpe_b_color}]",
    )
    qpe_table.add_row(
        "MI (normalized)",
        f"{comparison.qpe_a.mi_normalized:.3f}",
        f"{comparison.qpe_b.mi_normalized:.3f}",
    )
    qpe_table.add_row(
        "Smell Penalty",
        f"{comparison.qpe_a.smell_penalty:.3f}",
        f"{comparison.qpe_b.smell_penalty:.3f}",
    )
    console.print(qpe_table)

    # Advantage display
    adv = comparison.aggregate_advantage
    adv_color = _color_for_advantage(adv)
    console.print(f"\n  [bold]GRPO Advantage (B over A):[/bold] [{adv_color}]{adv:+.4f}[/{adv_color}]")

    winner_color = "green" if comparison.winner != "tie" else "yellow"
    console.print(f"  [bold]Winner:[/bold] [{winner_color}]{comparison.winner}[/{winner_color}]")

    # Per-smell advantage breakdown (only show non-zero)
    non_zero_smells = [sa for sa in comparison.smell_advantages if sa.weighted_delta != 0.0]
    if non_zero_smells:
        smell_table = Table(title="Per-Smell Advantage (B vs A)", show_header=True)
        smell_table.add_column("Smell", style="cyan")
        smell_table.add_column("A Count", justify="right")
        smell_table.add_column("B Count", justify="right")
        smell_table.add_column("Weight", justify="right")
        smell_table.add_column("Weighted Δ", justify="right")

        for sa in non_zero_smells:
            delta_color = _color_for_positive_negative(sa.weighted_delta)
            smell_table.add_row(
                get_smell_label(sa.smell_name),
                str(sa.baseline_count),
                str(sa.candidate_count),
                f"{sa.weight:.2f}",
                f"[{delta_color}]{sa.weighted_delta:+.3f}[/{delta_color}]",
            )

        console.print(smell_table)

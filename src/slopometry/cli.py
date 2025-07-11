"""CLI for the slopometry Claude Code tracker."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from slopometry.database import EventDatabase
from slopometry.experiment_orchestrator import ExperimentOrchestrator
from slopometry.models import NextFeaturePrediction, UserStory
from slopometry.settings import settings

console = Console()


def complete_session_id(ctx, param, incomplete):
    """Complete session IDs from the database."""
    try:
        db = EventDatabase()
        sessions = db.list_sessions()
        return [session for session in sessions if session.startswith(incomplete)]
    except Exception:
        return []


def complete_nfp_id(ctx, param, incomplete):
    """Complete NFP objective IDs from the database."""
    try:
        db = EventDatabase()
        objectives = db.list_nfp_objectives()
        return [obj.id for obj in objectives if obj.id.startswith(incomplete)]
    except Exception:
        return []


def complete_experiment_id(ctx, param, incomplete):
    """Complete experiment IDs from the database."""
    try:
        db = EventDatabase()
        with db._get_db_connection() as conn:
            rows = conn.execute("SELECT id FROM experiment_runs ORDER BY start_time DESC").fetchall()
            return [row[0] for row in rows if row[0].startswith(incomplete)]
    except Exception:
        return []


def create_slopometry_hooks() -> dict:
    """Create slopometry hook configuration for Claude Code."""
    base_command = settings.hook_command.replace("hook-handler", "hook-{}")

    return {
        "PreToolUse": [
            {"matcher": ".*", "hooks": [{"type": "command", "command": base_command.format("pre-tool-use")}]}
        ],
        "PostToolUse": [
            {"matcher": ".*", "hooks": [{"type": "command", "command": base_command.format("post-tool-use")}]}
        ],
        "Notification": [{"hooks": [{"type": "command", "command": base_command.format("notification")}]}],
        "Stop": [{"hooks": [{"type": "command", "command": base_command.format("stop")}]}],
        "SubagentStop": [{"hooks": [{"type": "command", "command": base_command.format("subagent-stop")}]}],
    }


@click.group()
def cli():
    """Slopometry - Claude Code session tracker."""
    pass


@cli.command("hook-handler", hidden=True)
def hook_handler():
    """Internal command for processing hook events."""
    from slopometry.hook_handler import handle_hook

    sys.exit(handle_hook())


@cli.command("hook-pre-tool-use", hidden=True)
def hook_pre_tool_use():
    """Internal command for processing PreToolUse hook events."""
    from slopometry.hook_handler import handle_hook
    from slopometry.models import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.PRE_TOOL_USE))


@cli.command("hook-post-tool-use", hidden=True)
def hook_post_tool_use():
    """Internal command for processing PostToolUse hook events."""
    from slopometry.hook_handler import handle_hook
    from slopometry.models import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.POST_TOOL_USE))


@cli.command("hook-notification", hidden=True)
def hook_notification():
    """Internal command for processing Notification hook events."""
    from slopometry.hook_handler import handle_hook
    from slopometry.models import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.NOTIFICATION))


@cli.command("hook-stop", hidden=True)
def hook_stop():
    """Internal command for processing Stop hook events."""
    from slopometry.hook_handler import handle_hook
    from slopometry.models import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.STOP))


@cli.command("hook-subagent-stop", hidden=True)
def hook_subagent_stop():
    """Internal command for processing SubagentStop hook events."""
    from slopometry.hook_handler import handle_hook
    from slopometry.models import HookEventType

    sys.exit(handle_hook(event_type_override=HookEventType.SUBAGENT_STOP))


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell):
    """Generate shell completion script."""

    if shell == "bash":
        console.print("[bold]Add this to your ~/.bashrc:[/bold]")
        console.print('eval "$(_SLOPOMETRY_COMPLETE=bash_source slopometry)"')
        console.print("\n[bold]Or install directly:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=bash_source slopometry > ~/.slopometry-complete.sh")
        console.print("echo 'source ~/.slopometry-complete.sh' >> ~/.bashrc")
    elif shell == "zsh":
        console.print("[bold]Add this to your ~/.zshrc:[/bold]")
        console.print('eval "$(_SLOPOMETRY_COMPLETE=zsh_source slopometry)"')
        console.print("\n[bold]Or install directly:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=zsh_source slopometry > ~/.slopometry-complete.zsh")
        console.print("echo 'source ~/.slopometry-complete.zsh' >> ~/.zshrc")
    elif shell == "fish":
        console.print("[bold]Add this to your fish config:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=fish_source slopometry | source")
        console.print("\n[bold]Or install directly:[/bold]")
        console.print("_SLOPOMETRY_COMPLETE=fish_source slopometry > ~/.config/fish/completions/slopometry.fish")

    console.print("\n[yellow]Note: Restart your shell or source your config file after installation.[/yellow]")


@cli.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Install hooks globally (~/.claude) or locally (./.claude)",
)
def install(global_):
    """Install slopometry hooks into Claude Code settings to automatically track all sessions and tool usage."""
    settings_dir = Path.home() / ".claude" if global_ else Path.cwd() / ".claude"
    settings_file = settings_dir / "settings.json"

    settings_dir.mkdir(exist_ok=True)

    existing_settings = {}
    if settings_file.exists():
        with open(settings_file) as f:
            existing_settings = json.load(f)

    if "hooks" in existing_settings and settings.backup_existing_settings:
        backup_file = settings_dir / f"settings.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, "w") as f:
            json.dump(existing_settings, f, indent=2)
        console.print(f"[yellow]Backed up existing settings to {backup_file}[/yellow]")

    slopometry_hooks = create_slopometry_hooks()

    if "hooks" not in existing_settings:
        existing_settings["hooks"] = {}

    for hook_type, hook_configs in slopometry_hooks.items():
        if hook_type not in existing_settings["hooks"]:
            existing_settings["hooks"][hook_type] = []

        existing_settings["hooks"][hook_type] = [
            h
            for h in existing_settings["hooks"][hook_type]
            if not (
                isinstance(h.get("hooks"), list)
                and any("slopometry hook-" in hook.get("command", "") for hook in h.get("hooks", []))
            )
        ]

        existing_settings["hooks"][hook_type].extend(hook_configs)

    with open(settings_file, "w") as f:
        json.dump(existing_settings, f, indent=2)

    scope = "globally" if global_ else "locally"
    console.print(f"[green]Slopometry hooks installed {scope} in {settings_file}[/green]")
    console.print("[cyan]All Claude Code sessions will now be automatically tracked[/cyan]")


@cli.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Remove hooks globally (~/.claude) or locally (./.claude)",
)
def uninstall(global_):
    """Remove slopometry hooks from Claude Code settings to completely stop automatic session tracking."""
    settings_dir = Path.home() / ".claude" if global_ else Path.cwd() / ".claude"
    settings_file = settings_dir / "settings.json"

    if not settings_file.exists():
        console.print(f"[yellow]No settings file found at {settings_file}[/yellow]")
        return

    with open(settings_file) as f:
        settings_data = json.load(f)

    if "hooks" not in settings_data:
        console.print("[yellow]No hooks configuration found[/yellow]")
        return

    removed_any = False
    for hook_type in settings_data["hooks"]:
        original_length = len(settings_data["hooks"][hook_type])
        settings_data["hooks"][hook_type] = [
            h
            for h in settings_data["hooks"][hook_type]
            if not (
                isinstance(h.get("hooks"), list)
                and any("slopometry hook-" in hook.get("command", "") for hook in h.get("hooks", []))
            )
        ]
        if len(settings_data["hooks"][hook_type]) < original_length:
            removed_any = True

    settings_data["hooks"] = {k: v for k, v in settings_data["hooks"].items() if v}

    if not settings_data["hooks"]:
        del settings_data["hooks"]

    with open(settings_file, "w") as f:
        json.dump(settings_data, f, indent=2)

    scope = "globally" if global_ else "locally"
    if removed_any:
        console.print(f"[green]Slopometry hooks removed {scope} from {settings_file}[/green]")
    else:
        console.print(f"[yellow]No slopometry hooks found to remove {scope}[/yellow]")


@cli.command()
@click.option("--limit", default=None, help="Number of recent sessions to show")
def ls(limit):
    """List recent Claude Code sessions."""
    list_sessions_impl(limit)


def list_sessions_impl(limit):
    """Implementation for listing sessions."""
    db = EventDatabase()
    all_sessions = db.list_sessions()
    sessions = all_sessions[: int(limit)] if limit else all_sessions

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        console.print("[dim]Run 'slopometry install' to start tracking Claude Code sessions[/dim]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Project", style="magenta")
    table.add_column("Start Time", style="green")
    table.add_column("Events", justify="right")
    table.add_column("Tools Used", justify="right")

    for session_id in sessions:
        stats = db.get_session_statistics(session_id)
        if stats:
            project_display = f"{stats.project.name} ({stats.project.source.value})" if stats.project else "N/A"
            table.add_row(
                session_id,
                project_display,
                stats.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                str(stats.total_events),
                str(len(stats.tool_usage)),
            )

    console.print(table)


@cli.command()
@click.argument("session_id", shell_complete=complete_session_id)
def show(session_id):
    """Show detailed statistics for a session."""
    show_session_summary(session_id)


@cli.command()
def latest():
    """Show detailed statistics for the most recent session."""
    db = EventDatabase()
    sessions = db.list_sessions()

    if not sessions:
        console.print("[red]No sessions found[/red]")
        return

    most_recent = sessions[0]
    console.print(f"[bold]Showing most recent session: {most_recent}[/bold]\n")
    show_session_summary(most_recent)


def show_session_summary(session_id: str):
    """Display session statistics."""
    db = EventDatabase()
    stats = db.get_session_statistics(session_id)

    if not stats:
        console.print(f"[red]No data found for session {session_id}[/red]")
        return

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
        table = Table(title="Events by Type")
        table.add_column("Event Type", style="cyan")
        table.add_column("Count", justify="right")

        for event_type, count in sorted(stats.events_by_type.items()):
            table.add_row(event_type.value, str(count))

        console.print(table)

    if stats.tool_usage:
        table = Table(title="Tool Usage")
        table.add_column("Tool", style="green")
        table.add_column("Count", justify="right")

        for tool_type, count in sorted(stats.tool_usage.items(), key=lambda x: x[1], reverse=True):
            table.add_row(tool_type.value, str(count))

        console.print(table)

    if stats.average_tool_duration_ms:
        console.print(f"\nAverage tool duration: {stats.average_tool_duration_ms:.0f}ms")

    if stats.error_count > 0:
        console.print(f"[red]Errors: {stats.error_count}[/red]")

    if stats.initial_git_state and stats.initial_git_state.is_git_repo:
        console.print("\n[bold]Git Metrics[/bold]")
        console.print(f"Commits made: [green]{stats.commits_made}[/green]")

        if stats.initial_git_state.current_branch:
            console.print(f"Branch: {stats.initial_git_state.current_branch}")

        if stats.initial_git_state.has_uncommitted_changes:
            console.print("[yellow]Had uncommitted changes at start[/yellow]")

        if stats.final_git_state and stats.final_git_state.has_uncommitted_changes:
            console.print("[yellow]Has uncommitted changes at end[/yellow]")

    if stats.complexity_metrics and stats.complexity_metrics.total_files_analyzed > 0:
        console.print("\n[bold]Complexity Metrics[/bold]")

        overview_table = Table(title="Complexity Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", justify="right")

        overview_table.add_row("Files analyzed", str(stats.complexity_metrics.total_files_analyzed))
        overview_table.add_row("[bold]Cyclomatic Complexity[/bold]", "")
        overview_table.add_row("  Total complexity", str(stats.complexity_metrics.total_complexity))
        overview_table.add_row("  Average complexity", f"{stats.complexity_metrics.average_complexity:.1f}")
        overview_table.add_row("  Max complexity", str(stats.complexity_metrics.max_complexity))
        overview_table.add_row("  Min complexity", str(stats.complexity_metrics.min_complexity))
        overview_table.add_row("[bold]Halstead Metrics[/bold]", "")
        overview_table.add_row("  Total volume", f"{stats.complexity_metrics.total_volume:.1f}")
        overview_table.add_row("  Average volume", f"{stats.complexity_metrics.average_volume:.1f}")
        overview_table.add_row("  Total difficulty", f"{stats.complexity_metrics.total_difficulty:.1f}")
        overview_table.add_row("  Average difficulty", f"{stats.complexity_metrics.average_difficulty:.1f}")
        overview_table.add_row("  Total effort", f"{stats.complexity_metrics.total_effort:.1f}")
        overview_table.add_row("[bold]Maintainability Index[/bold]", "")
        overview_table.add_row("  Total MI", f"{stats.complexity_metrics.total_mi:.1f}")
        overview_table.add_row("  Average MI", f"{stats.complexity_metrics.average_mi:.1f} (higher is better)")

        console.print(overview_table)

        if stats.complexity_metrics.files_by_complexity:
            files_table = Table(title="Files by Complexity")
            files_table.add_column("File", style="cyan")
            files_table.add_column("Complexity", justify="right")

            sorted_files = sorted(
                stats.complexity_metrics.files_by_complexity.items(), key=lambda x: x[1], reverse=True
            )[:10]

            for file_path, complexity in sorted_files:
                files_table.add_row(file_path, str(complexity))

            console.print(files_table)

    if stats.complexity_delta:
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
        changes_table.add_row(
            "  Total difficulty", f"[{diff_color}]{delta.total_difficulty_change:+.1f}[/{diff_color}]"
        )
        changes_table.add_row(
            "  Average difficulty", f"[{diff_color}]{delta.avg_difficulty_change:+.1f}[/{diff_color}]"
        )
        effort_color = (
            "green" if delta.total_effort_change < 0 else "red" if delta.total_effort_change > 0 else "yellow"
        )
        changes_table.add_row("  Total effort", f"[{effort_color}]{delta.total_effort_change:+.1f}[/{effort_color}]")

        # Maintainability Index changes (note: higher is better, so green for positive)
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

    if stats.plan_evolution and stats.plan_evolution.total_plan_steps > 0:
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
                ratio_display = (
                    f"{step.search_to_implementation_ratio:.2f}" if step.implementation_events > 0 else "N/A"
                )
                table.add_row(str(step.step_number), str(step.events_in_step), ratio_display, change_summary)

            if len(evolution.plan_steps) > 5:
                table.add_row("...", "...", "...", f"... and {len(evolution.plan_steps) - 5} more steps")

            console.print(table)


@cli.command()
@click.argument("session_id", required=False, shell_complete=complete_session_id)
@click.option("--all", "all_sessions", is_flag=True, help="Delete all sessions")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def cleanup(session_id, all_sessions, yes):
    """Clean up session data.

    If SESSION_ID is provided, delete that specific session.
    If --all is provided, delete all sessions.
    Otherwise, show usage help.
    """
    db = EventDatabase()

    if session_id and all_sessions:
        console.print("[red]Error: Cannot specify both session ID and --all[/red]")
        return

    if not session_id and not all_sessions:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  slopometry cleanup SESSION_ID    # Delete specific session")
        console.print("  slopometry cleanup --all         # Delete all sessions")
        return

    if session_id:
        stats = db.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        console.print(f"\n[bold]Session to delete: {session_id}[/bold]")
        console.print(f"Start time: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Total events: {stats.total_events}")

        if not yes:
            confirm = click.confirm("\nAre you sure you want to delete this session?", default=False)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        events_deleted, files_deleted = db.cleanup_session(session_id)
        console.print(f"[green]Deleted {events_deleted} events and {files_deleted} files[/green]")

    else:
        sessions = db.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions to delete[/yellow]")
            return

        console.print(f"\n[bold red]WARNING: This will delete ALL {len(sessions)} sessions![/bold red]")
        console.print("This action cannot be undone.")

        if not yes:
            confirm = click.confirm("\nAre you sure you want to delete all sessions?", default=False)
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                return

        sessions_deleted, events_deleted, files_deleted = db.cleanup_all_sessions()
        console.print(
            f"[green]Deleted {sessions_deleted} sessions, {events_deleted} events, and {files_deleted} files[/green]"
        )


@cli.command()
def status():
    """Show installation status and hook configuration."""
    global_settings = Path.home() / ".claude" / "settings.json"
    local_settings = Path.cwd() / ".claude" / "settings.json"

    console.print("[bold]Slopometry Installation Status[/bold]\n")

    console.print(f"[cyan]Data directory:[/cyan] {settings.resolved_database_path.parent}")
    console.print(f"[cyan]Database:[/cyan] {settings.resolved_database_path}\n")

    global_installed = _check_hooks_installed(global_settings)
    status_icon = "[green]✓[/green]" if global_installed else "[red]✗[/red]"
    console.print(f"{status_icon} Global hooks: {global_settings}")

    local_installed = _check_hooks_installed(local_settings)
    status_icon = "[green]✓[/green]" if local_installed else "[red]✗[/red]"
    console.print(f"{status_icon} Local hooks: {local_settings}")

    if not global_installed and not local_installed:
        console.print("\n[yellow]No slopometry hooks found. Run 'slopometry install' to start tracking.[/yellow]")
    else:
        console.print("\n[green]Hooks are installed. Claude Code sessions are being tracked automatically.[/green]")

        db = EventDatabase()
        sessions = db.list_sessions()[:3]
        if sessions:
            console.print("\n[bold]Recent Sessions:[/bold]")
            for session_id in sessions:
                stats = db.get_session_statistics(session_id)
                if stats:
                    console.print(f"  • {session_id} ({stats.total_events} events)")


@cli.command()
@click.option("--enable/--disable", default=None, help="Enable or disable stop event feedback")
def feedback(enable):
    """Configure complexity feedback on stop events."""
    if enable is None:
        current_status = "enabled" if settings.enable_complexity_feedback else "disabled"
        console.print(f"[bold]Complexity feedback is currently {current_status}[/bold]")
        console.print("")
        console.print("To change this setting:")
        console.print("  slopometry feedback --enable    # Enable feedback")
        console.print("  slopometry feedback --disable   # Disable feedback")
        console.print("")
        if not settings.enable_complexity_feedback:
            console.print("[yellow]Note: Feedback is disabled by default. Enable it to receive[/yellow]")
            console.print("[yellow]complexity analysis when Claude Code sessions end.[/yellow]")
        return

    env_file = Path(".env")
    env_var = "SLOPOMETRY_ENABLE_STOP_FEEDBACK"

    if enable:
        console.print("[green]Enabling[/green] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=true")

        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                lines = content.split("\n")
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=true")
                    else:
                        new_lines.append(line)
                env_file.write_text("\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\n{env_var}=true\n")
        else:
            env_file.write_text(f"{env_var}=true\n")

        console.print(f"[green]Added {env_var}=true to .env file[/green]")

    else:
        console.print("[yellow]Disabling[/yellow] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=false")

        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                lines = content.split("\n")
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=false")
                    else:
                        new_lines.append(line)
                env_file.write_text("\n".join(new_lines))
            else:
                with env_file.open("a") as f:
                    f.write(f"\n{env_var}=false\n")
        else:
            env_file.write_text(f"{env_var}=false\n")

        console.print(f"[green]Added {env_var}=false to .env file[/green]")

    console.print("")
    console.print("[bold]Note:[/bold] You may need to restart Claude Code for changes to take effect.")


@cli.command()
@click.option("--commits", "-c", default=5, help="Number of commits to analyze (default: 5)")
@click.option("--max-workers", "-w", default=4, help="Maximum parallel workers (default: 4)")
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def run_experiments(commits: int, max_workers: int, repo_path: Path | None):
    """Run parallel experiments across git commits to track and analyze code complexity evolution patterns."""
    if repo_path is None:
        repo_path = Path.cwd()

    orchestrator = ExperimentOrchestrator(repo_path)

    # Generate commit pairs (HEAD~n to HEAD~n-1)
    commit_pairs = []
    for i in range(commits, 0, -1):
        start_commit = f"HEAD~{i}"
        target_commit = f"HEAD~{i - 1}" if i > 1 else "HEAD"
        commit_pairs.append((start_commit, target_commit))

    console.print(f"[bold]Running {len(commit_pairs)} experiments with up to {max_workers} workers[/bold]")
    console.print(f"Repository: {repo_path}")
    console.print("Commit pairs:")
    for start, target in commit_pairs:
        console.print(f"  {start} → {target}")

    try:
        experiments = orchestrator.run_parallel_experiments(commit_pairs, max_workers)

        console.print(f"\n[green]✓ Completed {len(experiments)} experiments[/green]")

        # Show summary
        for experiment in experiments.values():
            status_color = "green" if experiment.status.value == "completed" else "red"
            console.print(
                f"  {experiment.start_commit} → {experiment.target_commit}: [{status_color}]{experiment.status.value}[/]"
            )

    except Exception as e:
        console.print(f"[red]Failed to run experiments: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--base-commit", "-b", default="HEAD~10", help="Base commit (default: HEAD~10)")
@click.option("--head-commit", "-h", default="HEAD", help="Head commit (default: HEAD)")
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def analyze_commits(base_commit: str, head_commit: str, repo_path: Path | None):
    """Analyze complexity evolution across a chain of commits."""
    if repo_path is None:
        repo_path = Path.cwd()

    orchestrator = ExperimentOrchestrator(repo_path)

    console.print(f"[bold]Analyzing commits from {base_commit} to {head_commit}[/bold]")
    console.print(f"Repository: {repo_path}")

    try:
        orchestrator.analyze_commit_chain(base_commit, head_commit)
        console.print("\n[green]✓ Analysis complete[/green]")

    except Exception as e:
        console.print(f"[red]Failed to analyze commits: {e}[/red]")
        sys.exit(1)


@cli.command()
def list_experiments():
    """List all experiment runs."""
    db = EventDatabase()

    # Get all experiments (we'll need to add this method)
    try:
        with db._get_db_connection() as conn:
            rows = conn.execute("""
                SELECT id, repository_path, start_commit, target_commit, 
                       start_time, end_time, status
                FROM experiment_runs 
                ORDER BY start_time DESC
            """).fetchall()

        if not rows:
            console.print("[yellow]No experiments found[/yellow]")
            return

        table = Table(title="Experiment Runs")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Repository", style="magenta")
        table.add_column("Commits", style="blue")
        table.add_column("Start Time", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Status", style="bold")

        for row in rows:
            experiment_id, repo_path, start_commit, target_commit, start_time, end_time, status = row

            start_dt = datetime.fromisoformat(start_time)
            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                duration = str(end_dt - start_dt)
            else:
                duration = "Running..."

            status_style = "green" if status == "completed" else "red" if status == "failed" else "yellow"

            table.add_row(
                experiment_id,  # Show full ID for easy copy-paste
                Path(repo_path).name,
                f"{start_commit} → {target_commit}",
                start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                duration,
                f"[{status_style}]{status}[/]",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list experiments: {e}[/red]")


@cli.command()
@click.argument("experiment_id", shell_complete=complete_experiment_id)
def show_experiment(experiment_id: str):
    """Show detailed progress for an experiment."""
    db = EventDatabase()

    try:
        with db._get_db_connection() as conn:
            # Get experiment info
            experiment_row = conn.execute(
                "SELECT * FROM experiment_runs WHERE id LIKE ?", (f"{experiment_id}%",)
            ).fetchone()

            if not experiment_row:
                console.print(f"[red]Experiment {experiment_id} not found[/red]")
                return

            # Get progress history
            progress_rows = conn.execute(
                """
                SELECT timestamp, cli_score, complexity_score, halstead_score, maintainability_score
                FROM experiment_progress 
                WHERE experiment_id = ?
                ORDER BY timestamp
            """,
                (experiment_row[0],),
            ).fetchall()

        console.print(f"[bold]Experiment: {experiment_row[0]}[/bold]")
        console.print(f"Repository: {experiment_row[1]}")
        console.print(f"Commits: {experiment_row[2]} → {experiment_row[3]}")
        console.print(f"Status: {experiment_row[8]}")

        if progress_rows:
            table = Table(title="Progress History")
            table.add_column("Timestamp", style="cyan")
            table.add_column("CLI Score", style="green", justify="right")
            table.add_column("Complexity", style="blue", justify="right")
            table.add_column("Halstead", style="yellow", justify="right")
            table.add_column("Maintainability", style="magenta", justify="right")

            for row in progress_rows:
                timestamp, cli_score, complexity_score, halstead_score, maintainability_score = row
                dt = datetime.fromisoformat(timestamp)

                table.add_row(
                    dt.strftime("%H:%M:%S"),
                    f"{cli_score:.3f}",
                    f"{complexity_score:.3f}",
                    f"{halstead_score:.3f}",
                    f"{maintainability_score:.3f}",
                )

            console.print(table)

            # Show final score
            final_row = progress_rows[-1]
            final_cli = final_row[1]
            console.print(f"\n[bold]Final CLI Score: {final_cli:.3f}[/bold]")
        else:
            console.print("[yellow]No progress data found[/yellow]")

    except Exception as e:
        console.print(f"[red]Failed to show experiment: {e}[/red]")


# NFP (Next Feature Prediction) Management Commands


@cli.command()
@click.option("--title", "-t", required=True, help="Title for this feature set")
@click.option("--description", "-d", required=True, help="High-level description")
@click.option("--base-commit", "-b", default="HEAD~1", help="Base commit (default: HEAD~1)")
@click.option("--target-commit", "-c", default="HEAD", help="Target commit (default: HEAD)")
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def create_nfp(title: str, description: str, base_commit: str, target_commit: str, repo_path: Path | None):
    """Create a new NFP (Next Feature Prediction) objective to define feature development goals and user stories."""
    if repo_path is None:
        repo_path = Path.cwd()

    db = EventDatabase()

    # Create NFP objective
    nfp = NextFeaturePrediction(
        title=title,
        description=description,
        base_commit=base_commit,
        target_commit=target_commit,
        repository_path=repo_path,
    )

    try:
        db.save_nfp_objective(nfp)
        console.print(f"[green]✓ Created NFP objective: {nfp.id}[/green]")
        console.print(f"Title: {title}")
        console.print(f"Commits: {base_commit} → {target_commit}")
        console.print(f"Repository: {repo_path}")
        console.print(f"\nUse '[cyan]slopometry add-user-story {nfp.id}[/cyan]' to add user stories.")

    except Exception as e:
        console.print(f"[red]Failed to create NFP: {e}[/red]")


@cli.command()
@click.argument("nfp_id", shell_complete=complete_nfp_id)
@click.option("--title", "-t", required=True, help="Title of the user story")
@click.option("--description", "-d", required=True, help="Detailed description")
@click.option("--priority", "-p", default=1, type=int, help="Priority level (1=highest, 5=lowest)")
@click.option("--complexity", "-c", default=0, type=int, help="Estimated complexity points")
@click.option("--acceptance-criteria", "-a", multiple=True, help="Acceptance criteria (can be used multiple times)")
@click.option("--tags", multiple=True, help="Tags for categorization (can be used multiple times)")
def add_user_story(
    nfp_id: str, title: str, description: str, priority: int, complexity: int, acceptance_criteria: tuple, tags: tuple
):
    """Add a detailed user story with acceptance criteria and priority levels to an existing NFP objective."""
    db = EventDatabase()

    # Get existing NFP
    nfp = db.get_nfp_objective(nfp_id)
    if not nfp:
        console.print(f"[red]NFP objective {nfp_id} not found[/red]")
        return

    # Create user story
    story = UserStory(
        title=title,
        description=description,
        priority=priority,
        estimated_complexity=complexity,
        acceptance_criteria=[*acceptance_criteria],
        tags=[*tags],
    )

    # Add to NFP and save
    nfp.user_stories.append(story)
    nfp.updated_at = datetime.now()
    db.save_nfp_objective(nfp)

    console.print(f"[green]✓ Added user story to NFP {nfp_id[:8]}...[/green]")
    console.print(f"Title: {title}")
    console.print(f"Priority: {priority}")
    console.print(f"Complexity: {complexity}")
    if acceptance_criteria:
        console.print(f"Acceptance criteria: {', '.join(acceptance_criteria)}")
    if tags:
        console.print(f"Tags: {', '.join(tags)}")


@cli.command()
@click.option("--repo-path", "-r", type=click.Path(exists=True, path_type=Path), help="Repository path filter")
def list_nfp(repo_path: Path | None):
    """List all NFP objectives."""
    db = EventDatabase()

    try:
        repo_filter = str(repo_path) if repo_path else None
        objectives = db.list_nfp_objectives(repo_filter)

        if not objectives:
            console.print("[yellow]No NFP objectives found[/yellow]")
            return

        table = Table(title="NFP Objectives")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Commits", style="blue")
        table.add_column("Stories", style="green", justify="right")
        table.add_column("Complexity", style="yellow", justify="right")
        table.add_column("Created", style="dim")

        for nfp in objectives:
            table.add_row(
                nfp.id,  # Show full ID for easy copy-paste
                nfp.title,
                f"{nfp.base_commit} → {nfp.target_commit}",
                str(nfp.story_count),
                str(nfp.total_estimated_complexity),
                nfp.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list NFP objectives: {e}[/red]")


@cli.command()
@click.argument("nfp_id", shell_complete=complete_nfp_id)
def show_nfp(nfp_id: str):
    """Show detailed information for an NFP objective."""
    db = EventDatabase()

    try:
        nfp = db.get_nfp_objective(nfp_id)
        if not nfp:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")
            return

        console.print(f"[bold]NFP Objective: {nfp.id}[/bold]")
        console.print(f"Title: {nfp.title}")
        console.print(f"Description: {nfp.description}")
        console.print(f"Commits: {nfp.base_commit} → {nfp.target_commit}")
        console.print(f"Created: {nfp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Updated: {nfp.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if nfp.user_stories:
            console.print(f"\n[bold]User Stories ({len(nfp.user_stories)})[/bold]")

            # Group by priority
            for priority in range(1, 6):
                stories = nfp.get_stories_by_priority(priority)
                if stories:
                    priority_names = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Nice to Have"}
                    console.print(f"\n[bold]Priority {priority} ({priority_names[priority]})[/bold]")

                    for story in stories:
                        console.print(f"  • [cyan]{story.title}[/cyan]")
                        console.print(f"    {story.description}")
                        if story.acceptance_criteria:
                            console.print(f"    Acceptance: {', '.join(story.acceptance_criteria)}")
                        if story.tags:
                            console.print(f"    Tags: {', '.join(story.tags)}")
                        console.print(f"    Complexity: {story.estimated_complexity}")
                        console.print("")
        else:
            console.print("\n[yellow]No user stories defined[/yellow]")
            console.print(f"Use '[cyan]slopometry add-user-story {nfp.id}[/cyan]' to add stories.")

        console.print("\n[bold]Summary[/bold]")
        console.print(f"Total Stories: {nfp.story_count}")
        console.print(f"Total Estimated Complexity: {nfp.total_estimated_complexity}")
        console.print(f"High Priority Stories: {len(nfp.get_high_priority_stories())}")

    except Exception as e:
        console.print(f"[red]Failed to show NFP: {e}[/red]")


@cli.command()
@click.argument("nfp_id", shell_complete=complete_nfp_id)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete_nfp(nfp_id: str, yes: bool):
    """Delete an NFP objective and all its user stories."""
    db = EventDatabase()

    if not yes:
        nfp = db.get_nfp_objective(nfp_id)
        if not nfp:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")
            return

        console.print(f"[yellow]About to delete NFP: {nfp.title}[/yellow]")
        console.print(f"This will delete {nfp.story_count} user stories.")

        if not click.confirm("Are you sure?"):
            console.print("Cancelled")
            return

    try:
        if db.delete_nfp_objective(nfp_id):
            console.print(f"[green]✓ Deleted NFP objective {nfp_id}[/green]")
        else:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")

    except Exception as e:
        console.print(f"[red]Failed to delete NFP: {e}[/red]")


def _check_hooks_installed(settings_file: Path) -> bool:
    """Check if slopometry hooks are installed in a settings file."""
    if not settings_file.exists():
        return False

    try:
        with open(settings_file) as f:
            settings_data = json.load(f)

        hooks = settings_data.get("hooks", {})
        for hook_type in hooks:
            for hook_config in hooks[hook_type]:
                if isinstance(hook_config.get("hooks"), list):
                    for hook in hook_config["hooks"]:
                        command = hook.get("command", "")
                        if "slopometry hook-" in command:
                            return True
        return False
    except (json.JSONDecodeError, KeyError):
        return False


if __name__ == "__main__":
    cli()

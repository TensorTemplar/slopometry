"""CLI for the slopometry Claude Code tracker."""

import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .database import EventDatabase
from .settings import settings

console = Console()

# Save builtin list before Click commands override it
builtin_list = list


def create_slopometry_hooks() -> dict:
    """Create slopometry hook configuration for Claude Code."""
    hook_command = settings.hook_command

    return {
        "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": hook_command}]}],
        "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": hook_command}]}],
        "Notification": [{"hooks": [{"type": "command", "command": hook_command}]}],
        "Stop": [{"hooks": [{"type": "command", "command": hook_command}]}],
        "SubagentStop": [{"hooks": [{"type": "command", "command": hook_command}]}],
    }


@click.group()
def cli():
    """Slopometry - Claude Code session tracker."""
    pass


@cli.command()
@click.option(
    "--global/--local",
    "global_",
    default=False,
    help="Install hooks globally (~/.claude) or locally (./.claude)",
)
def install(global_):
    """Install slopometry hooks for automatic Claude Code tracking."""
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
                isinstance(h.get("hooks"), builtin_list)
                and any("slopometry.hook_handler" in hook.get("command", "") for hook in h.get("hooks", []))
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
    """Remove slopometry hooks to stop automatic tracking."""
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
                isinstance(h.get("hooks"), builtin_list)
                and any("slopometry.hook_handler" in hook.get("command", "") for hook in h.get("hooks", []))
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
def list(limit):
    """List recent Claude Code sessions."""
    db = EventDatabase()
    limit = int(limit) if limit else settings.recent_sessions_limit
    sessions = db.list_sessions()[:limit]

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        console.print("[dim]Run 'slopometry install' to start tracking Claude Code sessions[/dim]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Start Time", style="green")
    table.add_column("Events", justify="right")
    table.add_column("Tools Used", justify="right")

    for session_id in sessions:
        stats = db.get_session_statistics(session_id)
        if stats:
            table.add_row(
                session_id,
                stats.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                str(stats.total_events),
                str(len(stats.tool_usage)),
            )

    console.print(table)


@cli.command()
@click.argument("session_id")
def show(session_id):
    """Show detailed statistics for a session."""
    show_session_summary(session_id, detailed=True)


def show_session_summary(session_id: str, detailed: bool = False):
    """Display session statistics."""
    db = EventDatabase()
    stats = db.get_session_statistics(session_id)

    if not stats:
        console.print(f"[red]No data found for session {session_id}[/red]")
        return

    console.print(f"\n[bold]Session Statistics: {session_id}[/bold]")
    console.print(f"Start: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if stats.end_time:
        duration = (stats.end_time - stats.start_time).total_seconds()
        console.print(f"Duration: {duration:.1f} seconds")
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

    # Display git metrics if available
    if stats.initial_git_state and stats.initial_git_state.is_git_repo:
        console.print("\n[bold]Git Metrics[/bold]")
        console.print(f"Commits made: [green]{stats.commits_made}[/green]")

        if stats.initial_git_state.current_branch:
            console.print(f"Branch: {stats.initial_git_state.current_branch}")

        if stats.initial_git_state.has_uncommitted_changes:
            console.print("[yellow]Had uncommitted changes at start[/yellow]")

        if stats.final_git_state and stats.final_git_state.has_uncommitted_changes:
            console.print("[yellow]Has uncommitted changes at end[/yellow]")

    # Display complexity metrics if available
    if stats.complexity_metrics and stats.complexity_metrics.total_files_analyzed > 0:
        console.print("\n[bold]Complexity Metrics[/bold]")
        console.print(f"Files analyzed: {stats.complexity_metrics.total_files_analyzed}")
        console.print(f"Total complexity: {stats.complexity_metrics.total_complexity}")
        console.print(f"Average complexity: {stats.complexity_metrics.average_complexity:.1f}")
        console.print(f"Max complexity: {stats.complexity_metrics.max_complexity}")
        console.print(f"Min complexity: {stats.complexity_metrics.min_complexity}")
        
        # Show top complex files in a table
        if stats.complexity_metrics.files_by_complexity:
            table = Table(title="Files by Complexity")
            table.add_column("File", style="cyan")
            table.add_column("Complexity", justify="right")
            
            # Sort by complexity (descending) and show top 10
            sorted_files = sorted(
                stats.complexity_metrics.files_by_complexity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for file_path, complexity in sorted_files:
                table.add_row(file_path, str(complexity))
            
            console.print(table)

    # Display complexity delta if available
    if stats.complexity_delta:
        delta = stats.complexity_delta
        console.print("\n[bold]Complexity Delta (vs Previous Commit)[/bold]")
        
        # Overall change summary
        if delta.total_complexity_change != 0:
            change_color = "green" if delta.total_complexity_change < 0 else "red"
            console.print(f"Total complexity change: [{change_color}]{delta.total_complexity_change:+d}[/{change_color}]")
        else:
            console.print("Total complexity change: [yellow]0[/yellow]")
        
        if delta.avg_complexity_change != 0:
            change_color = "green" if delta.avg_complexity_change < 0 else "red"
            console.print(f"Average complexity change: [{change_color}]{delta.avg_complexity_change:+.1f}[/{change_color}]")
        
        # File changes summary
        if delta.net_files_change != 0:
            console.print(f"Net files change: {delta.net_files_change:+d}")
        
        # Show detailed file changes
        if delta.files_added:
            console.print(f"[green]Files added ({len(delta.files_added)})[/green]: {', '.join(delta.files_added[:3])}")
            if len(delta.files_added) > 3:
                console.print(f"  ... and {len(delta.files_added) - 3} more")
        
        if delta.files_removed:
            console.print(f"[red]Files removed ({len(delta.files_removed)})[/red]: {', '.join(delta.files_removed[:3])}")
            if len(delta.files_removed) > 3:
                console.print(f"  ... and {len(delta.files_removed) - 3} more")
        
        if delta.files_changed:
            # Show files with biggest complexity changes
            sorted_changes = sorted(delta.files_changed.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            if sorted_changes:
                table = Table(title="Biggest Complexity Changes")
                table.add_column("File", style="cyan")
                table.add_column("Change", justify="right")
                
                for file_path, change in sorted_changes:
                    change_color = "green" if change < 0 else "red"
                    table.add_row(file_path, f"[{change_color}]{change:+d}[/{change_color}]")
                
                console.print(table)

    # Display plan evolution if available
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
            
            for step in evolution.plan_steps[:5]:  # Show first 5 steps
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

    if detailed:
        events = db.get_session_events(session_id)
        tree = Tree("[bold]Event Sequence[/bold]")

        display_limit = settings.event_display_limit
        for event in events[:display_limit]:
            label = f"{event.sequence_number}. {event.event_type.value}"
            if event.tool_name:
                label += f" - {event.tool_name}"
            if event.duration_ms:
                label += f" ({event.duration_ms}ms)"

            node = tree.add(label)
            if event.error_message:
                node.add(f"[red]Error: {event.error_message}[/red]")

        if len(events) > display_limit:
            tree.add(f"... and {len(events) - display_limit} more events")

        console.print(tree)


@cli.command()
@click.argument("session_id", required=False)
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
        # Check if session exists
        stats = db.get_session_statistics(session_id)
        if not stats:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        # Show session info
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

    else:  # all_sessions
        # Get session count
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
        # Show current setting
        current_status = "enabled" if settings.enable_stop_feedback else "disabled"
        console.print(f"[bold]Complexity feedback is currently {current_status}[/bold]")
        console.print("")
        console.print("To change this setting:")
        console.print("  slopometry feedback --enable    # Enable feedback")
        console.print("  slopometry feedback --disable   # Disable feedback")
        console.print("")
        if not settings.enable_stop_feedback:
            console.print("[yellow]Note: Feedback is disabled by default. Enable it to receive[/yellow]")
            console.print("[yellow]complexity analysis when Claude Code sessions end.[/yellow]")
        return

    # Update setting via environment variable suggestion
    env_file = Path(".env")
    env_var = "SLOPOMETRY_ENABLE_STOP_FEEDBACK"
    
    if enable:
        console.print("[green]Enabling[/green] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=true")
        
        # Update .env file if it exists or create it
        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                # Replace existing line
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=true")
                    else:
                        new_lines.append(line)
                env_file.write_text('\n'.join(new_lines))
            else:
                # Append new line
                with env_file.open('a') as f:
                    f.write(f"\n{env_var}=true\n")
        else:
            # Create new .env file
            env_file.write_text(f"{env_var}=true\n")
            
        console.print(f"[green]Added {env_var}=true to .env file[/green]")
        
    else:
        console.print("[yellow]Disabling[/yellow] complexity feedback on stop events")
        console.print("")
        console.print("To persist this setting, add to your .env file:")
        console.print(f"  {env_var}=false")
        
        # Update .env file
        if env_file.exists():
            content = env_file.read_text()
            if env_var in content:
                # Replace existing line
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith(f"{env_var}="):
                        new_lines.append(f"{env_var}=false")
                    else:
                        new_lines.append(line)
                env_file.write_text('\n'.join(new_lines))
            else:
                # Append new line
                with env_file.open('a') as f:
                    f.write(f"\n{env_var}=false\n")
        else:
            # Create new .env file
            env_file.write_text(f"{env_var}=false\n")
            
        console.print(f"[green]Added {env_var}=false to .env file[/green]")

    console.print("")
    console.print("[bold]Note:[/bold] You may need to restart Claude Code for changes to take effect.")


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
                if isinstance(hook_config.get("hooks"), builtin_list):
                    for hook in hook_config["hooks"]:
                        if "slopometry.hook_handler" in hook.get("command", ""):
                            return True
        return False
    except (json.JSONDecodeError, KeyError):
        return False


if __name__ == "__main__":
    cli()
